# model.py
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from config import *

class BioBARTPromptModel(nn.Module):
    """
    Visual-prefix based prompt model for BioBART.
    Projects WSI features -> prefix token embeddings and prepends them to LM input embeddings.
    Returns (lm_loss, lm_logits, organ_logits, sample_logits, finding_logits).
    Auxiliary losses are NOT computed inside the model (train.py does that).
    """

    def __init__(
        self,
        model_name: str = MODEL_NAME,
        prefix_tokens: int = PREFIX_LENGTH,
        visual_feature_dim: int = PREFIX_DIM,
        num_organs: int = NUM_ORGANS
    ):
        super().__init__()

        # tokenizer + base seq2seq LM (we will *not* use PEFT wrapper here to avoid runtime wrapper issues)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        base_model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

        # Freeze LM parameters (we only train the visual projector and classification heads)
        for p in base_model.parameters():
            p.requires_grad = False

        self.model = base_model

        # prefix / projection config
        self.prefix_tokens = prefix_tokens
        self.d_model = self.model.config.d_model

        # visual features -> prefix embeddings projector
        self.visual_projector = nn.Sequential(
            nn.Linear(visual_feature_dim, 1024),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(1024, self.prefix_tokens * self.d_model),
            nn.LayerNorm(self.prefix_tokens * self.d_model),
        )

        # classification heads (returned as logits; losses are computed by train.py)
        self.use_org = USE_ORGAN_LOSS
        self.organ_head = nn.Linear(visual_feature_dim, num_organs) if self.use_org else None

        self.use_sample = USE_SAMPLE_LOSS
        self.sample_head = nn.Linear(visual_feature_dim, NUM_SAMPLE_TYPES) if self.use_sample else None

        self.use_finding = USE_FINDING_LOSS
        self.finding_head = nn.Linear(visual_feature_dim, NUM_FINDINGS) if self.use_finding else None

    def _move_tokenizer_outputs_to_device(self, tensor_dict, device):
        return {k: v.to(device) for k, v in tensor_dict.items()}

    def forward(
        self,
        features=None,               # WSI features (preferred)
        prompt_embedding=None,       # fallback if features is not provided
        organ_idx=None,
        sample_idx=None,
        finding_idx=None,
        input_texts=None,            # list of strings, optional (prefix/prompts)
        labels=None                  # list of ground-truth reports (strings) OR token ids
    ):
        """
        If `labels` is provided, compute LM loss; otherwise lm_loss = 0.0.
        Returns: (lm_loss, lm_logits, organ_logits, sample_logits, finding_logits)
        """

        device = None
        # determine device based on provided features or prompt_embedding
        if isinstance(features, torch.Tensor):
            device = features.device
        elif isinstance(prompt_embedding, torch.Tensor):
            device = prompt_embedding.device
        else:
            # fallback: CPU - but training expects tensors on GPU
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # choose visual input: prefer features (WSI), otherwise prompt_embedding (precomputed)
        visual_input = None
        if features is not None:
            visual_input = features
        elif prompt_embedding is not None:
            visual_input = prompt_embedding
        else:
            visual_input = None

        # prepare LM loss (only if labels provided)
        lm_loss = torch.tensor(0.0, device=device)
        lm_logits = None

        if labels is not None:
            # default prompt text if not provided
            if input_texts is None:
                input_texts = ["Pathology report:" for _ in labels]

            # Tokenize inputs
            inputs = self.tokenizer(
                input_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=64
            )
            inputs = self._move_tokenizer_outputs_to_device(inputs, device)
            input_ids = inputs["input_ids"]
            attention_mask = inputs["attention_mask"]

            # compute prefix embeddings using visual input (must exist)
            if visual_input is None:
                raise ValueError("visual features (features or prompt_embedding) must be provided when computing LM loss")

            prefix = self.visual_projector(visual_input)  # [B, prefix_tokens * d_model]
            batch_size = input_ids.size(0)
            prefix = prefix.view(batch_size, self.prefix_tokens, self.d_model)  # [B, T, D]

            # get input token embeddings and concat
            with torch.no_grad():  # embeddings are from frozen LM -> no grad needed here
                input_embeds = self.model.get_input_embeddings()(input_ids)  # [B, L, D]

            inputs_embeds = torch.cat([prefix, input_embeds], dim=1)  # [B, T+L, D]

            # attention mask needs prefix ones
            prefix_mask = torch.ones(batch_size, self.prefix_tokens, device=device, dtype=attention_mask.dtype)
            full_attention_mask = torch.cat([prefix_mask, attention_mask], dim=1)  # [B, T+L]

            # prepare labels -> token ids with pad -> -100
            token_labels = self.tokenizer(
                labels,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=128
            ).input_ids
            token_labels[token_labels == self.tokenizer.pad_token_id] = -100
            token_labels = token_labels.to(device)

            # call base model with inputs_embeds (do not pass input_ids)
            outputs = self.model(
                input_ids=None,
                inputs_embeds=inputs_embeds,
                attention_mask=full_attention_mask,
                labels=token_labels,
                return_dict=True
            )
            lm_loss = outputs.loss if outputs.loss is not None else torch.tensor(0.0, device=device)
            lm_logits = outputs.logits  # [B, T_dec, V]

        # compute auxiliary logits (do not compute their loss here — train.py does it)
        organ_logits = self.organ_head(features) if (self.use_org and features is not None) else None
        sample_logits = self.sample_head(features) if (self.use_sample and features is not None) else None
        finding_logits = self.finding_head(features) if (self.use_finding and features is not None) else None

        return lm_loss, lm_logits, organ_logits, sample_logits, finding_logits

    def generate(
        self,
        input_texts,
        visual_feats,
        max_length: int = 128,
        num_beams: int = 4,
        length_penalty: float = 1.0,
        repetition_penalty: float = 1.0,
        early_stopping: bool = True
    ):
        """
        Generate reports conditioned on visual_feats by prepending visual prefix embeddings.
        """

        self.eval()
        device = visual_feats.device

        # Tokenize text prompts
        inputs = self.tokenizer(
            input_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=64
        )
        inputs = self._move_tokenizer_outputs_to_device(inputs, device)
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        batch_size = input_ids.size(0)

        # visual prefix
        prefix = self.visual_projector(visual_feats)  # [B, T * D]
        prefix = prefix.view(batch_size, self.prefix_tokens, self.d_model)  # [B, T, D]

        # input token embeddings
        with torch.no_grad():
            input_embeds = self.model.get_input_embeddings()(input_ids)  # [B, L, D]

        inputs_embeds = torch.cat([prefix, input_embeds], dim=1)  # [B, T+L, D]
        prefix_mask = torch.ones(batch_size, self.prefix_tokens, device=device, dtype=attention_mask.dtype)
        full_attention_mask = torch.cat([prefix_mask, attention_mask], dim=1)  # [B, T+L]

        # call generate with inputs_embeds
        generated_ids = self.model.generate(
            input_ids=None,
            inputs_embeds=inputs_embeds,
            attention_mask=full_attention_mask,
            max_length=max_length,
            num_beams=num_beams,
            length_penalty=length_penalty,
            repetition_penalty=repetition_penalty,
            early_stopping=early_stopping,
            no_repeat_ngram_size=3,
        )

        reports = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        reports = [r.replace(";", "; ").replace("  ", " ").strip() for r in reports]
        return reports
