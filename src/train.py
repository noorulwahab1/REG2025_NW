import os
import json
import glob
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup

from evaluator import REG_Evaluator
from model import BioBARTPromptModel
from dataset import ReportDataset
from config import *

torch.backends.cudnn.benchmark = True

# -----------------------------
# Extra training config
# -----------------------------

def get_trainable_params(model):
    """Return only trainable parameters (prefix, visual projection, organ head)."""
    return [p for p in model.parameters() if p.requires_grad]

def overfit_single_batch(model, train_loader, device="cuda", lr=1e-6, max_steps=10):
    model.train()
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.eval()
    model.to(device)

    batch = next(iter(train_loader))

    # === Filter invalid labels ===
    valid_mask = (
        (batch['organ_idx'] != -1)
        & (batch['sample_idx'] != -1)
        & (batch['finding_idx'] != -1)
    )

    if valid_mask.sum() == 0:
        print("[WARN] Skipping batch: all labels are invalid.")
        return False

    for key in ['features', 'organ_idx', 'sample_idx', 'finding_idx']:
        if key in batch and isinstance(batch[key], torch.Tensor):
            batch[key] = batch[key][valid_mask]

    batch['report'] = [r for i, r in enumerate(batch['report']) if valid_mask[i]]

    optimizer = torch.optim.Adam(get_trainable_params(model), lr=lr)
    aux_loss_fn = nn.CrossEntropyLoss(ignore_index=-1)

    losses = []
    for step in range(1, max_steps + 1):
        optimizer.zero_grad()

        #print("Batch keys are:", batch.keys())

        # use actual reports as labels and provide a short prompt
        input_prompts = ["Pathology report:" for _ in batch["report"]]
        loss, _, organ_logits, sample_logits, finding_logits = model(
            features=batch["features"].to(device),
            label_embedding=batch.get("label_embedding", None).to(device) if batch.get("label_embedding", None) is not None else None,
            input_texts=input_prompts,
            labels=batch["report"],   # list of strings (reports)
        )

        # === Organ loss ===
        if USE_ORGAN_LOSS and organ_logits is not None and 'organ_idx' in batch:
            organ_ids = batch['organ_idx'].to(device)
            org_loss = aux_loss_fn(organ_logits, organ_ids)
            loss += ORGAN_LOSS_WEIGHT * org_loss
        else:
            org_loss = torch.tensor(0.0, device=device)

        # === Sample loss ===
        if USE_SAMPLE_LOSS and sample_logits is not None and 'sample_idx' in batch:
            sample_ids = batch['sample_idx'].to(device)
            sample_loss = aux_loss_fn(sample_logits, sample_ids)
            loss += SAMPLE_LOSS_WEIGHT * sample_loss
        else:
            sample_loss = torch.tensor(0.0, device=device)

        # === Finding loss ===
        if USE_FINDING_LOSS and finding_logits is not None and 'finding_idx' in batch:
            finding_ids = batch['finding_idx'].to(device)
            finding_loss = aux_loss_fn(finding_logits, finding_ids)
            loss += FINDING_LOSS_WEIGHT * finding_loss
        else:
            finding_loss = torch.tensor(0.0, device=device)

        if torch.isnan(loss):
            print("NaN loss encountered. Stopping.")
            break

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        losses.append(loss.item())
        print(f"[Overfit Step {step}] Loss = {loss.item():.4f} "
              f"(Org={org_loss.item():.4f} | Samp={sample_loss.item():.4f} | Find={finding_loss.item():.4f})")

    if losses[0] > losses[-1]:
        print(f"[INFO] Overfit success! Loss decreased from {losses[0]:.4f} → {losses[-1]:.4f}")
        return True
    else:
        print(f"[WARN] Overfit failed. Loss did not decrease.")
        return False

def train_one_fold(train_data, val_data, fold):
    print(f"\n--- Fold {fold + 1}/{K_FOLDS} ---")

    model = BioBARTPromptModel().to(DEVICE)
    
    optimizer = AdamW(get_trainable_params(model), lr=LR, weight_decay=0.01)
    total_steps = (len(train_data) // BATCH_SIZE) * EPOCHS

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=100,
        num_training_steps=total_steps
    )

    aux_loss_fn = nn.CrossEntropyLoss(ignore_index=-1)

    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=BATCH_SIZE)

    evaluator = REG_Evaluator(MODEL_NAME)
    #evaluator = REG_Evaluator(LM_MODEL)

    #success = overfit_single_batch(model, train_loader, device=DEVICE)
    success = overfit_single_batch(model, train_loader, device=DEVICE, lr=1e-4, max_steps=30)

    if not success:
        print("[WARNING] Single-batch overfit failed. Skipping this fold to save time.")
        return -float("inf")
    print("[INFO] Single-batch overfit passed. Starting full training...\n")

    train_losses = []
    best_score = -float("inf")
    best_loss =  -float("inf")
    best_model_state = None
    patience_counter = 0

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0
        total_organ_loss = 0.0
        total_sample_loss = 0.0
        total_finding_loss = 0.0

        for step, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}")):
            visual_feats = batch['features'].to(DEVICE)
            reports = list(batch['report'])

            organs = batch['organ']
            organ_ids = batch.get('organ_idx', None)
            if organ_ids is not None:
                organ_ids = organ_ids.to(DEVICE)

            prompts = ["Pathology report:" if np.random.rand() > 0.2 else "" for _ in organs]

            optimizer.zero_grad()
            with torch.amp.autocast(device_type="cuda", enabled=USE_AMP):
                prompts = ["Pathology report:" if np.random.rand() > 0.2 else "" for _ in organs]
                loss, _, organ_logits, sample_logits, finding_logits = model(
                    features=batch["features"].to(DEVICE),
                    label_embedding=(batch.get("label_embedding", None).to(DEVICE) if batch.get("label_embedding", None) is not None else None),
                    input_texts=prompts,
                    labels=list(batch["report"]),   # list of strings
                )

                # Default losses
                org_loss = torch.tensor(0.0, device=DEVICE)
                sample_loss = torch.tensor(0.0, device=DEVICE)
                finding_loss = torch.tensor(0.0, device=DEVICE)

                # Organ loss
                if USE_ORGAN_LOSS and organ_logits is not None and 'organ_idx' in batch:
                    organ_ids = batch['organ_idx'].to(DEVICE)
                    if (organ_ids != -1).any():
                        org_loss = aux_loss_fn(organ_logits, organ_ids)
                        loss += ORGAN_LOSS_WEIGHT * org_loss
                    else:
                        org_loss = torch.tensor(0.0, device=DEVICE)
                else:
                    org_loss = torch.tensor(0.0, device=DEVICE)

                # Sample loss
                if USE_SAMPLE_LOSS and sample_logits is not None and 'sample_idx' in batch:
                    sample_ids = batch['sample_idx'].to(DEVICE)
                    if (sample_ids != -1).any():
                        sample_loss = aux_loss_fn(sample_logits, sample_ids)
                        loss += SAMPLE_LOSS_WEIGHT * sample_loss
                    else:
                        sample_loss = torch.tensor(0.0, device=DEVICE)
                else:
                    sample_loss = torch.tensor(0.0, device=DEVICE)

                # Finding loss
                if USE_FINDING_LOSS and finding_logits is not None and 'finding_idx' in batch:
                    finding_ids = batch['finding_idx'].to(DEVICE)
                    if (finding_ids != -1).any():
                        finding_loss = aux_loss_fn(finding_logits, finding_ids)
                        loss += FINDING_LOSS_WEIGHT * finding_loss
                    else:
                        finding_loss = torch.tensor(0.0, device=DEVICE)
                else:
                    finding_loss = torch.tensor(0.0, device=DEVICE)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(get_trainable_params(model), 1.0)
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()
            total_organ_loss += org_loss.item()
            total_sample_loss += sample_loss.item()
            total_finding_loss += finding_loss.item()

            # if epoch % PRINT_EVERY == 0:
            #     #print(f"[Epoch {epoch+1} | Step {step}] Loss = {loss.item():.4f}")
            #     print(f"[Epoch {epoch+1}] Total Loss = {loss.item():.4f} "
            #       f"(Org={aux_loss.item():.4f} | Samp={sample_loss.item():.4f} | Find={finding_loss.item():.4f})")

        avg_train_loss = total_loss / len(train_loader)
        avg_organ_loss = total_organ_loss / len(train_loader)
        avg_sample_loss = total_sample_loss / len(train_loader)
        avg_finding_loss = total_finding_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        #print(f"Epoch {epoch+1} Average Train Loss: {avg_train_loss:.4f}")
        if epoch % PRINT_EVERY == 0:
                #print(f"[Epoch {epoch+1} | Step {step}] Loss = {loss.item():.4f}")
                print(f"[Epoch {epoch+1}] Average Losses: Total Loss = {avg_train_loss:.4f} "
                  f"(Organ={avg_organ_loss:.4f} | Sample={avg_sample_loss:.4f} | Find={avg_finding_loss:.4f})")

        # ------------------------- Validation -------------------------

        if (epoch + 1) % VAL_AFTER_EPOCH != 0:
            continue

        print(f"\n=== Running Validation after Epoch {epoch+1} ===")
        model.eval()
        all_preds, all_refs = [], []

        with torch.no_grad(), torch.amp.autocast(device_type="cuda", enabled=USE_AMP):
            for batch in tqdm(val_loader, desc="Validation"):
                visual_feats = batch['features'].to(DEVICE)
                reports = batch['report']
                prompts = ["Pathology report:" for _ in reports]
                generated = model.generate(prompts, visual_feats)
                preds = [p.strip().replace(";", "; ") for p in generated]
                all_preds.extend(preds)
                all_refs.extend(list(reports))

        eval_pairs = list(zip(all_refs, all_preds))
        score = evaluator.evaluate_dummy(eval_pairs)
        print(f"[Validation] Score: {score:.4f}")

        # print("=== Sample Predictions ===")
        # for ref, pred in list(zip(all_refs, all_preds))[:3]:
        #     print(f"GT: {ref}\nPRED: {pred}\n")
        
        if score > best_score:
        #if best_loss:
            best_score = score
            #best_loss = avg_train_loss
            best_model_state = {k: v.cpu() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            print(f"No improvement. Patience: {patience_counter}/{PATIENCE}")

        if patience_counter >= PATIENCE:
            print("Early stopping triggered.")
            break

    # Save best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        os.makedirs(GLOBAL_DIR, exist_ok=True)
        save_path = os.path.join(GLOBAL_DIR, f"best_model_fold{fold}.pt")
        torch.save(model.state_dict(), save_path)
        print(f"Best model saved to {save_path}")
    else:
        print("Warning: No best model found, skipping save.")

    # Plot train loss
    plt.figure()
    plt.plot(range(1, len(train_losses) + 1), train_losses, label='Train Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'Training Fold {fold+1}')
    plt.legend()
    plot_path = os.path.join(GLOBAL_DIR, f"loss_plot_fold{fold+1}.png")
    plt.savefig(plot_path)
    plt.close()
    print(f"Loss plot saved to {plot_path}")

    return best_score

def kfold_training():
    fold_paths = sorted(glob.glob(os.path.join(FOLDS_PATH, "fold_*.csv")))
    all_scores = []
    fold_dfs = [pd.read_csv(f) for f in fold_paths]
    

    for fold in range(K_FOLDS):
        print(f"\n Preparing Fold {fold + 1}/{K_FOLDS}")

        val_df = fold_dfs[fold]
        val_ids = val_df["Case_ID"].str.replace(".tiff", "", regex=False).tolist()

        train_dfs = [df for i, df in enumerate(fold_dfs) if i != fold]
        train_df = pd.concat(train_dfs, ignore_index=True)
        train_ids = train_df["Case_ID"].str.replace(".tiff", "", regex=False).tolist()

        # Collect features for scaling
        print(" Loading training features for scaling...")
        temp_dataset = ReportDataset(selected_ids=train_ids)

        all_feats = []
        for i in range(len(temp_dataset)):
            try:
                item = temp_dataset[i]
                all_feats.append(item['features'])
            except Exception as e:
                print('Error loading dataset')
                print('Error detail: ', e)
                #continue
                exit()

        all_feats = np.vstack(all_feats)
        scaler = StandardScaler()
        scaler.fit(all_feats)
        
        # Save scaler
        os.makedirs(GLOBAL_DIR, exist_ok=True)
        scaler_path = os.path.join(GLOBAL_DIR, f"scaler_fold{fold}.pkl")
        joblib.dump(scaler, scaler_path)
        print(f" Scaler saved to {scaler_path}")

        # Final train/val datasets
        train_data = ReportDataset(selected_ids=train_ids, scaler=scaler)
        val_data = ReportDataset(selected_ids=val_ids, scaler=scaler)
        
        fold_score = train_one_fold(train_data, val_data, fold)
        all_scores.append(fold_score)

    avg_score = sum(all_scores) / len(all_scores)
    print(f"\nAverage Validation Score Across {K_FOLDS} Folds: {avg_score:.4f}")

    scores_dict = {f"fold{idx}": float(score) for idx, score in enumerate(all_scores)}
    score_path = os.path.join(GLOBAL_DIR, "fold_scores.json")
    with open(score_path, "w") as f:
        json.dump(scores_dict, f, indent=2)
    print(f"[INFO] Fold validation scores saved to {score_path}")


if __name__ == "__main__":
    ## ==== Step 1: WSI embeddings ====
    # a) Use Trident (https://github.com/mahmoodlab/TRIDENT) to generate WSI embeddings as below:
    # b) python run_batch_of_slides.py --task all --wsi_dir /Reg2025/train/ --job_dir /Reg2025/titan_wsi_rep --slide_encoder titan --mag 20 --patch_size 1024 --max_workers=32
    # c) Put .h5 files in the corresponding train and test folder at ./WSI_features/

    print(f"==== Step 2: generating a csv file from the train reports and embeddings for auxilary tasks ====")
    ## run preprocessing.py for Step 1
    ## ## Note that findings list is refined manually and added as a new column Findings_Label to the generated labels.csv file.

    print(f"==== Step 3: running {K_FOLDS}-folds training ====")
    kfold_training()
