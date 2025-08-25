import os
import torch
import json
import pandas as pd

def normalize_loss_weights(organ, sample, finding):
    """
    Normalize three loss weights so they sum to 1.
    Returns: (organ_loss, sample_loss, finding_loss)
    """
    total = organ + sample + finding
    if total == 0:
        raise ValueError("Sum of weights is zero; cannot normalize.")
    factor = 1.0 / total
    organ_loss = organ * factor
    sample_loss = sample * factor
    finding_loss = finding * factor
    return organ_loss, sample_loss, finding_loss

USE_LABEL_EMBEDDINGS = True
USE_ORGAN_LOSS = True
USE_SAMPLE_LOSS = True
USE_FINDING_LOSS = True
ORGAN_LOSS_WEIGHT, SAMPLE_LOSS_WEIGHT, FINDING_LOSS_WEIGHT = normalize_loss_weights(3, 3, 6)

# === Data paths ===
FEATS_DIR = "./WSI_features/train/"
REPORTS_JSON = "./experiments/misc/train.json"
LABEL_CSV = "./experiments/misc/labels.csv"
CORRUPTED_IDS_CSV = "./experiments/misc/corrupted_id.csv"
ORGAN_TO_INDEX_JSON = "./experiments/misc/organ_to_index.json"
LABEL_EMBEDDING_PATH = "./experiments/misc/prompt_embeddings.pt"
FOLDS_PATH = "./experiments/folds/"

EPOCHS = 141  ###@@@@@@@@@@@@@@@@@@@@@@@
PATIENCE = 20
PREFIX_DIM = 768

MODEL_NAME = "GanjinZero/BioBART-Base" #"gpt2" ## this is used instead of "aaditya/Llama3-OpenBioLLM-8B" which the challenge will use for generating the score for evaluation
K_FOLDS = 5
PREFIX_LENGTH = 150        # number of prefix tokens
DROPOUT = 0.5
BATCH_SIZE = 32
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

LR = 1e-4
# Enable or disable automatic mixed precision
USE_AMP = True  # or False
VAL_AFTER_EPOCH = 5 ## validate only after this number of train epochs  ######@@@@@@@@@@@@@@@@@@@@@@@@

PRINT_EVERY = 5        # print loss every 5 epochs  ######@@@@@@@@@@@@@@@@@@@@@@@@

# === Findings (classification head) ===
label_df = pd.read_csv(LABEL_CSV)

FINDING_LIST = sorted(label_df["Findings_Label"].dropna().unique().tolist())
FINDING2IDX = {f: i for i, f in enumerate(FINDING_LIST)}
NUM_FINDINGS = len(FINDING_LIST)

# === Organ map ===
ORGAN_MAP = {}
NUM_ORGANS = 15
if os.path.exists(ORGAN_TO_INDEX_JSON):
    with open(ORGAN_TO_INDEX_JSON, "r") as f:
        ORGAN_MAP = json.load(f)
        NUM_ORGANS = len(ORGAN_MAP)

# === Sample Type maps ===
SAMPLE_TYPE_LIST = sorted(label_df["Sample_Type"].dropna().unique().tolist())
SAMPLE_TYPE2IDX = {s.lower(): i for i, s in enumerate(SAMPLE_TYPE_LIST)}
NUM_SAMPLE_TYPES = len(SAMPLE_TYPE2IDX)

SAMPLE_TYPE_MAP = {
    row["Case_ID"].replace(".tiff", ""): row["Sample_Type"].strip().lower()
    for _, row in label_df.iterrows()
    if pd.notna(row["Sample_Type"])
}

SAMPLE_IDX_MAP = {
    k: SAMPLE_TYPE2IDX.get(v.lower(), -1)
    for k, v in SAMPLE_TYPE_MAP.items()
}

# === Finding maps ===
FINDING_MAP = {
    row["Case_ID"].replace(".tiff", ""): FINDING2IDX.get(row["Findings_Label"], -1)
    for _, row in label_df.iterrows()
    if pd.notna(row["Findings_Label"])
}

## === Inference ===
USE_ENSEMBLE = True

GLOBAL_DIR = f"./experiments/results/MODEL_{MODEL_NAME.split('/')[1]}_PREFIX_{PREFIX_LENGTH}_FOLDS_{K_FOLDS}_EPOCHS_{EPOCHS}/"
