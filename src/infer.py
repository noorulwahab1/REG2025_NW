import os
import json
import torch
import joblib
from torch.utils.data import DataLoader
from tqdm import tqdm
import pandas as pd
import numpy as np
from collections import Counter
from difflib import SequenceMatcher

from dataset import ReportDataset
from model import BioBARTPromptModel
from config import *
from evaluator import REG_Evaluator

def evaluate_submission(results, ground_truth_path):
    """Evaluate generated reports against ground truth JSON."""
    with open(ground_truth_path) as f:
        gt_raw = json.load(f)
    gt_data = {x["id"].replace(".tiff", ""): x["report"] for x in gt_raw}

    preds = []
    missing = []
    for x in results:
        key = x["id"].replace(".tiff", "")
        if key in gt_data:
            preds.append((gt_data[key], x["report"]))
        else:
            missing.append(x["id"])

    print(f"Found {len(preds)}/{len(results)} matching reports for evaluation.")
    if missing:
        print(f"[WARN] {len(missing)} predictions had no GT match. Sample: {missing[:5]}")

    #evaluator = REG_Evaluator("aaditya/Llama3-OpenBioLLM-8B")  #######@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ commented out the score generation via BioLLM as it was not used in training but the challenge eval will be done by BioLLM
    evaluator = REG_Evaluator(MODEL_NAME)
    score = evaluator.evaluate_dummy(preds)
    print(f" Evaluation Score: {score:.4f}")


def fuzzy_majority_vote(outputs):
    """
    Improved ensemble voting:
    Groups similar strings using fuzzy matching and picks the most common cluster.
    """
    if len(outputs) == 1:
        return outputs[0]

    # Simple clustering by string similarity
    clusters = []
    for out in outputs:
        placed = False
        for cluster in clusters:
            if SequenceMatcher(None, out, cluster[0]).ratio() > 0.8:  # 80% similar
                cluster.append(out)
                placed = True
                break
        if not placed:
            clusters.append([out])

    # Pick largest cluster, break ties by majority
    clusters.sort(key=len, reverse=True)
    majority_cluster = clusters[0]
    vote = Counter(majority_cluster).most_common(1)[0][0]
    return vote


def generate_reports(models, dataset, scalers=None):
    """
    Generate reports using one or multiple models.
    Applies fold-specific scaling if scalers list is provided.
    """
    for m in models:
        m.eval()
        m.to(DEVICE)

    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)
    results = []

    for batch in tqdm(dataloader, desc="Inference"):
        base_feats = batch["features"].cpu().numpy()  # original features
        case_ids = batch["id"]
        organs = batch["organ"]
        
        prompts = ["Pathology report:" if np.random.rand() > 0.2 else "" for _ in organs]


        all_outputs = []

        # Iterate over models (each with its own scaler)
        for idx, model in enumerate(models):
            feats = base_feats.copy()
            if scalers and scalers[idx] is not None:
                feats = scalers[idx].transform(feats)

            feats_tensor = torch.tensor(feats, dtype=torch.float32).to(DEVICE)

            with torch.no_grad():
                output = model.generate(prompts, feats_tensor)
                all_outputs.append([x.strip() for x in output])

        # Ensemble voting
        if len(models) == 1:
            final_outputs = all_outputs[0]
        else:
            final_outputs = []
            for i in range(len(case_ids)):
                votes = [out[i] for out in all_outputs]
                if USE_ENSEMBLE:
                    final_outputs.append(fuzzy_majority_vote(votes))
                else:
                    final_outputs.append(Counter(votes).most_common(1)[0][0])

        for case_id, report in zip(case_ids, final_outputs):
            results.append({"id": case_id + ".tiff", "report": report})

    return results


def load_models_and_scalers():
    """Load trained models and their feature scalers."""
    models, scalers = [], []

    if USE_ENSEMBLE:
        print("Using ensemble of all folds.")
        for i in range(K_FOLDS):
            model_path = os.path.join(GLOBAL_DIR, f"best_model_fold{i}.pt")
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Missing model checkpoint: {model_path}")

            model = BioBARTPromptModel()
            model.load_state_dict(torch.load(model_path, map_location=DEVICE))
            models.append(model)

            scaler_path = os.path.join(GLOBAL_DIR, f"scaler_fold{i}.pkl")
            if os.path.exists(scaler_path):
                scalers.append(joblib.load(scaler_path))
                print(f"Loaded scaler for fold {i}")
            else:
                scalers.append(None)
                print(f"[WARN] No scaler for fold {i}, proceeding unscaled.")

    else:
        # Select best fold dynamically
        score_file = os.path.join(GLOBAL_DIR, "fold_scores.json")
        if not os.path.exists(score_file):
            raise FileNotFoundError("Missing fold_scores.json for non-ensemble mode")

        with open(score_file) as f:
            scores = json.load(f)
        best_fold = max(scores, key=scores.get)  # e.g., "fold0"
        fold_index = int(best_fold.replace("fold", ""))
        print(f"Using best fold: {best_fold} (score: {scores[best_fold]})")

        model_path = os.path.join(GLOBAL_DIR, f"best_model_fold{fold_index}.pt")
        model = BioBARTPromptModel()
        model.load_state_dict(torch.load(model_path, map_location=DEVICE))
        models = [model]

        # Load scaler
        scaler_path = os.path.join(GLOBAL_DIR, f"scaler_fold{fold_index}.pkl")
        if os.path.exists(scaler_path):
            scalers = [joblib.load(scaler_path)]
            print(f"Loaded best-fold scaler: {scaler_path}")
        else:
            scalers = [None]
            print("[WARN] No scaler found for best fold, proceeding unscaled.")

    return models, scalers


if __name__ == "__main__":
    os.makedirs(GLOBAL_DIR, exist_ok=True)
    print(f"Loading models from {GLOBAL_DIR}")

    # Load models & scalers
    models, scalers = load_models_and_scalers()

    # Load test IDs
    test_csv = os.path.join(FOLDS_PATH, "test_split.csv")
    if not os.path.exists(test_csv):
        raise FileNotFoundError(f"Test split file not found: {test_csv}")

    test_df = pd.read_csv(test_csv)
    test_ids = test_df["Case_ID"].str.replace(".tiff", "", regex=False).tolist()
    print(f"Loaded {len(test_ids)} test IDs from test_split.csv")

    # Use first scaler by default for now (multi-scaler per fold optional)
    scaler_to_use = scalers[0]

    ##### temp code for loading the challenge Test cases for inference ###@@@@@@ START @@@@@@@@@@
    test_ids_list = os.listdir("./WSI_features/test2/")
    test_ids = []
    for t in test_ids_list:
        test_ids.append(t.split('.tiff')[0])
    ##### temp code for loading the challenge Test cases for inference ###@@@@@@@ END @@@@@@@@@

    test_dataset = ReportDataset(selected_ids=test_ids, scaler=scaler_to_use, mode="test")

    results = generate_reports(models, test_dataset, scalers)

    output_path = os.path.join(GLOBAL_DIR, "inference_results.json")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Inference complete. Reports saved to {output_path}")

    evaluate_submission(results, REPORTS_JSON)
