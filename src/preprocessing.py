import json
import csv
import torch
import pandas as pd
import numpy as np
from transformers import GPT2Tokenizer, GPT2Model
from tqdm import tqdm

## script to generate a csv file from the train reports.

def generate_labels_csv(json_file, to_csv):
    # Load the JSON data
    with open(json_file, "r") as f:
        data = json.load(f)

    # Write to CSV
    with open(to_csv, mode="w", newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Case_ID", "Organ", "Sample_Type", "Findings"])  # Header

        for item in data:
            case_id = item.get("id", "")
            report = item.get("report", "").strip()

            organ, sample_type, findings = "", "", report

            # Try to split into header and findings
            if ";\n" in report:
                header, findings = report.split(";\n", 1)
                findings = findings.strip()

                # Try to split header into organ and sample type
                if "," in header:
                    organ, sample_type = [x.strip() for x in header.split(",", 1)]
                else:
                    organ = header.strip()

            elif "," in report:  # No ;\n but still structured
                organ, sample_type = [x.strip() for x in report.split(",", 1)]
                findings = ""

            writer.writerow([case_id, organ, sample_type, findings])

    print(f"Label csv created and saved to {to_csv}")

def mean_pool(hidden_states, attention_mask):
    """Mean pooling over non-padded token embeddings"""
    mask = attention_mask.unsqueeze(-1).expand(hidden_states.size())
    masked_hidden = hidden_states * mask
    summed = masked_hidden.sum(1)
    counts = mask.sum(1)
    return summed / counts

def generate_label_embeddings(lbl_csv, save_to, model_name='gpt2', device='cuda' if torch.cuda.is_available() else 'cpu'):
    print(f"Loading labels from: {lbl_csv}")
    df = pd.read_csv(lbl_csv)

    # Check for required columns
    ## ToDo: ideally the Findings_Label column should have been used but keeping this for the current model states
    required_cols = {"Case_ID", "Organ", "Sample_Type", "Findings"}
    if not required_cols.issubset(set(df.columns)):
        raise ValueError(f"Missing columns in label CSV. Required: {required_cols}")

    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    
    # Fix: Add pad_token if missing
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = GPT2Model.from_pretrained(model_name).to(device)
    model.eval()

    embeddings = {}

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Encoding labelss"):
        case_id = row["Case_ID"]
        organ = str(row["Organ"])
        sample_type = str(row["Sample_Type"])
        findings = str(row["Findings"]) ## ToDo: ideally the Findings_Label column should have been used but keeping this for the current model states
    
        label = f"Organ: {organ}. Sample Type: {sample_type}. Findings: {findings}."

        encoded = tokenizer(label, return_tensors='pt', padding=True, truncation=True, max_length=128)
        input_ids = encoded['input_ids'].to(device)
        attention_mask = encoded['attention_mask'].to(device)

        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            pooled = mean_pool(outputs.last_hidden_state, attention_mask)
            embeddings[case_id] = pooled.squeeze(0).cpu()

    torch.save(embeddings, save_to)
    print(f"Saved label embeddings to: {save_to}")

def generate_organ_index(lbl_csv, save_to):
    organ_set = set()

    with open(lbl_csv, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            organ = row["Organ"].strip().lower()
            if organ:
                organ_set.add(organ)

    organ_to_index = {organ: idx for idx, organ in enumerate(sorted(organ_set))}

    with open(save_to, "w") as f:
        json.dump(organ_to_index, f, indent=2)

    print(f"organ_to_index.json created at {save_to} with {len(organ_to_index)} organs.")

if __name__ == "__main__":
    reports_json = "./experiments/misc/train.json" ##"path to the train reports json file"
    label_csv = "./experiments/misc/labels.csv" ## path to save the labels csv file generated from the report json
    generate_labels_csv(reports_json, label_csv)
    ## Note that findings list is refined manually and added as a new column Findings_Label to the generated labels.csv file.

    emb_save_to = "./experiments/misc/prompt_embeddings.pt"
    generate_label_embeddings(label_csv, emb_save_to)

    org_ind_save_to = "./experiments/misc/organ_to_index.json"
    generate_organ_index(label_csv, org_ind_save_to)
