import os
import json
import pandas as pd
import torch
import h5py


from config import *

class ReportDataset(torch.utils.data.Dataset):
    def __init__(self, selected_ids=None, mode="train", scaler=None):
        self.mode = mode
        self.organ2idx = ORGAN_MAP if ORGAN_MAP is not None else {}
        self.scaler = scaler

        self.data = []
        if self.mode != "test" and REPORTS_JSON is not None:
            with open(REPORTS_JSON, "r") as f:
                self.data = json.load(f)

        corrupted_df = pd.read_csv(CORRUPTED_IDS_CSV)
        corrupted_ids = set(corrupted_df['Case_ID'].str.replace('.tiff', '', regex=False))

        feature_files = [f.replace('.h5', '') for f in os.listdir(FEATS_DIR) if f.endswith('.h5')]
        feature_files_set = set(feature_files)
        
        # Load organ info
        label_map = {}
        if LABEL_CSV is not None:
            label_df = pd.read_csv(LABEL_CSV)
            label_map = {
                row['Case_ID'].replace('.tiff', ''): str(row['Organ']).strip().lower()
                for _, row in label_df.iterrows()
            }

        if self.mode == "test":
            feature_ids = feature_files
            if selected_ids is not None:
                selected_set = set(selected_ids)
                feature_ids = [fid for fid in feature_files if fid in selected_set]

            for fid in feature_ids:
                if fid in corrupted_ids:
                    continue
                self.data.append({
                    "id": fid,
                    "organ": label_map.get(fid, "")
                })
        else:
            filtered_data = []
            for item in self.data:
                sample_id = item['id'].replace('.tiff', '')
                if sample_id in corrupted_ids or sample_id not in feature_files_set:
                    continue
                item['id'] = sample_id
                item['organ'] = label_map.get(sample_id, "")
                filtered_data.append(item)

            if selected_ids is not None:
                selected_set = set(selected_ids)
                filtered_data = [item for item in filtered_data if item['id'] in selected_set]

            self.data = filtered_data

        # Load label embeddings if needed
        self.label_embeddings = {}
        if USE_LABEL_EMBEDDINGS and LABEL_EMBEDDING_PATH and os.path.exists(LABEL_EMBEDDING_PATH):
            self.label_embeddings = torch.load(LABEL_EMBEDDING_PATH)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        sample_id = item['id']

        feature_path = os.path.join(FEATS_DIR, f"{sample_id}.h5")
        with h5py.File(feature_path, 'r') as f:
            features = f['features'][()]

        if self.scaler is not None:
            features = self.scaler.transform(features.reshape(1, -1)).squeeze()

        output = {
            'id': sample_id,
            'features': torch.tensor(features, dtype=torch.float32),
            'organ': item.get('organ', '')
        }

        if self.mode != "test":
            output['report'] = item.get('report', '')

            if USE_ORGAN_LOSS:
                organ_name = item.get('organ', '').lower()
                organ_idx = self.organ2idx.get(organ_name, -1)
                output['organ_idx'] = torch.tensor(organ_idx, dtype=torch.long)

            if USE_SAMPLE_LOSS:
                sample_idx = SAMPLE_IDX_MAP.get(sample_id, -1)
                if sample_idx == -1:
                    print(f"[WARN] Sample ID '{sample_id}' has unknown sample type.")
                output['sample_idx'] = torch.tensor(sample_idx, dtype=torch.long)

            if USE_FINDING_LOSS:
                finding_idx = FINDING_MAP.get(sample_id, -1)
                if finding_idx == -1:
                    print(f"[WARN] Sample ID '{sample_id}' has unknown finding.")
                output['finding_idx'] = torch.tensor(finding_idx, dtype=torch.long)

            if USE_LABEL_EMBEDDINGS and sample_id in self.label_embeddings:
                output['label_embedding'] = torch.tensor(
                    self.label_embeddings[sample_id], dtype=torch.float32
                )

        return output
