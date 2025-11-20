import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModel
import pandas as pd
import numpy as np
from tqdm import tqdm
from glob import glob
from pathlib import Path
from multiprocessing import Pool, set_start_method
import os
import argparse

from rdkit import Chem


# ───── argparse ──────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=1024)
parser.add_argument('--workers', type=int, default=16)
parser.add_argument('--pretrain', default=True, type=lambda s: s in ['True', 'true', 1])
args = parser.parse_args()



root = './data'
source = os.path.join(root, 'polyone')
dest = os.path.join(root, 'polyone_tokenized')
Path(dest).mkdir(parents=True, exist_ok=True)


prop_list = ['Tg', 'Tm', 'Td',
             'Cp', 'Eat', 'LOI', 'Xc', 'Xe', 'rho',
             'Egc', 'Egb', 'Eea', 'Ei', 'Eib', 'CED',
             'YM', 'TSy', 'TSb', 'epsb',
             'permO2', 'permCO2', 'permN2', 'permH2', 'permHe', 'permCH4',
             'nc', 'ne', 'epsc', 'epse_6.0', 'epse_1.78', 'epse_2.0', 'epse_3.0', 'epse_4.0', 'epse_5.0', 'epse_7.0', 'epse_9.0', 'epse_15.0']

# ───── Dataset clas ──────────────────────────────────────
class PolyDataset(Dataset):
    def __init__(self, df, tokenizer, max_token_len=128):
        self.df = df
        self.tokenizer = tokenizer
        self.max_token_len = max_token_len

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        content = list(self.df.loc[idx])
        # psmiles = content[2]
        psmiles = content[0]
        targets = np.array(content[1:])
        targets = torch.from_numpy(targets)
        return psmiles, targets


# ───── Mean Pooling func ──────────────────────────────────────
def mean_pooling(model_output, attention_mask):
    if isinstance(model_output, tuple):
        token_embeddings = model_output[0]
    else:
        token_embeddings = model_output.last_hidden_state
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


# ───── subprocess ───────────────────────
def token_extraction(file_path):
    model_name = 'kuelumbus/polyBERT'
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    fname = os.path.basename(file_path)
    dest_path = os.path.join(dest, fname)

    df = pd.read_parquet(file_path).reset_index(drop=True)
    dataset = PolyDataset(df, tokenizer)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, drop_last=False, num_workers=0)

    token_indices, token_type_indices, masks = [], [], []
    prop = []
    pad_idx = tokenizer.pad_token_id
    eos_idx = tokenizer.eos_token_id
    cnt = 0

    for (psmiles, _) in loader:
        with torch.no_grad():
            encoded_batch = tokenizer(psmiles, padding="max_length", max_length=160, truncation=True, return_tensors="pt")
            
            for input_ids, type_ids, mask in zip(
                encoded_batch['input_ids'],
                encoded_batch['token_type_ids'],
                encoded_batch['attention_mask']
            ):
                token_list = input_ids.tolist()
                mask_list = mask.tolist()
                type_list = type_ids.tolist()

                if len(input_ids) == 0:
                    continue

                token_indices.append(torch.tensor(token_list, dtype=torch.long))
                token_type_indices.append(torch.tensor(type_list, dtype=torch.long))
                masks.append(torch.tensor(mask_list, dtype=torch.long))

                prop.append(np.array([df[x][cnt] for x in prop_list]))
                cnt += 1


    new_len = len(token_indices)
    df = {'smiles': df['smiles'],
          'token_ids': [x.cpu().numpy().astype(np.int16) for x in token_indices],
          'mask': [np.argmin(x.cpu().numpy()).astype(np.int16) if 0 in x else 160 for x in masks],
          'properties': prop
         }
    df = pd.DataFrame.from_dict(data=df, orient="columns")
    df.to_parquet(dest_path, index=False)

    return f"Processed: {fname} ({new_len} valid sequences)"


# ───── Main 함수 ──────────────────────────────────────
def main():
    files = glob(os.path.join(source, '*.parquet'))

    with Pool(processes=args.workers) as pool:
        results = list(tqdm(pool.imap(token_extraction, files), total=len(files)))
        for r in results:
            print(r)


# ───── Entry Point ──────────────────────────────────────
if __name__ == '__main__':
    torch.multiprocessing.set_start_method('spawn', force=True)
    main()