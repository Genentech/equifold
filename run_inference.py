import gzip
import argparse
from models import NN
from tqdm import tqdm
import os
import numpy as np
import torch
from torch_geometric.data import Data
from torch.utils.data import DataLoader
from utils_data import (MAX_DIST, cg_X0, collate_fn, x_to_pdb, sequence_to_feats)
import json
import pandas as pd
from multiprocessing import Pool
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def process_one(job):
    uid, seq1, seq2 = job
    cg_cgidx, cg_resnum, scatter_index, scatter_w, dst_resnum, dst_atom, dst_resname, offset = sequence_to_feats(seq1, dst_idx_offset=0)
    if seq2 is not None:
        cg_cgidx2, cg_resnum2, scatter_index2, scatter_w2, dst_resnum2, dst_atom2, dst_resname2, _ = sequence_to_feats(seq2, dst_idx_offset=offset)
        seq2_offset = len(seq1) + MAX_DIST
        cg_cgidx = np.concatenate([cg_cgidx, cg_cgidx2])
        cg_resnum = np.concatenate([cg_resnum, cg_resnum2 + seq2_offset])
        scatter_index = np.concatenate([scatter_index, scatter_index2])
        scatter_w = np.concatenate([scatter_w, scatter_w2])
        dst_resnum = np.concatenate([dst_resnum, dst_resnum2 + seq2_offset])
        dst_atom = np.concatenate([dst_atom, dst_atom2])
        dst_resname = np.concatenate([dst_resname, dst_resname2])

    dtype=torch.float32
    cg_cgidx = torch.from_numpy(cg_cgidx)
    data = Data(num_nodes=len(cg_cgidx),
                cg_resnum=torch.from_numpy(cg_resnum),
                cg_cgidx=cg_cgidx,
                cg_X0=cg_X0[cg_cgidx].type(dtype),
                scatter_index=torch.from_numpy(scatter_index),
                scatter_w=torch.from_numpy(scatter_w).type(dtype),
                dst_resnum=torch.from_numpy(dst_resnum),
                dst_atom=dst_atom,
                dst_resname=dst_resname,
                uid=uid
                )

    return data


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="ab", type=str, choices=["ab", "science"])
    parser.add_argument("--model_dir", default="models", type=str)
    parser.add_argument("--seqs", default=None, type=str)
    parser.add_argument("--ncpu", default=1, type=int)
    parser.add_argument("--out_dir", default="out", type=str)    
    args = parser.parse_args()

    # load model
    model_path = f"{args.model_dir}/{args.model}_weights.pt"
    config_path = f"{args.model_dir}/{args.model}_config.json"
    with open(config_path, "r") as f:
        config = json.load(f)
    model = NN(**config)
    model.load_state_dict(torch.load(model_path))
    model = model.to(device)
    model.eval()

    # load data
    df = pd.read_csv(args.seqs)
    uids = df["uid"].tolist()
    assert len(uids) == len(set(uids))
    if args.model == "ab":
        seqs1 = df["heavy"].tolist()
        seqs2 = df["light"].tolist()
    else:
        seqs1 = df["seq"].tolist()
        seqs2 = [None] * len(seqs1)

    # prepare data structures using multiproc
    jobs = list(zip(uids, seqs1, seqs2))
    with Pool(args.ncpu) as p:
        dataset = list(tqdm(p.imap_unordered(process_one, jobs), total=len(jobs)))
        p.close()
        p.join()

    # run inference and save
    loader = DataLoader(dataset, batch_size=1, drop_last=False, shuffle=False, num_workers=0, collate_fn=collate_fn, pin_memory=True)

    os.makedirs(args.out_dir, exist_ok=True)
    with torch.no_grad():
        for data in tqdm(loader):
            data = data.to(device)
            results_dict = model(data, compute_loss=False, return_struct=True, set_RT_to_ground_truth=False)

            # get pred
            x_pred = results_dict["x_pred"][0][-1]

            # write pred
            with gzip.open(f"{args.out_dir}/{data[0].uid}.pred.pdb.gz", "wb") as f:
                f.write(x_to_pdb(x_pred.cpu(), 
                         data[0]["dst_resnum"], 
                         data[0]["dst_resname"], 
                         data[0]["dst_atom"]).encode())
