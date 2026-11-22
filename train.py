"""Train and evaluate the paper's R-NoContext model on the frozen ten folds."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import random

import numpy as np
import torch
from sklearn.metrics import (accuracy_score, average_precision_score, f1_score,
                            precision_score, recall_score, roc_auc_score)

from nc_model import RNoContext
from nc_model.data import load_fold, load_graph, verify_files


ROOT = Path(__file__).resolve().parent
CONFIG = dict(seed=0, num_layers=2, share_layers=True, hidden_dim=64, head_hidden=64,
              dropout=0.2, learning_rate=0.002, weight_decay=0.0005, epochs=100)
METRICS = ("ACC", "Precision", "Recall", "F1", "AUROC", "AUPR")


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save_checkpoint(path, checkpoint):
    temp = path.with_suffix(".tmp")
    torch.save(checkpoint, temp)
    os.replace(temp, path)


def write_csv(path, records):
    temp = path.with_suffix(".tmp")
    with temp.open("w", encoding="utf8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=records[0], lineterminator="\n")
        writer.writeheader()
        writer.writerows(records)
    os.replace(temp, path)


def load_checkpoint(path, identity):
    checkpoint = torch.load(path, weights_only=True, map_location="cpu")
    if checkpoint["identity"] != identity:
        raise ValueError(f"Saved model configuration or inputs differ: {path}")
    return checkpoint


def probabilities(model, graph, positive, negative):
    # Keep the original order of the two encoder calls and their dropout draws.
    pos = model(graph, positive).reshape(-1).sigmoid()
    neg = model(graph, negative).reshape(-1).sigmoid()
    return torch.cat((pos, neg)), torch.cat((torch.ones_like(pos), torch.zeros_like(neg)))


def fit(graph, split, folder, identity, device):
    cfg = identity["config"]
    random.seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    torch.manual_seed(cfg["seed"])
    model = RNoContext(cfg, identity["input_dim"]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["learning_rate"],
                                 weight_decay=cfg["weight_decay"])
    resume = folder / "resume.pt"
    start = 0
    if resume.exists():
        saved = load_checkpoint(resume, identity)
        if saved["device_type"] != device.type:
            raise ValueError("Resume with the device type used for this fold")
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        torch.set_rng_state(saved["cpu_rng"])
        if device.type == "cuda":
            torch.cuda.set_rng_state(saved["cuda_rng"], device)
        start = saved["epoch"]
        if not 0 < start <= cfg["epochs"]:
            raise ValueError(f"Invalid saved epoch {start}")

    model.train()
    for epoch in range(start+1, cfg["epochs"]+1):
        optimizer.zero_grad(set_to_none=True)
        scores, labels = probabilities(model, graph, split["train_pos"], split["train_neg"])
        loss = torch.nn.functional.binary_cross_entropy(scores, labels)
        if not torch.isfinite(loss).item():
            raise ValueError(f"Nonfinite training loss, epoch {epoch}")
        loss.backward()
        optimizer.step()
        save_checkpoint(resume, dict(identity=identity, model=model.state_dict(),
            optimizer=optimizer.state_dict(), epoch=epoch, device_type=device.type,
            cpu_rng=torch.get_rng_state(),
            cuda_rng=torch.cuda.get_rng_state(device) if device.type=="cuda" else None))
    save_checkpoint(folder/"final.pt", dict(identity=identity, epoch=cfg["epochs"], model=model.state_dict()))


def measure(labels, scores):
    pred = scores > 0.5
    return dict(ACC=accuracy_score(labels, pred), Precision=precision_score(labels, pred, zero_division=0),
                Recall=recall_score(labels, pred, zero_division=0), F1=f1_score(labels, pred, zero_division=0),
                AUROC=roc_auc_score(labels, scores), AUPR=average_precision_score(labels, scores))


def run_fold(graph, split, folder, identity, device):
    folder.mkdir(parents=True, exist_ok=True)
    final = folder/"final.pt"
    csv_path = folder/"test_predictions.csv"
    receipt = folder/"metrics.json"
    if not final.exists():
        if receipt.exists() or csv_path.exists():
            raise FileNotFoundError(f"Predictions exist but final weights are missing: {final}")
        fit(graph, split, folder, identity, device)
    ck = load_checkpoint(final, identity)
    if ck["epoch"] != identity["config"]["epochs"]:
        raise ValueError(f"Unexpected final epoch: {final}")
    model = RNoContext(identity["config"], identity["input_dim"]).to(device)
    model.load_state_dict(ck["model"], strict=True)
    pairs = torch.cat((split["test_pos"], split["test_neg"])).cpu().numpy()
    labels = np.repeat([1, 0], [len(split["test_pos"]),len(split["test_neg"])])
    if receipt.exists():
        saved = json.loads(receipt.read_text(encoding="utf8"))
        if sha256(final) != saved["checkpoint_sha256"] or sha256(csv_path) != saved["predictions_sha256"]:
            raise ValueError(f"Previously completed fold has changed: {folder}")
    else:
        model.eval()
        with torch.inference_mode():
            scores = probabilities(model, graph, split["test_pos"], split["test_neg"])[0].cpu().tolist()
        write_csv(csv_path, [dict(row=i, herb=int(h), symptom=int(s), label=int(y), probability=p)
                             for i, ((h,s), y, p) in enumerate(zip(pairs, labels, scores))])
    with csv_path.open(encoding="utf8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    expected = [(i,int(h),int(s),int(y)) for i,((h,s),y) in enumerate(zip(pairs,labels))]
    if [tuple(int(r[k]) for k in ("row","herb","symptom","label")) for r in rows] != expected:
        raise ValueError(f"Frozen test rows differ: {folder}")
    scores = np.array([float(r["probability"]) for r in rows])
    if not np.all(np.isfinite(scores) & (scores>=0) & (scores<=1)):
        raise ValueError(f"Invalid probability: {csv_path}")
    record = dict(dataset=identity["dataset"], fold=identity["fold"], **measure(labels,scores))
    result = dict(**record,checkpoint_sha256=sha256(final),predictions_sha256=sha256(csv_path))
    if receipt.exists() and saved != result:
        raise ValueError(f"Saved metrics disagree with predictions: {receipt}")
    temp = receipt.with_suffix(".tmp")
    temp.write_text(json.dumps(result,indent=2)+"\n",encoding="utf8")
    os.replace(temp,receipt)
    (folder/"resume.pt").unlink(missing_ok=True)
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--data-dir", type=Path, default=ROOT/"data")
    parser.add_argument("--output-dir", type=Path, default=ROOT/"outputs")
    args = parser.parse_args()
    device = torch.device(args.device)
    torch.set_num_threads(1)
    summaries = []
    for dataset in ("HIT","TCMIO"):
        data_hash = verify_files(args.data_dir,dataset)
        graph = load_graph(args.data_dir, dataset, device)
        records = []
        for fold in range(10):
            split = load_fold(args.data_dir,dataset,fold,device)
            folder = args.output_dir/dataset/f"fold_{fold:02d}"
            identity = dict(model="R-NoContext", dataset=dataset, fold=fold, config=CONFIG,
                            input_dim=graph.x.shape[1], dataset_sha256=data_hash)
            record = run_fold(graph, split, folder, identity, device)
            records.append(record)
            print(f"[{dataset} fold {fold+1}/10] " + "  ".join(f"{k} {record[k]:.4f}" for k in METRICS), flush=True)
        write_csv(args.output_dir/dataset/"fold_metrics.csv", records)
        summary = dict(dataset=dataset, folds=len(records))
        for key in METRICS:
            values = [r[key] for r in records]
            summary[key+"_mean"] = float(np.mean(values))
            summary[key+"_std"] = float(np.std(values,ddof=0))
        summaries.append(summary)
        del graph
    write_csv(args.output_dir/"summary.csv",summaries)


if __name__ == "__main__":
    main()
