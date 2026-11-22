# R-NoContext

PyTorch implementation of **OurModel (R-NoContext)** for herb–symptom association prediction. The model propagates protein representations through two membership hypergraphs and the protein–protein interaction (PPI) graph. Directed cross-relation messages use content scores, relation biases and receiver gates. The classifier reads both the first-round local entity representations and the final entity representations.

## Training

Use Python 3.10+ with a CUDA-enabled PyTorch 2.8 installation. Dependencies are listed in `requirements.txt`. From the repository root on Linux, select the physical GPU and run both datasets with:

```bash
GPU_ID=1 bash run_all.sh
```

The launcher maps the selected GPU to process-local `cuda:0` and uses Linux `flock` to protect the default output directory. Run the same command to resume. Each completed fold has an independent `outputs/<dataset>/fold_XX/final.pt`, test predictions and metrics. Training saves an atomic rolling checkpoint after each epoch and removes it once the final model and predictions are verified.

## Data and protocol

- `data/shared/` contains the shared 18,505 × 512 protein features, ordered protein IDs and real PPI edge list. `data/HIT/` and `data/TCMIO/` contain ordered entity IDs, herb and symptom protein memberships, and the ten frozen outer folds.
- The inputs follow the MHGNN benchmark. Features originate from its protein embeddings, and each PPI endpoint is a zero-based row in the shared feature array. Prediction CSV herb and symptom indices address the ordered lists in the corresponding `ids.json`. In each fold archive, `train_indices` and `test_indices` identify original positive rows. The four positive/negative pair arrays preserve their original order and repeated pairs. Input files are verified against `data/sha256.json`. Dataset reference: [Liang et al., MHGNN](https://doi.org/10.1109/TNNLS.2026.3677056).
- The paper configuration uses seed 0 independently for each fold, hidden width 64, two rounds sharing weights, dropout 0.2, Adam with learning rate 0.002 and weight decay 0.0005. Positive and negative examples use separate full-graph forwards. Their probabilities enter one mean BCE and one optimizer update per epoch. Each fold trains for 100 epochs and evaluates the final checkpoint.
- Each prediction is positive when its probability is strictly greater than 0.5. AUPR uses average precision. `outputs/<dataset>/fold_metrics.csv` lists ACC, Precision, Recall, F1, AUROC and AUPR. `outputs/summary.csv` reports their equally weighted ten-fold means and population standard deviations (`ddof=0`).
- `nc_model/model.py` preserves the paper model's parameter names and initialization order, including the registered context projection tensors retained for strict checkpoint compatibility. Cross-relation scoring uses the content term and relation bias.
