# Experiment settings

## Models and training

`configs/baseline.json` and `configs/iterative.json` define the two models. Both use BERT-base-cased, learned node type/position/sentence features, dropout 0.25, learning rate 5e-5, and 35 epochs. Runs use seeds 30, 31 and 32.

The edge loss is focal loss with gamma 3 and class weights computed from the training split. The graph loss is binary cross entropy on logits. The total loss weights graph and edge loss by 0.8 and 0.2. The iterative model uses two GCN layers, hidden dimension 512 and adjacency retention weight 0.3. Checkpoints are selected by DEV performance.

Training adds a single Gumbel perturbation to arc scores. Evaluation keeps temperature scaling and removes the noise. Intermediate states score arcs from pre-update features and labels from post-update features; the final state uses final features for both.

`<X>` marks inactive configuration fields. 


## Metrics and output files

Document-macro F1 averages F1 over documents. Label-macro F1 averages over relation labels; micro F1 pools edge counts. The temporal-only metrics exclude `Depend-on`. Reported spreads are sample standard deviations across seeds (`ddof=1`); gains are paired within each seed before averaging.

| File in `results/` | Contents |
|---|---|
| `paper_per_seed.csv` | TEST metrics and selected epochs |
| `paper_dev_iterations.csv` | DEV metrics at each refinement state |
| `paper_dev_stability.csv` | Jaccard similarity between consecutive predictions |
| `paper_dev_distance.csv` | DEV micro F1 by sentence distance |
| `paper_dev_relations.csv` | DEV micro F1 by relation |

The paper's self-Jaccard values use the micro variant. These compare predictions across iterations; the general `jaccard_micro` metric compares predictions with gold edges. File hashes and experiment identifiers are recorded in `results/provenance.json`.

Each new training run writes checkpoints, logs and metrics to its own directory. The top-level `manifest.json` identifies the runs for `summarize.py` and `report.py`. A full report requires both models, all three seeds and the iterative DEV diagnostics.

`report.py` exports LaTeX tables, diagnostic CSVs, JSON and an optional DEV figure. The literature rows in `comparison.tex` come from the paper; their evaluation protocols differ from ours.

## Dependencies

`requirements-linux-cu121-py310.lock` pins the Python 3.10/Linux/CUDA 12.1 environment, including PyTorch 2.3.0, DGL 2.3.0 and Transformers 4.55.4. `requirements-linux-cu121-py310.txt` lists the direct dependencies. Figure generation additionally uses `requirements-report.txt`.
