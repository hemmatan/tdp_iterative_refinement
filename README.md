# Iterative refinement for temporal dependency parsing

Code for **Raffinement itératif pour l’analyse des dépendances temporelles à l’échelle du document**, Ensieh Hemmatan, Timothée Bernard and Benoît Crabbé, TALN 2026.

The models parse temporal dependencies between gold event and timex spans.

## Installation

Install the dependencies in a virtual environment:

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-linux-cu121-py310.lock
```


## Data

Use the TDG corpus from *Annotating Temporal Dependency Graphs via Crowdsourcing* (Yao et al., EMNLP 2020). Place `train.txt`, `dev.txt` and `test.txt` in a data directory, then check the splits:

```bash
python scripts/check_data.py /path/to/tdg_data
```

The splits contain 400, 50 and 50 documents. Corpus files and trained checkpoints are not included.

## Training

```bash
python train.py --data-dir /path/to/tdg_data --output-dir outputs/paper-runs --gpu 0
python summarize.py --runs outputs/paper-runs/manifest.json
```

Each run trains for 35 epochs. Use a new output directory. Add `--dry-run` to preview the commands, or `--mode baseline --seeds 30` to run one configuration. Settings and metric definitions are in [docs/EXPERIMENTS.md](docs/EXPERIMENTS.md).

## Paper results

The files in `results/` contain results from the paper experiments. The table shows document-macro F1 (%) as mean ± sample standard deviation over the three seeds. The temporal subset excludes `Depend-on`.

| Model | All labeled | All unlabeled | Temporal labeled | Temporal unlabeled |
|---|---:|---:|---:|---:|
| Single-pass, T=0 | 66.56 ± 0.19 | 73.97 ± 0.31 | 60.99 ± 0.16 | 71.53 ± 0.44 |
| Refinement, T=3 | 68.95 ± 0.80 | 76.06 ± 0.81 | 63.98 ± 0.71 | 73.96 ± 0.77 |

Summarize these results and export the tables:

```bash
python summarize.py
python report.py --output-dir outputs/paper-report --no-plot
```


## Citation and license

Please cite the paper when using this code; see [CITATION.cff](CITATION.cff).

Project-authored code is available under the [MIT License](LICENSE). Attribution and license scope are listed in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
