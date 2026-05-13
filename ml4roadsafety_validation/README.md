# ML4RoadSafety HeteroGAT validation

This folder is an isolated harness for validating the local `src.gat_model.HeteroGAT`
implementation on the ML4RoadSafety road-safety benchmark.

Sources:

- ML4RoadSafety repository: https://github.com/VirtuosoResearch/ML4RoadSafety
- Dataset DOI: https://doi.org/10.7910/DVN/V71K5R
- Paper: https://papers.nips.cc/paper_files/paper/2023/hash/a365be0950259c9624edfb4d26eabd46-Abstract-Datasets_and_Benchmarks.html

## What It Builds

The adapter maps ML4RoadSafety road-edge accident labels into the local project
contract:

- Node type `pm`: one road segment in one month.
- Label `y`: binary accident occurrence for that segment-month.
- Relation `("pm", "spatial", "pm")`: line-graph connectivity between road
  segments sharing an endpoint inside the same month.
- Relation `("pm", "temporal", "pm")`: same segment across consecutive months.
- Node features are normalized with train-month statistics only.
- Threshold selection is done on validation only; test is evaluated once with
  that threshold.

The default pilot uses `MA` and months `2022-01 2022-02 2022-03`. To keep the
first run practical, it caps the graph at 5000 road segments while preserving
all positive segments found in the selected months.

## Run

Create the project environment first:

```bash
python venv_start.py --python python3.12
source .venv/bin/activate
```

Download/validate Massachusetts data:

```bash
python ml4roadsafety_validation/download.py --state MA
```

Run the pilot:

```bash
python ml4roadsafety_validation/run_pilot.py \
  --state MA \
  --months 2022-01 2022-02 2022-03 \
  --max-epochs 30
```

If the automatic download fails, manually download `MA.zip` from Dataverse,
extract it under `ml4roadsafety_validation/data/`, and ensure this layout exists:

```text
ml4roadsafety_validation/data/MA/
  adj_matrix.pt
  accidents_monthly.csv
  Edges/edge_features.pt
  Nodes/node_features_2022_1.csv
```

Then run with:

```bash
python ml4roadsafety_validation/run_pilot.py --skip-download
```

## Outputs

Generated files are ignored by git:

- `ml4roadsafety_validation/data/`
- `ml4roadsafety_validation/results/pilot_summary_*.json`
- `ml4roadsafety_validation/results/pilot_metrics_*.csv`

The primary model-selection metric is validation AUCPR. The report also includes
AUROC, precision, recall, F1, F0.5, split prevalence, and confusion counts.

## Tests

The tests use synthetic tensors and do not download ML4RoadSafety:

```bash
pytest ml4roadsafety_validation/tests -q
```

