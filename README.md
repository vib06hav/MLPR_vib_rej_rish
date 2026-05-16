# BDD100K Day-to-Night Vehicle Classification and Domain Adaptation

This repo contains the core BDD100K classification and domain-adaptation project for `bus`, `car`, and `truck`.

The central question is:

How much does performance degrade when a model trained on daytime vehicle crops is tested on nighttime crops, and how much of that gap can be recovered with supervised fine-tuning and DANN-based adaptation?

## Main Files

- `config.py`: paths, hyperparameters, seeds, and run switches
- `Preprocessdata.py`: raw BDD100K crop extraction and balancing pipeline
- `dataset.py`: crop dataset loading, transforms, and Direction 1 target-label masking
- `models.py`: classifier backbones and heads
- `dann.py`: gradient reversal and domain classifier
- `train.py`: standard and DANN training loops
- `evaluate.py`: metrics, per-seed aggregation, analysis helpers
- `main.py`: experiment orchestrator
- `run_tsne_bdd.py`: standalone t-SNE visualization script

## Existing Result Summaries

Repo-local summaries:

- `all_results_summary.csv`
- `direction1_results_summary.csv`

These correspond to the BDD classification/domain-adaptation study, not detection.

## Data Layout

The repo expects large data and checkpoints under:

`C:\Users\vibha\Downloads\archive\`

Important external folders include:

- `bdd100k\bdd100k\images\100k\`
- `bdd100k_labels_release\bdd100k\labels\`
- `processed_dataset_51k\`
- `checkpoints_datasetB_final\`
- `checkpoints_51k_label_sweep\`

## Scope Notes

- The BDD classification and DANN pipeline is the main project.
- IDD-related scripts remain in the repo as adjacent work.
- Detection-extension code has been removed from the repo workspace.
