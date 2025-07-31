# Rethink-Merge

The code repository of paper "Rethinking Weight-Averaged Model-merging".

## Installation

```commandline
pip install -r requirements.txt
```

For more requirements, please refer to requirements.txt

## Data Preparation

In the code implementation, the download of cifar10 / cifar100 / tinyimagenet datasets will be handled by the code automatically.

## Wt_Patterns is the folder of Section 3 (Patterns in Weights)

For model training, the commandline is:

```commandline
bash run.sh [GPU id]
```

## Wt_vs_Ft is the folder of Section 4 (Merging VS. Ensemble) and Section 5 (Sensitivity to Weight Magnitudes)

For model training, the commandline is:

```commandline
bash run.sh [GPU id]
```

For model merging:

```commandline
bash run_merge.sh 0
```

If you got a chance to use our code, you could consider to cite our paper.

Enjoy!!
