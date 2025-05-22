# Optimizing Learning-to-Rank Models for Ex-Post Fair Relevance

[Paper URL](https://dl.acm.org/doi/pdf/10.1145/3626772.3657751)

Previous works used in this paper:

**Learning-to-Rank at the Speed of Sampling: Plackett-Luce Gradient Estimation With Minimal Computational Complexity**: [This repository](https://github.com/HarrieO/2022-SIGIR-plackett-luce) contains the code for PL-Rank-3; ([pdf available here](https://harrieo.github.io//publication/2022-sigir-short)).

**Sampling Ex-Post Group-Fair Rankings**: [This repository](https://github.com/sruthigorantla/sampling_random_group_fair_rankings) contains the code used for the distribution over the fair group assignments; ([pdf available here](https://arxiv.org/pdf/2203.00887.pdf)).

Both of the above algorithms are also included in our implementation. The users need not download anything from the repositories above to replicate our results.

## Installation

```bash
# Clone the repository
git clone Group-Fair-PL.git
cd Group-Fair-PL

# Make an editable installation
pip install -e .
```

## Our algorithm

A file is required that explains the location and details of the LTR datasets available on the system, example file is available in `configs/local_dataset_info.txt`
Open this and edit the paths to the folders where the train/test/vali files are placed.

Many experiments can be run one after the other by removing the comments in the file `run.sh`

```bash
python -m groupfair_pl.run_main --help
usage: run_main.py [-h] --file FILE [--loss {plrank3,groupfairpl,policygradient,placementpolicygradient}] [-p POSTPROCESS_ALGORITHMS] [-b BIAS] [-r RUN_NO]
                   [--device DEVICE]

options:
  -h, --help            show this help message and exit
  --file FILE           path to a config file for the algorithms
  --loss {plrank3,groupfairpl,policygradient,placementpolicygradient}
                        name of the loss function
  -p POSTPROCESS_ALGORITHMS, --postprocess_algorithms POSTPROCESS_ALGORITHMS
                        a comma-delimited list of postprocessing algorithms to apply. [options: none, GDL23, GAK19]
  -b BIAS, --bias BIAS  bias component of (0.bias) the dataset; use --bias -1 for no bias
  -r RUN_NO, --run_no RUN_NO
                        run number
  --device DEVICE       torch device to use for training [defaults to `cuda` if available]
```

The following command optimizes our relevance metric with Group-Fair-PL on the German Credit dataset:

```bash
python -m groupfair_pl.run_main --file configs/config_German.jsonc --loss Group-Fair-PL --postprocess_algorithms none --run_no $i --bias $b
```

The following command optimizes NDCG with PL-rank-3 on the German Credit dataset with (1) no postproccessing, (2) GDL23 postprocessing and (3) GAK19 postprocessing algroithms:

```bash
python -m groupfair_pl.run_main --file configs/config_German.jsonc --loss PL-rank-3 --postprocess_algorithms none,GDL23,GAK19 --run_no $i --bias $b
```

---

To add bias to the dataset, go to `scripts` and run `add_bias.py` with a command line argument `--bias` indicating the bias (the number after the decimal point in the bias factor). For example, the following command adds bias of 0.5 on the German credit dataset and creates a new data folder called Germanbiased_0_5

```bash
python add_bias.py --dataset German --bias 5 --ispca false
```
The `--ispca` flag indicates whether we want to use the pca preprocessed dataset. PCA preprocessing can be applied using `do_pca.py`.

## Datasets

Example datasets have been included in this repository in `data` folder.
