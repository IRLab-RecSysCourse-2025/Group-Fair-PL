# 🔁 Reproducibility Instructions

**Optimizing Learning-to-Rank Models for Ex-Post Fair Relevance** [(Paper URL)](https://dl.acm.org/doi/pdf/10.1145/3626772.3657751)

This document provides the full set of instructions to reproduce our project results from scratch, including data setup, environment configuration, training, and evaluation.





## 🧱 Project Structure

```bash
.
├── config                  # Contains configuration files
├── data                    # Contains raw and processed datasets
├── scripts                 # Contains slurm job scripts and other utilities
├── src                     # All source code (models, training, evaluation)
    ├── algorithms          # Bare LTR algorithms
    ├── utils               # Helpers and utilities
    └── main.py             # Main entry point for training and evaluation
├── pyproject.toml          # Python dependencies
├── README.md               # README file
└── REPRO.md                # Reproducibility instructions
```

---

## ⚙️ Environment Setup


Setup project as you would any other Python project:

```bash
# Clone the repository
git clone https://github.com/IRLab-RecSysCourse-2025/Group-Fair-PL.git
cd Group-Fair-PL

# Make an editable installation (this will also install all dependencies)
pip install -e .
```

---

## 📂 Download & Prepare Datasets

Place your datasets in the `data/` directory.

!UNDER CONSTRUCTION!

### Example Dataset
```bash
mkdir -p data/example_dataset
cd data/example_dataset
wget xxxxx
python -m src.preprocess_example_dataset.py xxxx
cd ../..
```

> [!IMPORTANT]
> A file is required that explains the location and details of the LTR datasets available on the system, example file is available in `config/local_dataset_info.json`.

### Adding Bias

To add bias to the dataset, go to `scripts` and run `add_bias.py` with a command line argument `--bias` indicating the bias (the number after the decimal point in the bias factor). For example, the following command adds bias of 0.5 on the German credit dataset and creates a new data folder called Germanbiased_0_5

```bash
python add_bias.py --dataset German --bias 5 --ispca false
```

The `--ispca` flag indicates whether we want to use the PCA preprocessed dataset. PCA preprocessing can be applied using `do_pca.py`.

---

## ⚙️ Configuration

Parameters are stored in the `config` directory. The configuration files are in the extended JSONC format.

## 🚀 Usage

### Baselines

Run the following command to train the baseline:

```bash
python -m src.main --help
usage: main.py [-h] --file FILE [--loss {plrank3,groupfairpl,policygradient,placementpolicygradient}] [-p POSTPROCESS_ALGORITHMS] [-b BIAS] [-r RUN_NO]
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

Alternatively, if you are using SLURM, you may make use of the provided SLURM template job scripts in `scripts`.

<!-- --- -->

<!-- ## 📈 Evaluation

After training, evaluate all models with:

```bash
python XXXX
``` -->

<!-- ---


## 📎 Misc. Notes (optional)

--- -->

## 📦 Dependencies / References

This project repository uses code from the following frameworks / refers to the following papers:

- **Learning-to-Rank at the Speed of Sampling: Plackett-Luce Gradient Estimation With Minimal Computational Complexity**: [This repository](https://github.com/HarrieO/2022-SIGIR-plackett-luce) contains the code for PL-Rank-3; ([pdf available here](https://harrieo.github.io//publication/2022-sigir-short)).
- **Sampling Ex-Post Group-Fair Rankings**: [This repository](https://github.com/sruthigorantla/sampling_random_group_fair_rankings) contains the code used for the distribution over the fair group assignments; ([pdf available here](https://arxiv.org/pdf/2203.00887.pdf)).

Both of the above are also included in this code repository. You do not need to download anything from the repositories above, but you can browse them for reference.
