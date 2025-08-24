
The implementation code for the research paper "High Computational Density Nanophotonic Media for Machine Learning Inference".
## System Requirements

### Core Dependencies
- **Python**: Version 3.9
- **Python Packages**:
  - `splayout` == 0.5.15
  - `numpy` == 1.24.3
  - `scipy` == 1.10.1
  - `jaxlib` == 0.4.30
  - `jax` == 0.4.30

### Required Software
- **Ansys Lumerical**: Version 2024 R1 or newer

## Execution Instructions

First, install the package:
```bash
python setup.py install
```
To run the iris classification task (Default on GPU): 
```bash
# go to the iris_classification directory
python iris.py
```
To run the digits classification task (On CPU, Require 64GB RAM to run): 
```bash
# go to the digits_classification directory
python ocr.py
```
## Datasets

- **Iris Dataset**  
  The dataset located at `datasets/iris` is a standardized version of the classic Iris dataset:  
  Fisher, R. A. *Iris*. UCI Machine Learning Repository. https://doi.org/10.24432/C56C76 (1988).  

- **Optical Recognition of Handwritten Digits Dataset**  
  The dataset located at `datasets/ocr` is a standardized version of the Optical Recognition of Handwritten Digits dataset:  
  Alpaydin, E. & Kaynak, C. *Optical Recognition of Handwritten Digits*. UCI Machine Learning Repository. https://doi.org/10.24432/C50P49 (1998).  

## Citation (Preprint)
```text
@misc{zhao2025high,
  title        = {High computational density nanophotonic media for machine learning inference},
  author       = {Zhenyu Zhao and Yichen Pan and Jinlong Xiang and Yujia Zhang and An He and Yaotian Zhao and Youlve Chen and Yu He and Xinyuan Fang and Yikai Su and Min Gu and Xuhan Guo},
  year         = {2025},
  eprint       = {arXiv:2506.14269},
  archivePrefix = {arXiv},
  primaryClass = {physics.optics},
}
```

## Project Structure
```shell
code/
│
├── datasets/                       # directory for datasets
│
├── nao/                            # design method and constraints implementation
│   ├── rodconstrain.py             # constraints implementation
│   ├── adjointrodcomplexdirect.py  # nao method implementation
│   ├── backend.py         
│   └── rodregion.py                # nao method implementation
│
├── iris_classification/            # directory for iris classification task
│   ├── batch_preprocess.py             
│   └── iris.py                     # script to run the design for the iris classification task
│
├── digits_classification/          # directory for digits classification task
│   ├── batch_preprocess.py             
│   └── ocr.py                      # script to run the design for the digits classification task
│
├── setup.py                        # setup for the package of the design method
├── LICENSE                         # license
└── README.md                       # project overview
```
