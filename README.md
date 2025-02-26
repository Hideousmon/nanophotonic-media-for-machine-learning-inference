
The implementation code for the research paper "High Computational Density Nanophotonic Media for Machine Learning Inference".
## System Requirements

### Core Dependencies
- **Python**: Version 3.6 or higher
- **Python Packages**:
  - `splayout` >= 0.5.4
  - `numpy` >= 1.24.3
  - `scipy` >= 1.10.1

### Required Software
- **Ansys Lumerical**: Version 2020 R2 or newer

## Execution Instructions

To run the primary design script, use the following command in your terminal:

```bash
python iris_design.py
```

## Dataset
The dataset utilized in this study is a standardized version of the Iris dataset originally Fisher,R. A. Iris. UCI Machine Learning Repository https://doi.org/10.24432/C56C76 (1988).


## Project Structure
```shell
code/
│
├── dataset/                    # Directory for iris dataset files (standardized with Fisher,R. A. Iris. UCI Machine Learning Repository https://doi.org/10.24432/C56C76 (1988).)
│
├── naa/                      # design method and constraints implementation
│   ├── constraints.py             # constraints implementation
│   └── rodmetamaterialopt.py         # adjoint method implementation
│
├── iris_design.py                 # main design script
├── LICENSE                    # license
└── README.md                   # project overview
```
