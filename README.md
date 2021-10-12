# Dense and well-connected subgraph detection in dual networks

Tianyi Chen, Francesco Bonchi, David Garcia-Soriano, Atsushi Miyauchi, and Charalampos E. Tsourakakis

Code for SDM 2022 submission

## Dataset

Twitter dataset is available from https://github.com/twittermancer/The_Twitter_Mancer_Project. Download and put the folder under data/ to run the corresponding Notebook.

The Brain network dataset we used is from https://github.com/tlancian/contrast-subgraph.

## Package installation

To run the Jupyter Notebooks in this repository, create a python 2.7 virtual environment and run the following installation commands.

```
# cannot install with pip since Python2 is not supported
conda install -c conda-forge cvxopt
conda install -c conda-forge cvxpy

pip install networkx numpy matplotlib gurobipy pandas ipykernel

ipython kernel install --user --name=env_name
```

Start jupyter notebook from your terminal, open a notebook file, and switch the kernel to "env_name".
