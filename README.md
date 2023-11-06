# MapLight TDC

[![arXiv](https://img.shields.io/badge/arXiv-2310.00174-b31b1b.svg)](https://arxiv.org/abs/2310.00174)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/maplightrx/MapLight-TDC/blob/main/submission.ipynb)
[![Powered by RDKit](https://img.shields.io/badge/Powered%20by-RDKit-3838ff.svg?logo=data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABAAAAAQBAMAAADt3eJSAAAABGdBTUEAALGPC/xhBQAAACBjSFJNAAB6JgAAgIQAAPoAAACA6AAAdTAAAOpgAAA6mAAAF3CculE8AAAAFVBMVEXc3NwUFP8UPP9kZP+MjP+0tP////9ZXZotAAAAAXRSTlMAQObYZgAAAAFiS0dEBmFmuH0AAAAHdElNRQfmAwsPGi+MyC9RAAAAQElEQVQI12NgQABGQUEBMENISUkRLKBsbGwEEhIyBgJFsICLC0iIUdnExcUZwnANQWfApKCK4doRBsKtQFgKAQC5Ww1JEHSEkAAAACV0RVh0ZGF0ZTpjcmVhdGUAMjAyMi0wMy0xMVQxNToyNjo0NyswMDowMDzr2J4AAAAldEVYdGRhdGU6bW9kaWZ5ADIwMjItMDMtMTFUMTU6MjY6NDcrMDA6MDBNtmAiAAAAAElFTkSuQmCC)](https://www.rdkit.org/)

This repository contains the source code for MapLight's [Therapeutics Data Commons (TDC) ADMET Benchmark Group](https://tdcommons.ai/benchmark/admet_group/overview/) submission.

## Installation

This codebase describes MapLight's two submissions to the TDC leaderboards:

1. MapLight model ([`submission.ipynb`](https://github.com/maplightrx/MapLight-TDC/blob/main/submission.ipynb)): [CatBoost](https://catboost.ai/) gradient boosted decision trees with [ECFP](https://pubmed.ncbi.nlm.nih.gov/20426451/), [Avalon](https://pubmed.ncbi.nlm.nih.gov/16995723/), and [ErG fingerprints](https://pubmed.ncbi.nlm.nih.gov/16426057/), as well as [200 physicochemical descriptors](https://www.blopig.com/blog/2022/06/how-to-turn-a-molecule-into-a-vector-of-physicochemical-descriptors-using-rdkit/). Runnable on Colab.

2. MapLight + GNN model ([`submission_gnn.ipynb`](https://github.com/maplightrx/MapLight-TDC/blob/main/submission_gnn.ipynb)): the same as the MapLight model with [graph isomorphism network (GIN) supervised masking fingerprints](https://arxiv.org/abs/1905.12265) from [`molfeat`](https://molfeat.datamol.io/featurizers/gin_supervised_masking). __WARNING__: Not runnable on Colab becuase of [this issue](https://github.com/datamol-io/molfeat/issues/61).

Both notebooks will install all dependencies in a new Python environment with [JupyterLab](https://jupyterlab.readthedocs.io/en/stable/getting_started/overview.html):

```
# Create an environment for this project
mamba create -n maplight python=3.10 -y && mamba activate maplight

mamba install jupyterlab
```

