# EquiFold

This is the official open source repository for [EquiFold](https://www.biorxiv.org/content/10.1101/2022.10.07.511322v1) developed by [Prescient Design, a Genentech accelerator.](https://gene.com/prescient)


## Notes
- This light-weight research version of the code was used to produce figures reported in the manuscript (to be updated soon). We plan to release a higher-quality version of the code with additional, user-level features in the future.
- There are known issues occasionally seen in predicted structures including nonphysical bond geometry and clashes. We are currently developing approaches to minimize these issues for future releases.


## Setup and Usage
### Environment
We used the following GPU-enabled setup with `conda` (originally run in an HPC environment with NVIDIA A100 GPUs).
```
$ conda create -n ef python=3.9 -y
$ conda activate ef
$ conda install pytorch=1.11 cudatoolkit=11.3 -c pytorch -y
$ pip install torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric -f https://data.pyg.org/whl/torch-1.11.0+cu113.html 
$ pip install e3nn pytorch-lightning biopython pandas tqdm einops
```

Alternatively, for use without GPUs:
```
conda create -n ef python=3.9 -y
conda activate ef
conda install pytorch=1.12 -c pytorch -y
conda install pyg -c pyg
pip install e3nn pytorch-lightning biopython pandas tqdm einops
```


### Model weights
PyTorch model weights and hyper-parameter configs for the models trained on mini-protein and antibody datasets as described in the manuscript are stored in `models` directory.


### Run model predictions
To make predictions using a trained model, users can run the following scripts providing input sequences as a CSV table:

```
# For antibodies
$ python run_inference.py --model ab --model_dir models --seqs tests/data/inference_ab_input.csv --ncpu 1 --out_dir out_tests

# For mini-proteins
$ python run_inference.py --model science --model_dir models --seqs tests/data/inference_science_input.csv --ncpu 1 --out_dir out_tests
```
## Contributing

We welcome contributions. If you would like to submit pull requests, please make sure you base your pull requests off the latest version of the `main` branch. Keep your fork synced by setting its upstream remote to `Genentech/equifold` and running:

```sh
# If your branch only has commits from master but is outdated:

$ git pull --ff-only upstream main


# If your branch is outdated and has diverged from main branch:

$ git pull --rebase upstream main
```

## License
Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at https://www.apache.org/licenses/LICENSE-2.0.

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.


## Citations
If you use the code and/or model, please cite:
```
@article {Lee2022.10.07.511322,
    author = {Lee, Jae Hyeon and Yadollahpour, Payman and Watkins, Andrew and Frey, Nathan C. and Leaver-Fay, Andrew and Ra, Stephen and Cho, Kyunghyun and Gligorijevi{\'c}, Vladimir and Regev, Aviv and Bonneau, Richard},
    title = {EquiFold: Protein Structure Prediction with a Novel Coarse-Grained Structure Representation},
    elocation-id = {2022.10.07.511322},
    year = {2023},
    doi = {10.1101/2022.10.07.511322},
    publisher = {Cold Spring Harbor Laboratory},
    URL = {https://www.biorxiv.org/content/early/2023/01/02/2022.10.07.511322},
    eprint = {https://www.biorxiv.org/content/early/2023/01/02/2022.10.07.511322.full.pdf},
    journal = {bioRxiv}
}
```
