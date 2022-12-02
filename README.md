# EquiFold

This is an official open source code repo of [EquiFold](#citations) developed by the Prescient Design team at Genentech.


## Notes
- This light-weight research version of the code was used to produce figures reported in the manuscript (to be updated soon). We plan to release a higher quality version of the code with additional features in the future.
- There are known issues occasionally seen in predicted structures including nonphysical bond geometry and clashes. We are researching ways to improve the model to minimize such issues.


## Installation and usage
### Environment set up
We used the following set up on a HPC platform equipped with A100 GPUs.
```
conda create -n ef python=3.9 -y
conda activate ef
conda install pytorch=1.11 cudatoolkit=11.3 -c pytorch -y
pip install torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric -f https://data.pyg.org/whl/torch-1.11.0+cu113.html 
pip install e3nn pytorch-lightning biopython pandas tqdm einops
```

For use without GPU, you may try the following:
```
conda create -n ef python=3.9 -y
conda activate ef
conda install pytorch=1.12 -c pytorch -y
conda install pyg -c pyg
pip install e3nn pytorch-lightning biopython pandas tqdm einops
```

### Download model weights
Pytorch model weights and hyper-parameter configs for the models trained on mini-protein and antibody (Ab) datasets are stored in `models` directory.

### Run model predictions
To make predictions using a trained model, users can provide input sequences as a csv table:

```
# for Ab
python run_inference.py --model ab --model_dir models --seqs tests/data/inference_ab_input.csv --ncpu 1 --out_dir out_tests
# for mini proteins
python run_inference.py --model science --model_dir models --seqs tests/data/inference_science_input.csv --ncpu 1 --out_dir out_tests
```



## License
Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at https://www.apache.org/licenses/LICENSE-2.0.

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.


## Citations
If you use the code and/or model, please cite:
```
@article {Lee2022.10.07.511322,
    author  = {Lee, Jae Hyeon and Yadollahpour, Payman and Watkins, Andrew and Frey, Nathan C. and Leaver-Fay, Andrew and Ra, Stephen and Cho, Kyunghyun and Gligorijevic, Vladimir and Regev, Aviv and Bonneau, Richard},
    title   = {EquiFold: Protein Structure Prediction with a Novel Coarse-Grained Structure Representation},
    elocation-id = {2022.10.07.511322},
    year    = {2022},
    doi     = {10.1101/2022.10.07.511322},
    publisher = {Cold Spring Harbor Laboratory},
    URL     = {https://www.biorxiv.org/content/early/2022/10/08/2022.10.07.511322},
    eprint  = {https://www.biorxiv.org/content/early/2022/10/08/2022.10.07.511322.full.pdf},
    journal = {bioRxiv}
}
```

An updated version of the preprint is under preparation.