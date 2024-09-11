### CrossQ: Batch Normalization in Deep Reinforcement Learning for Greater Sample Efficiency and Simplicity

## Setup
Execute the following commands to set up a conda environment to run experiments
```bash
conda create -n crossq python=3.11.5
conda activate crossq
conda install -c nvidia cuda-nvcc=12.3.52

# conda install python=3.11.5
python -m pip install -e .
# here probably you will have to install sth from requirements or just install from entropy_requirements.txt
```

## Running Experiments
The main entry point for running experiments is `train.py`. You can configure experiments with the appropriate environment and agent flags. For more info run `python train.py --help`.

To train **with WandB logging**, run the following command to train a CrossQ agent on the `Humanoid-v4` environment with seed `9`, which will log the results to your WandB entity and project (set these for yourself in `train.py`):
```bash
python train.py --algo crossq --env Humanoid-v4 --seed 9 --wandb_mode 'online'
```
To train **without WandB logging**, run the following command, and in a different terminal run `tensorboard --logdir logs` to visualize training progress:
```bash
python train.py --algo crossq --env Humanoid-v4 --seed 9 --wandb_mode 'disabled'
```

To train **on a cluster**, we provide examples of slurm scripts in `/slurm` to run various experiments, baselines and ablations performed in the paper on a slurm cluster.
These configurations are very cluster specific and probably need to be adjusted for your specific cluster. However, they should surve as a starting point.

<p align="center">
  <img src="http://adityab.github.io/CrossQ/static/images/crossq_efficiency_brief.png" align="center" width="80%"/>
</p>

## Citing this Project and the Paper

To cite our paper and/or this repository in publications:

```bibtex
@inproceedings{
  bhatt2024crossq,
  title={CrossQ: Batch Normalization in Deep Reinforcement Learning for Greater Sample Efficiency and Simplicity},
  author={Aditya Bhatt and Daniel Palenicek and Boris Belousov and Max Argus and Artemij Amiranashvili and Thomas Brox and Jan Peters},
  booktitle={The Twelfth International Conference on Learning Representations},
  year={2024},
  url={https://openreview.net/forum?id=PczQtTsTIX}
}
```

## Acknowledgements

The implementation is built upon code from [Stable Baselines JAX](https://github.com/araffin/sbx/).
