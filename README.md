# BRO baselines
Repository contains baselines for the BRO project.
🔗 https://arxiv.org/abs/2405.16158
Baselines come from official implementations.

Baselines:
- SAC
- TD3
- CrossQ
- TD-MPC2
- BRO (code not present in this repository, can be accessed here:
🔗 https://github.com/naumix/BiggerRegularizedOptimistic
)

Benchmark environments:
- DMC
- MetaWorld
- Maniskill2
- Gym (some of the agents)

## Installation

### Conda
To install Conda, run:
Normally conda is installed in home directory (~/). but on clusters due to memory limit, it is better to install it in
a directory where the quota is bigger.

```bash
mkdir -p ./miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ./miniconda3/miniconda.sh
bash ./miniconda3/miniconda.sh -b -u -p ./miniconda3
rm -rf ./miniconda3/miniconda.sh
./miniconda3/bin/conda init bash
````

### TD3
Create Conda environment with Python 3.8

```bash
conda create -n td3 python=3.8
~/miniconda3/envs/td3/bin/pip install -r entropy_requirements.txt
```

If the command:
```bash
which pip
```
Correctly shows the path to pip in the conda environment, just `pip` can be run.

Then run `main.py` to start training. See the optional flags to TD3.

```bash
python main.py
```

### SAC
Create Conda environment with Python 3.8 same as in TD3.

Run `train.py` to start training. See the hydra config for arguments.

### Scripts
Repository contains some scripts used for the evaluation of time and memory usage of the agents.