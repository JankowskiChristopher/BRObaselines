conda create -n crossq python=3.11.5 -y
conda activate crossq
conda install -c nvidia cuda-nvcc=12.3.52 -y

python -m pip install -e .
python -m pip install "jax[cuda12_pip]==0.4.19" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html