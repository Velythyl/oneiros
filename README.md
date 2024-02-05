# oneiros
Sim2Sim as a benchmark for Sim2Real

## INSTALL

```bash
python3 -m venv venv
pip install --upgrade pip

# IF YOU WANT TO ENABLE GPU FOR BRAX
pip install --upgrade "jax[cuda11_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

# IF YOU ARE OK WITH CPU-ONLY BRAX
pip install --upgrade "jax[cpu]"

pip3 install -r requirements.txt

# NOW, MAKE SURE JAX WORKS PROPERLY
python3 does_jax_work.py

mkdir external && cd external && git clone git@github.com:google/brax.git && cd brax && pip install -e . && cd ../..
```