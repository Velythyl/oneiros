# oneiros
Sim2Sim as a benchmark for Sim2Real

## INSTALL

```bash
python3 -m venv venv
pip install --upgrade pip

# IF YOU WANT TO ENABLE GPU FOR BRAX
pip install --upgrade "jax[cuda11_pip]==0.4.20" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

# IF YOU ARE OK WITH CPU-ONLY BRAX
pip install --upgrade "jax[cpu]"

pip3 install -r requirements.txt

# NOW, MAKE SURE JAX WORKS PROPERLY
python3 does_jax_work.py

mkdir external && cd external && git clone git@github.com:google/brax.git && cd brax && pip install -e . && cd ../..
```

## RENDERING

You might need to use an `LD_PRELOAD` trick if you're using conda. Here: [https://stackoverflow.com/questions/71010343/cannot-load-swrast-and-iris-drivers-in-fedora-35](https://stackoverflow.com/questions/71010343/cannot-load-swrast-and-iris-drivers-in-fedora-35)

Also note that we're not doing anything to the underlying simulators, so videos might look drastically different for brax vs mujoco vs PyBullet, etc.