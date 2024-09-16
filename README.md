# oneiros

Oneiros is the ancient greek personifications of dreams, of reflections of our world. 
In this project, we utilize many different reflections of our world known as "simulators" to train policies in many different visions of our world.
We implement "sim+sim" training for increased policy transferability, and "sim2sim" transfer as a benchmark for sim2real.
We find that sim+sim training endows PPO policies with a greater transferability, both sim2sim and sim2real.

## INSTALL

```bash
python3 -m venv venv
pip install --upgrade pip

# IF YOU WANT TO ENABLE GPU FOR BRAX
# YOU WILL MOST LIKELY NEED TO TOY AROUND WITH THE BRAX VERSION
# FOR US, 0.4.16 and 0.4.23 WORKED BEST.
pip install --upgrade "jax[cuda11_pip]==0.4.23" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

pip3 install -r requirements.txt

# NOW, MAKE SURE JAX WORKS PROPERLY
python3 does_jax_work.py
```

If you run into installation errors (missing pip packages, incompatible versions, etc.), please have a look at the CCDB folder, where we include a way to build using a Dockerfile and converting the Docker images to Singularity. You might be able to use those as inspiration to fix your installation. 
Otherwise, shoot me a message at `charlie.gauthier [at] mila.quebec`. 


## TRAINING

Just run `main.py`. There's also a hydra/submitit combination to help you run on compute clusters. In the CCDB folder, you can also find a Dockerfile and a way to build a Singularity file for use on clusters that do not support submitit. The `hydra_splat.py` file can be helpful to set up `hydra --multirun` on such clusters.

## RENDERING

You might need to use an `LD_PRELOAD` trick if you're using conda. Here: [https://stackoverflow.com/questions/71010343/cannot-load-swrast-and-iris-drivers-in-fedora-35](https://stackoverflow.com/questions/71010343/cannot-load-swrast-and-iris-drivers-in-fedora-35)

Also note that we're not doing anything to the underlying simulators, so videos might look drastically different for brax vs mujoco, etc.

## CITATION

Bibtex file to come!
