1. make a venv with python3.8
2. install torch with 
    pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
3. install jax-cuda with
    pip install --upgrade "jax[cuda11_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
4. Test if Jax-cuda works by running `docs/scratch.py`
5. install issac gym 
  go to isaac gym folder 
  pip3 install -e .
  pip3 install gym
6. install libpython
  sudo apt-get install libpython3.8
7. if you get this error: `RuntimeError: No gym module found for the active version of Python (3.9)`
   it means that you're missing a .so file. Restart from scratch with python3.8 :)))))))))) yay
8. test isaac gym by goign to python/examples and running
    python3 domain_randomization.py --cuda
9. pip install wheel
10. install brax
     pip3 install https://github.com/Velythyl/brax/archive/refs/heads/vsys.zip --no-cache
11. install the rest of it all
    pip3 install -r requirements.txt
12. sudo apt-get install python3-dev