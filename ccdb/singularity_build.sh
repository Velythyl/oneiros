#sudo docker build --tag oneiros:latest .
sudo SINGULARITY_DISABLE_CACHE=1 APPTAINER_NOHTTPS=1 apptainer build oneiros.sif docker-daemon://oneiros:latest