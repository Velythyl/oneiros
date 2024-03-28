import os
import subprocess
from time import sleep

pid = os.getpid()
command=f"pgrep -fl python | awk '!/{pid}/{{print $1}}' | xargs kill"
process = subprocess.Popen(command, shell=True)
process.wait()
sleep(1)
print("allo")