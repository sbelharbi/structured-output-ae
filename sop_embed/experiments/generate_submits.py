import datetime as DT
import os
import numpy as np
import time

gpu = "p100.sl"
folder_jobs = "jobs"
bash_name = "submit.sh"
fsh = open(bash_name, "w")
fsh.write("#!/usr/bin/env bash \n")
max_rep = 7
flags = "THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 "
rep = 0
fgpu = open(gpu, "r")
gpu_cont = fgpu.read()

# Read the cmd lines
cmd_file = "cmd-python-to-submit.txt"
i = 0
with open(cmd_file) as f:
    cnt = f.readlines()
    cnt = [x.strip() for x in cnt]
    for l in cnt:
        if l.startswith("python"):
            time_exp = DT.datetime.now().strftime('%m_%d_%Y_%H_%M_%s')
            rnd = np.random.rand(1)[0]
            name_job = str(i) + "_" + str(rnd) + "_" + time_exp + ".sl"
            with open(folder_jobs + "/" + name_job, "w") as fjob:
                fjob.write(gpu_cont + "\n")
                fjob.write(flags + " " + l + " \n")
            fsh.write("sbatch ./" + folder_jobs + "/" + name_job + " \n")
            fsh.write("sleep 3s \n")
            time.sleep(3)
            i += 1

fsh.close()
fgpu.close()
os.system("chmod +x " + bash_name)
