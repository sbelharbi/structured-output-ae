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
runner = "python cnn-lfpw_2l.py"

optims = ["momentum"]
lrs = [1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10]
# Read the cmd lines
cmd_file = "cmd-python-to-submit.txt"
i = 0

for opt in optims:
    for lr in lrs:
        time_exp = DT.datetime.now().strftime('%m_%d_%Y_%H_%M_%s')
        rnd = np.random.rand(1)[0]
        name_job = opt + "_" + str(lr) + "_" + str(i) + "_" + str(rnd) + "_" +\
            time_exp + ".sl"
        with open(folder_jobs + "/" + name_job, "w") as fjob:
            fjob.write(gpu_cont + "\n")
            cmd = " ".join([flags, runner, opt, str(lr), "\n"])
            fjob.write(cmd)
        fsh.write("sbatch ./" + folder_jobs + "/" + name_job + " \n")
        fsh.write("sleep 10s \n")
        time.sleep(1)
        i += 1

fsh.close()
fgpu.close()
os.system("chmod +x " + bash_name)
