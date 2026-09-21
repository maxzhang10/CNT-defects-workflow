# %%
import os
import torch

from dpnegf.runner.NEGF import NEGF
from dpnegf.utils.argcheck import normalize_run
from dptb.nn.build import build_model
import json

from dpnegf.utils.loggers import set_log_handles
import logging
from pathlib import Path

import matplotlib.pyplot as plt




# %%
INPUT_file =  "./input.json" 

#model =  "./nnsk_dftb.json"
model_path = "/data/run01/scxk180/dpnegf/CNT-defects-workflow/5_5/500K/5_5/5775_L008_0.1016A-1/replica_001/dpnegf/2000/nnenv.ep181.pth"

structure =  "./5_5.xyz" 
output = "output"  

if os.path.exists(output):
    os.system('rm -rf %s' % output)


negf_json = json.load(open(INPUT_file))
negf_json = normalize_run(negf_json)


log_path = output+'/log'
log_level = logging.INFO
set_log_handles(log_level, Path(log_path) if log_path else None)

# model = build_model(model,model_options= model_json['model_options'],
#                     common_options=model_json['common_options'])



model = build_model(
    model_path,
    common_options={"device":"cpu"},
)
model.device = "cpu"

# %%
negf = NEGF(
    model=model,
    structure=structure,
    results_path=output,  
    **negf_json['task_options']
)
   
negf.compute()

# %%
negf_out = torch.load('./output/negf.out.pth')

# %%
negf_out.keys()

# %%
plt.plot(negf_out['uni_grid'], negf_out['T_avg'])
plt.xlabel('Energy (eV)')
plt.ylabel('Transmission')
plt.title('Transmission vs Energy')
plt.grid()
plt.savefig("Transmission.png")
plt.close()

