from pathlib import Path
from IPython.display import HTML,display
src=list(Path("/kaggle/input").rglob("COMPLETE.json"))[0].parent
# Supplement missing latency measurements using saved checkpoints only. No test loader.
import sys,subprocess,json,base64
meta=json.loads((src/'runs/F01/seed0/config.json').read_text())
subprocess.check_call([sys.executable,'-m','pip','install','-q','timm=='+meta['timm']])
sys.path.insert(0,str(src/'day2_lab_code'))
import torch,pandas as pd
from dataclasses import fields,replace
from train import Config,set_seed
from experiments import load_checkpoint
from benchmark import latency_report
from inference import fuse_conv_bn
assert torch.cuda.is_available()
set_seed(0)
rows=[]
def saved(exp):
    m=json.loads((src/'runs'/exp/'seed0/config.json').read_text())
    return replace(Config(**{f.name:m[f.name] for f in fields(Config)}),out_dir=str(src/'runs'))
for exp in ['B01','B02','B03','B04','B05']:
    model=load_checkpoint(saved(exp))
    rows.append({'exp_id':exp,**latency_report(model,1,224),'timm':meta['timm']})
    del model;torch.cuda.empty_cache()
model=load_checkpoint(saved('T06_ls'))
for exp,k,dtype in [('I00',1,'fp32'),('I01_hflip_prob',2,'fp32'),('I02_hflip_logit',2,'fp32'),('I03_temperature',1,'fp32'),('I04_fused',1,'fp32'),('I05_fp16',1,'fp16')]:
    rows.append({'exp_id':exp,**latency_report(fuse_conv_bn(model) if exp=='I04_fused' else model,32,224,dtype=dtype,k_views=k),'timm':meta['timm']})
pd.DataFrame(rows).to_csv('/kaggle/working/latency_supplement.csv',index=False)
print('LATENCY_READY',len(rows))
display(HTML('<textarea aria-label="Latency CSV" rows="3" cols="80">'+base64.b64encode(Path('/kaggle/working/latency_supplement.csv').read_bytes()).decode()+'</textarea>'))

# Complete Summary latency for the remaining top-10 recipes. Saved checkpoints only.
for exp in ['T03_color','T05_mixup','T07_focal']:
    model=load_checkpoint(saved(exp))
    rows.append({'exp_id':exp,**latency_report(model,1,224),'timm':meta['timm']})
    del model;torch.cuda.empty_cache()
pd.DataFrame(rows).to_csv('/kaggle/working/latency_supplement.csv',index=False)
print('LATENCY_READY',len(rows))
display(HTML('<textarea aria-label="Latency CSV completed" rows="3" cols="80">'+base64.b64encode(Path('/kaggle/working/latency_supplement.csv').read_bytes()).decode()+'</textarea>'))
