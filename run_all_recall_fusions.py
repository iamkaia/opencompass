#!/usr/bin/env python 
# -*- coding: utf-8 -*- 
import subprocess 

tasks = ["sst2", "squad2", "iwslt2017", "race", "medmcqa"] 

for t in tasks: 
    out_dir = f"recall_fused_{t}" 
    cmd = [ 
        "python", "recall_pipeline_full.py",
        "--anchor_task", t,
        "--output_dir", out_dir 
    ] 
    print(" ".join(cmd))
    subprocess.run(cmd)
