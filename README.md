# VLA-Flow (TOMO find-and-pick)
Same as dino_act_vla_model (DINOv2 RGB-D + SigLIP language + suction/done/phone heads), but the
action chunk (30x46 joint deltas) comes from **rectified flow matching**, not the ACT CVAE:
train: x_s = s*x0 + (1-s)*noise, decoder(queries + flow_in(x_s) + time_emb(s)) -> velocity, MSE to (x0 - noise)
infer: 10 Euler steps from noise (model.flow_steps). Suction/done read from the last step.
Files changed vs the ACT version: vla/model.py, vla/losses.py, vla/checkpoint.py, scripts/train_vla.py (schema).
Run: ./run_vlaflow.sh smoke | train | eval     (output runs/vlaflow/best.pt, metrics.csv)
