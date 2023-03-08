export CUDA_VISIBLE_DEVICES=3

python tools/train.py configs/baselines/one_center_r50_1x.py
# python tools/train.py configs/rfla/one_center_r50_rfla_kld_1x.py