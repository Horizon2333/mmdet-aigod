export CUDA_VISIBLE_DEVICES=3

python tools/test.py configs/baselines/one_center_r50_1x.py  work_dirs/one_center_r50_1x/epoch_12.pth --eval bbox