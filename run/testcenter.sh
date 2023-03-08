export CUDA_VISIBLE_DEVICES=3

python tools/test.py configs/baselines/aitod_center_r50_1x.py  work_dirs/aitod_center_r50_1x/epoch_10.pth --eval bbox