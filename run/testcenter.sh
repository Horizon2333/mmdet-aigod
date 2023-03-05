export CUDA_VISIBLE_DEVICES=2

python tools/test.py configs/baselines/aitod_center_r50_1x.py  work_dirs/aitod_center_r50_1x/epoch_27.pth --eval bbox