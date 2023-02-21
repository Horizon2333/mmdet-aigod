export CUDA_VISIBLE_DEVICES=5

python tools/test.py configs/rfla/aitod_faster_r50_rfla_kld_1x.py work_dirs/aitod_faster_r50_rfla_kld_1x/epoch_12.pth --eval bbox
