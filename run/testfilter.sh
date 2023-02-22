export CUDA_VISIBLE_DEVICES=3

python tools/test.py configs/rfla/filter_faster_r50_rfla_kld_1x.py work_dirs/filter_faster_r50_rfla_kld_1x/epoch_12.pth --eval bbox
