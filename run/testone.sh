export CUDA_VISIBLE_DEVICES=5

python tools/test.py configs/rfla/one_faster_r50_rfla_kld_1x.py work_dirs/one_faster_r50_rfla_kld_1x/epoch_1.pth --eval bbox
