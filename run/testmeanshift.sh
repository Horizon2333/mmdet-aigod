export CUDA_VISIBLE_DEVICES=3

python tools/test_pkl.py configs/rfla/aigod_faster_r50_rfla_kld_1x.py work_dirs/one_faster_r50_rfla_kld_1x/epoch_12.pth --eval bbox
