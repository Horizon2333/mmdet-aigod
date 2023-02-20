export CUDA_VISIBLE_DEVICES=5

# evaluate FCOS
# python tools/test.py configs/rfla/aitod_fcos_r50_rfla_kld_1x.py pretrained/aitod_fcos_r50_rfla_kld_1x.pth --eval bbox

# evaluate Faster R-CNN
# python tools/test.py configs/rfla/aitod_faster_r50_rfla_kld_1x.py pretrained/aitod_faster_r50_rfla_kld_1x.pth --eval bbox

# evaluate Cascade R-CNN
python tools/test.py configs/rfla/aitod_cascade_r50_rfla_kld_1x.py pretrained/aitod_cascade_r50_rfla_kld_1x.pth --eval bbox