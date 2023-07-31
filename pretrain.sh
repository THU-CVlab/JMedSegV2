CUDA_VISIBLE_DEVICES=0 log_silent=1 python3.7 run_ssl.py --model deeplab --layer aspp --channel 256 --dataset pancreas --save checkpoints/deeplab-ssl.pkl -e 50 --lr 5e-6
CUDA_VISIBLE_DEVICES=1 log_silent=1 python3.7 run_ssl.py --model unet --layer down4 --channel 512 --dataset xh --save checkpoints/unet-ssl.pkl -e 50 --lr 1e-5