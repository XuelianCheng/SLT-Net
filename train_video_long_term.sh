CUDA_VISIBLE_DEVICES=0 python train_video_long_term.py --dataset='MoCA' --batchsize=2 --lr 1e-5 --epoch 101 \
                        --short_pretrained='./snapshot/Net_epoch_MoCA_short_term_pseudo.pth' \
                        --save_path='./snapshot/MoCA/long_term_1e-5/' \
                        --threads 8 --input_length 10 --fsampling_rate 1 2>&1 |tee ./log.txt 

