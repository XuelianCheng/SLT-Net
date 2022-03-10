CUDA_VISIBLE_DEVICES=0 python train_video.py --dataset='MoCA' --batchsize=8 --trainsize 352 --lr 1e-6 --epoch 200 \
                       --pretrained_cod10k='snapshot/Net_epoch_cod10k.pth' \
                       --save_path='./snapshot/MoCA_short_1e-6/' 