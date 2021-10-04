CUDA_VISIBLE_DEVICES=0 python test_video.py --testsplit 'MoCA-Video-Test' \
--pth_path 'snapshot/Net_epoch_MoCA_short_term.pth'  2>&1 |tee ./log.txt
