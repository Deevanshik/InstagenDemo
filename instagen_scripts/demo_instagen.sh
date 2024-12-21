cd mmdetection/

python demo.py \
    --detector_config configs/instagen/instagen-4scale_fd_8xb2-12e_coco_demo.py \
    --detector_ckpt /kaggle/input/instagen-model-weights/pytorch/default/1/instagen-4scale_fd_8xb2-12e_coco.pth \
    --cls_names 'car, train' \
    --score_thr 0.4 \
    --out demo/instagen_pred.jpg
