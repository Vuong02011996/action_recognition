# Install 
```commandline
+ conda create -n openmmlab python=3.8 pytorch=1.10 cudatoolkit=11.3 torchvision -c pytorch -y
+ conda activate openmmlab
+ pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/cu113/torch1.10/index.html
    + Don't use: 
        + pip3 install openmim
        + mim install mmcv-full
+ git clone https://github.com/open-mmlab/mmpose.git
+ cd mmpose
+ pip3 install -e .
```

# Test
+ docs/en/getting_started.md
```
    python demo/top_down_img_demo.py \
    ${MMPOSE_CONFIG_FILE} ${MMPOSE_CHECKPOINT_FILE} \
    --img-root ${IMG_ROOT} --json-file ${JSON_FILE} \
    --out-img-root ${OUTPUT_DIR} \
    [--show --device ${GPU_ID}] \
    [--kpt-thr ${KPT_SCORE_THR}]
    
    python demo/top_down_img_demo.py \
    configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/hrnet_w48_coco_256x192.py \
    https://download.openmmlab.com/mmpose/top_down/hrnet/hrnet_w48_coco_256x192-b9e0b3ab_20200708.pth \
    --img-root tests/data/coco/ --json-file tests/data/coco/test_coco.json \
    --out-img-root vis_results
    
    python demo/top_down_img_demo.py \
    /media/vuong/AI1/Github_REF/mmpose/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/alexnet_coco_256x192.py \
    /media/vuong/AI1/My_Github/pose_estimate/Mmpose/Models/alexnet_coco_256x192-a7b1fd15_20200727.pth \
    --img-root tests/data/coco/ --json-file tests/data/coco/test_coco.json \
    --out-img-root vis_results
```

```commandline
            "keypoints": [
                "nose",
                "left_eye",
                "right_eye",
                "left_ear",
                "right_ear",
                "left_shoulder",
                "right_shoulder",
                "left_elbow",
                "right_elbow",
                "left_wrist",
                "right_wrist",
                "left_hip",
                "right_hip",
                "left_knee",
                "right_knee",
                "left_ankle",
                "right_ankle"
            ],
```
# Models
+ [model](https://mmpose.readthedocs.io/en/latest/topics/body%282d%2Ckpt%2Csview%2Cimg%29.html)

# Performance 
+ Hardware.
+ FPS 33 -> Alpha pose 16

# Note 
## Flow té ngã 
+ Nhận diện khung xương -> nhận diện hành động.
+ Fall detection 
  + Model detect pose (Select)
    + Mmpose (FPS 33)
    + PoseEstimate (FPS 16)
  + Model actions recognition
    + Train with Mmpose
    + Train with PoseEstimate
