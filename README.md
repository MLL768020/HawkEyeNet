# HawkEyeNet
 

HawkEye Conv-Driven YOLOv10 with Advanced Feature Pyramid Networks for Small Object Detection in UAV Imagery
1.
![HawkEyeNet](https://github.com/user-attachments/assets/ab60ca11-08f8-4d25-8052-3735216822a5)

The circles represent the performance of HawkEyeNet with different convolution types, with circle size indicating computational requirements. The horizontal axis shows the model's parameter count, and the vertical axis displays testing accuracy on the RFRB dataset. Models closer to the top left corner are superior.
2.data
①RFRB download:https://github.com/CV-Wang/RapeNet
②VisDrone2019 download: https://github.com/VisDrone
③In the supplementary materials, the AI-TOD dataset is a UAV dataset, but it differs from the dense area scenarios primarily studied in this paper. However, to more comprehensively validate the effectiveness and applicability of the proposed method, this study includes a comparison experiment between the proposed method and numerous baseline models in the supplementary materials. The download link for the dataset is: https://github.com/jwwangchn/AI-TOD
3.weights [path:runs/plus](https://drive.google.com/drive/folders/1cTVeXKHbkHlfAKGTGurVA5QHY78VFut3?usp=drive_link)
4.train : CUDA_VISIBLE_DEVICES= 1 python train.py
5. val : val.py --para: weight=weights; cfg: ultralytics/cfg/models/rt-detr/llf/vis/....yaml
6. detect: detect.py
7. coco metrice: get_coco_metrice   output: AP AP50 APS APM APL(RFRB and AI-TOD do not contain large objects. APL= -1)    --para: weight= weights
