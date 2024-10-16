import warnings

warnings.filterwarnings('ignore')
from ultralytics import RTDETRyh1,YOLO,RTDETRpyh1,RTDETR
# RTDETR giou
if __name__ == '__main__':

    model = RTDETRyh1('ultralytics/cfg/models/rt-detr/llf/vis/v10plus.yaml')
    #model.load(weights='weights/yolov8l.pt')  # loading pretrain weights
    model.train(data='/home/liyihang/lyhredetr/dataset/VisDrone.yaml',
                cache=False,
                imgsz=640,
                epochs=300,
                batch=8,
                workers=4,
                device=0,
                resume ='/home/liyihang/lyhredetr/plus/visdrone/v10plus9/weights/last.pt',
                project='/home/liyihang/lyhredetr/plus/visdrone',
                name='v10plus',
                )
