import warnings

warnings.filterwarnings('ignore')
from ultralytics import RTDETR,YOLO,RTDETRyh1

if __name__ == '__main__':
    model = RTDETR('/home/liyihang/lyhredetr/runs/rfrb/v10noapex_6/weights/best.pt')  # select your model.pt path
    model.predict(source='p15/RFRB/',
                  project='heat/RFRB',
                  name='noHEplus',
                  save=True,
                  visualize=True,  # visualize model features maps

                  )
