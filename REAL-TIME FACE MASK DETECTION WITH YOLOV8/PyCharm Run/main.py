from ultralytics import YOLO
from PIL import Image

# modeli yükle
model = YOLO('best.pt')  # load a pretrained model

# modeli kullanarak video ve webcam görüntüsü ile nesne tahmini yap
sonuc = model.predict(source="1",show=True)
