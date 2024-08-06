# models/identification_model.py

import torch
import torchvision.transforms as T
from PIL import Image
import os

# Load pre-trained YOLO model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
model.eval()

def identify_objects(object_images_dir="data/segmented_objects/"):
    descriptions = {}
    for object_image in os.listdir(object_images_dir):
        image_path = os.path.join(object_images_dir, object_image)
        image = Image.open(image_path)
        transform = T.Compose([T.ToTensor()])
        image = transform(image)

        # Get predictions
        with torch.no_grad():
            results = model(image)
        
        descriptions[object_image] = results.pandas().xyxy[0].to_dict(orient="records")

    return descriptions

# Test the function
if __name__ == "__main__":
    identify_objects()
