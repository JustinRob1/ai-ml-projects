import numpy as np
import torch
from PIL import Image
import PIL
from tensorflow.keras.models import load_model
import os

def detect_and_segment(images):
    N = images.shape[0]

    # pred_class: Your predicted labels for the 2 digits, shape [N, 2]
    pred_class = np.empty((N, 2), dtype=np.int32)
    # pred_bboxes: Your predicted bboxes for 2 digits, shape [N, 2, 4]
    pred_bboxes = np.empty((N, 2, 4), dtype=np.float64)
    # pred_seg: Your predicted segmentation for the image, shape [N, 4096]
    pred_seg = np.empty((N, 4096), dtype=np.int32)

    obj_model = torch.hub.load('ultralytics/yolov5', 'custom', path='best.pt', force_reload=True)
    seg_model = load_model('unet_model.h5')

    img_dir = 'images'
    to_jpeg(images, img_dir)

    for i in range(N):
        img = images[i].reshape(64, 64, 3)
        results = obj_model(img, size=64)
        num_objects = len(results.pred[0])

        for j in range(2):
            if j < num_objects:
                pred_class[i][j] = results.pred[0][j][5]
                y_min = round(results.pred[0][j][1].item())
                x_min = round(results.pred[0][j][0].item())
                y_max = y_min + 28
                x_max = x_min + 28
                pred_bboxes[i][j] = [y_min, x_min, y_max, x_max]
            else:
                pred_class[i][j] = -1  
                pred_bboxes[i][j] = np.zeros(4)  

        if pred_class[i][0] > pred_class[i][1]:
            temp_class = pred_class[i][0]
            pred_class[i][0] = pred_class[i][1]
            pred_class[i][1] = temp_class
            temp_bbox = pred_bboxes[i][0].copy()
            pred_bboxes[i][0] = pred_bboxes[i][1].copy()
            pred_bboxes[i][1] = temp_bbox.copy()

    img_files = os.listdir(img_dir)
    for i, img_file in enumerate(img_files):
        try:
            img = Image.open(os.path.join(img_dir, img_file))
        except PIL.UnidentifiedImageError:
            continue
        img = img.resize((64, 64))
        img_array = np.array(img).astype(np.float32) / 255
        img_array = img_array.reshape((1, 64, 64, 3))
        pred_mask = seg_model.predict(img_array, verbose=0)
        pred_mask = np.argmax(pred_mask, axis=-1)
        pred_mask = pred_mask[0]
        pred_seg[i] = pred_mask.reshape(-1)

    return pred_class, pred_bboxes, pred_seg

def to_jpeg(images, img_dir):
    os.makedirs(img_dir, exist_ok=True)
    for i in range(images.shape[0]):
        img = images[i].squeeze()
        img = img.reshape((64, 64, 3)).astype(np.uint8)
        img_obj = Image.fromarray(img)
        img_obj.save(os.path.join(img_dir, f'{i}.jpg'))