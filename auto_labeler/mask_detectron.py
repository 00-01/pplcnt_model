import os

from PIL import Image
from glob import glob
from time import time

import cv2
import numpy as np
import torch
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.data.detection_utils import read_image
from detectron2.engine import DefaultPredictor
from detectron2.structures import Instances
from detectron2.utils.visualizer import Visualizer


save = 1
show_image = 0

threshold = 0.4
segment_type = "semantic"
# segment_type = "panoptic"
# segment_type = "detection"
# segment_type = "keypoint"

machine = "/host"
base = f"{machine}/media/z/0/MVPC10/DATA"
device = "device_03"
path = f"{base}/{device}"

# path = "{base}DOWNLOAD/1650497896553.jpg"
# path = f"{base}DOWNLOAD/1650444796136.jpg"
# path = f"{base}20220513-182408_03_RGB.jpg"
# dir = f"{path}/**/*."

# w, h = 400, 400

cfg = get_cfg()
if segment_type == "semantic":
    file = "COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml"
elif segment_type == "detection":
    file = "COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml"
elif segment_type == "keypoint":
    file = "COCO-Keypoints/keypoint_rcnn_R_101_FPN_3x.yaml"
elif segment_type == "panoptic":
    file = "COCO-PanopticSegmentation/panoptic_fpn_R_101_3x.yaml"
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = threshold
cfg.merge_from_file(model_zoo.get_config_file(file))
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(file)


def crop_img(arr):
    new_arr = arr
    side = w//5
    # # side cut
    new_arr[:, :side] = 0
    new_arr[:, w-side:] = 0
    # # top cut
    new_arr[:h//4, :] = 0

    # lr, tb = w//4, h//4
    # new_arr = new_arr[:, w//4:w-w//4]
    # new_arr = new_arr[h//4:, :]

    return new_arr


def output_selector(outputs):
    # global masks_bool
    # global boxes_arr
    obj = Instances(image_size=(w, h))
    cls = outputs['instances'].pred_classes
    indx_to_remove = (cls != 0).nonzero().flatten().tolist()

    cls = np.delete(cls.cpu().numpy(), indx_to_remove)
    cls = torch.tensor(cls).to('cuda:0')
    obj.set('pred_classes', cls)

    scores = outputs["instances"].scores
    scores = np.delete(scores.cpu().numpy(), indx_to_remove)
    scores = torch.tensor(scores).to('cuda:0')
    obj.set('scores', scores)

    masks = outputs['instances'].pred_masks
    masks = np.delete(masks.cpu().numpy(), indx_to_remove, axis=0)
    bool_mask = [mask for i in masks for mask in i]
    masks = torch.tensor(masks).to('cuda:0')
    obj.set('pred_masks', masks)

    # boxes = outputs['instances'].pred_boxes
    # boxes = np.delete(boxes.tensor.cpu().numpy(), indx_to_remove, axis=0)
    # boxes = torch.tensor(boxes).to('cuda:0')
    # obj.set('pred_boxes', boxes)

    return obj, bool_mask


files = glob(f'{path}/**/*.jpg', recursive=True)
# print(files)

for i, j in enumerate(files):
    result_img_path = j.replace(f"{device}", f"{device}/output/img")
    result_mask_path = j.replace(f"{device}", f"{device}/output/mask")
    # if j in result_img_path: break

    img = read_image(j, format="BGR")
    w, h = img.shape[0], img.shape[1]
    # img = Image.open(i)
    # arr = np.array(img)
    # img = mask_img(arr)

    img = crop_img(img)

    start = time()

    predictor = DefaultPredictor(cfg)
    v = Visualizer(img[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.0)

    if segment_type == "panoptic":
        panoptic_seg, segments_info = predictor(img)["panoptic_seg"]
        out = v.draw_panoptic_seg_predictions(panoptic_seg.to("cpu"), segments_info)
    else:
        outputs = predictor(img)
        obj, bool_mask = output_selector(outputs)
        out = v.draw_instance_predictions(obj.to("cpu"))

    if len(bool_mask) > 0:
        bool2arr = np.array(bool_mask, dtype=np.uint8)
        mask_img = bool2arr*255

        if save == 1:
            img_path = os.path.dirname(result_img_path)
            mask_path = os.path.dirname(result_mask_path)

            if not os.path.exists(img_path): os.makedirs(img_path)
            if not os.path.exists(mask_path): os.makedirs(mask_path)

            im = Image.fromarray(mask_img)
            im.save(result_mask_path)
            out.save(result_img_path)

        print(f"{i} {time()-start:.2f}")

        if show_image == 1:
            cv2.namedWindow(segment_type, cv2.WINDOW_NORMAL)
            cv2.imshow(segment_type, out.get_image()[:, :, ::-1])

            win_name = "mask"
            cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
            cv2.imshow(win_name, mask_img)

            if cv2.waitKey(0) == 27:
                cv2.destroyAllWindows()
            break
    else:
        print(f"{i} no mask")
