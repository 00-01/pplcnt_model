# '''
# rgb ppl detection model output is tranferred to ir model label
#
# rgb data > model_1(rgb ppl cntr) > ouput (label) > model_2(ir ppl cntr) > output
# '''
#
#
# # modeljopen
#
#
# #
#
#
# # output to label
# '''
# 2022-04-18 06:38:06
#
# 1 : { "x": "34", "y": "52", "width": "10", "height": "10" }
# 2 : { "x": "76", "y": "27", "width": "21", "height": "19" }
# 3 : { "x": "17", "y": "35", "width": "29", "height": "18" }
# 4 : { "x": "55", "y": "48", "width": "8", "height": "8" }
# 5 : { "x": "43", "y": "16", "width": "7", "height": "8" }
# '''
#
#
#
# from PIL import Image
# from glob import glob
# from matplotlib import cm, image, patches, pyplot as plt
# import io
#
# import cv2
# import numpy as np
#
# # cut raw img data into n*n array
# # data_path = "/home/z/MVPC10/DATA/mvpc10/**/*.jpg"
# data_path = "/home/z/MVPC10/DATA/*.jpeg"
# #%%
# files = []
# for file in glob(data_path):
#     files.append(file)
# print(len(files))
#
# raw_img = Image.open(files[0]).convert('L')
#
# img_arr = np.array(raw_img)
# w, h = img_arr.shape
#
# r = 40
# center_x, center_y = (w/2)-1, (h/2)-1
#
# # mask = np.array([])
# mask = []
# for x, i in enumerate(img_arr):
#     for y, j in enumerate(i):
#         if abs(center_x-x)+abs(center_y+y) > r:
#             # print(x, y)
#             # mask = np.append(mask, (x,y))
#             mask.append([x,y])
#
#         else:
#             print(x, y)
#
# print(len(mask))
# for i in mask:
#     print(i)
#


#
# import torch
# import detectron2
# from detectron2.utils.logger import setup_logger
# setup_logger()
#
# import numpy as np
# import os, json, cv2, random
#
# from detectron2 import model_zoo
# from detectron2.engine import DefaultPredictor
# from detectron2.config import get_cfg
# from detectron2.utils.visualizer import Visualizer
# from detectron2.data import MetadataCatalog, DatasetCatalog
#
#
# im = cv2.imread("/media/z/e9503728-f419-4a14-9fc0-21e2947af50c/DOWNLOAD/1650501076675.jpg")
# cv2.imshow(im)
# cv2.waitKey(0)
#
# cfg = get_cfg()
# # add project-specific config (e.g., TensorMask) here if you're not running a model in detectron2's core library
# cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
# cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.6  # set threshold for this model
#
# # cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
#
# # Find a model from detectron2's model zoo. You can use the https://dl.fbaipublicfiles... url as well
# cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
# predictor = DefaultPredictor(cfg)
# outputs = predictor(im)
#
# # look at the outputs. See https://detectron2.readthedocs.io/tutorials/models.html#model-output-format for specification
# print(outputs["instances"].pred_classes)
# print(outputs["instances"].pred_boxes)
#
# # We can use `Visualizer` to draw the predictions on the image.
# v = Visualizer(im[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
# out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
# cv2.imshow(out.get_image()[:, :, ::-1])
# cv2.waitKey(0)


import fiftyone as fo
import fiftyone.zoo as foz


dataset = foz.load_zoo_dataset("quickstart")
session = fo.launch_app(dataset)
session.wait()