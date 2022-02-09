import os
from glob import glob


INPUT_PATH = "in"
IMAGES_PATH = os.path.join(INPUT_PATH, "images")
ANNOTS_PATH = os.path.join(INPUT_PATH, "_annotations.csv")

OUTPUT_PATH = "out"
MODEL_PATH = os.path.join(OUTPUT_PATH, "detector.h5")
PLOT_PATH = os.path.join(OUTPUT_PATH, "plot.png")
TEST_FILENAMES = os.path.join(OUTPUT_PATH, "test_images.txt")

# # the following are the same
# os.path.join("out", "images", "saved")
# os.path.sep.join(["out", "images", "saved"])
