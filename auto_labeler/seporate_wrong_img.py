import os.path
from glob import glob

save = 1
show_image = 0


base = "/host/media/z/0/MVPC10/DATA"
device = "device_03"
path = f"{base}/{device}"


file0 = glob(f'{path}/result/delete/*', recursive=True)
target = glob(f'{path}/mask/*.png', recursive=True)

subject = [os.path.basename(i).replace(".jpg", "") for i in file0]

# print(target)

for i in subject:
    os.replace(f"{path}/mask/{i}.png", f"{path}/mask/delete/{i}.png")
