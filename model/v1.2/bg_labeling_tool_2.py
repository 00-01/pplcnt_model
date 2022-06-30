## pyinstaller -F -w bg_labeling_tool1.py
import binascii
from glob import glob
import os
from tkinter import *
from tkinter import filedialog, messagebox
import traceback

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image, ImageTk


DEBUG = 1
BG = 2
SAVE_IMG = 0
AUTO_RUN = 0
# DRAW_CNT = 1

h, w = 80, 80
if BG == 1:  size = 12
else:  size = 10
re_size1, re_size2 = (w*size, h*size), (w*size, h*size)

min, mid, max = 0, 128, 255
error = 239
cnt_thresh = 1
i, j = 0, 0
new_label = 0

bgq = []
bgq_length = 30

x1, y1, x2, y2 = 120, 70, 0, 70
circle = [[1,1]]

df = pd.DataFrame()
save_dir = f"out"
save_csv_path = f"{save_dir}/output.csv"
data_dir = f"/media/z/0/MVPC10/DATA/v1.1/RAW/device_03"

################################################################ IMAGE FUNCTION

def drop_err(csv):
    img_list = glob(f"{save_dir}/*.png")
    df = pd.read_csv(save_csv_path)
    for i in range(len(df)):
        if f"out/{df.iloc[i, 0]}" not in img_list:
            df.iloc[i] = np.nan
    df = df.dropna()
    df.iloc[:, 1] = df.iloc[:, 1].astype(np.uint8)
    df.to_csv("output(err_dropped).csv", index=False)


def dataset_maker(img, label):
    data = [img, label]
    df = pd.DataFrame(data, columns=['img', 'cnt'])
    # TODO: finish auto making dataset


def histogramer(img):
    im = img.flatten()
    # counts, bins = np.histogram(im, range=(0,255))
    # plot histogram centered on values 0..255
    # plt.bar(bins[:-1]-0.5, counts, width=1, edgecolor='none')
    # plt.bar(bins[:-1]-0.5, counts, width=1, edgecolor='none')
    # plt.xlim([-0.5, 255.5])
    plt.hist(im, bins=range(img.min(),img.max(),1), histtype='bar', edgecolor='yellow', color='green')

    # n, bins, patches = plt.hist(x=im, bins='auto', color='#0504aa', alpha=0.7)
    # plt.grid(axis='both', alpha=0.5)
    # plt.xlabel('Value')
    # plt.ylabel('Frequency')
    # Set a clean upper y-axis limit.
    # maxfreq = n.max()
    # plt.ylim(ymax=np.ceil(maxfreq/10)*10 if maxfreq%10 else maxfreq+10)
    plt.show()


def crop_img(arr, W, H, X, Y, a, b, R, case=0):
    new_arr = arr
    if case == 1:
        for x in range(W):
            for y in range(H):
                if (((X-x)**2)/a**2)+(((Y-y)**2)/b**2) > R**2:
                    new_arr[y, x] = 0
    new_arr = new_arr[:, W//5:W-W//5]
    new_arr = new_arr[H//4:, :]

    return new_arr


def bg_filter(target):
    bg = np.zeros([h, w], dtype=int)
    for i in bgq:
        bg += i
    bg //= len(bgq)

    img = target-bg

    low = 8
    img[img < low] = 0
    img -= img.min()

    # high = 255
    # img *= high//img.max()
    # img[img > high] = high

    # img = img1 * img
    # img[abs(img) < thresh] = img.min()
    # histogramer(img)

    # bg_img = cv2.cvtColor(bg_img)
    # clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(2, 2))
    # bg_img = clahe.apply(bg_img)


    return img.astype(np.uint8)


def bg_mkr(img):
    global bgq
    error1 = len(img[img > 237])
    error2 = len(img[img < 1])
    if error1 > 512 or error2 > 256:
        log(f"{i}: white-{error1}, black-{error2}")
        return 0
    else:
        # log(f"{i}: white-{error1}, black-{error2}")
        bgq.insert(0, img)
        if len(bgq) > bgq_length:  bgq.pop(-1)
        else:  pass
        bg = bg_filter(img)
        return 1, bg


def convert_to_format():
    global dataset
    dataset['count'] = dataset['count'].fillna(0)
    df = pd.read_csv(file)
    df1 = df.replace({'.png': '.jpg'}, regex=True)
    df1 = df1.rename(columns={'img_name': 'rgb_name'})
    df = df.drop(df.columns[1], axis=1)
    dataset = pd.concat([df, df1], axis=1)
    dataset["count"] = ""


def process_image():
    global base_name, df, img1, img2, resized_image1, resized_image2

    image1 = cv2.imread(rgb, 1)
    image2 = cv2.imread(ir, 0)
    image1 = image1[:, :, ::-1]

    if BG != 0:
        num, bg = bg_mkr(image2)
        if BG == 1:  image1 = bg
        elif BG == 2:  image2 = bg

    if SAVE_IMG == 1:
        base_name = os.path.basename(ir)
        save_img_path = f"{save_dir}/{base_name}"
        # new_row = {'img': base_name, 'cnt': auto_cnt}
        # df = df.append(new_row, ignore_index=True)
        new_row = pd.DataFrame({'img':[base_name], 'cnt':[auto_cnt]})
        df = pd.concat([df, new_row], axis=0, ignore_index=True, sort=False)
        bg.save(save_img_path)

    return num


def show_image(img1, img2):
    img1_H, img1_W = image1.shape[0], image1.shape[1]
    img2_H, img2_W = image2.shape[0], image2.shape[1]
    img1 = crop_img(image1, img1_H, img1_W, img1_H//2, img1_W, 16, 38, 40, 0)
    img2 = crop_img(image2, img2_H, img2_W, img2_H//2, img2_W, 3.2, 7.6, 8, 1)

    interpolation = cv2.INTER_NEAREST  ## CUBIC LINEAR AREA
    resized_image1 = cv2.resize(img1, (re_size1), interpolation=interpolation)
    resized_image2 = cv2.resize(img2, (re_size2), interpolation=interpolation)

    if AUTO_RUN == 0:
        image111 = ImageTk.PhotoImage(Image.fromarray(img1))
        image222 = ImageTk.PhotoImage(Image.fromarray(img2))
        LEFT_IMG.configure(image=image111)
        RIGHT_IMG.configure(image=image222)
        LEFT_IMG.image111 = image111
        RIGHT_IMG.image222 = image222

def clear():
    count_text.set(0)
    index.delete(0, END)
    previous.delete(0, END)

def circle(img, x, y, r=1):
    new_img = cv2.circle(img, center=(x, y), radius=r, color=(255, 0, 0), thickness=1, lineType=1)
    show_image(resized_image1, resized_image2)
    return new_img

def uncircle():
    pass

def draw_txt(img, label, location):
    new_txt = cv2.putText(img, str(label), org=location, fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=3, color=(255, 0, 255), thickness=5, lineType=1)
    show_image(resized_image1, resized_image2)
    return new_txt

################################################################ EVENT FUNCTION

def log(info):
    if DEBUG == 1:
        print(info)

def change(e, idx):
    global i, ir, rgb, auto_cnt, cnt
    clear()
    i += idx
    data, auto_cnt, cnt = dataset.iloc[i, 0], dataset.iloc[i, 1], dataset.iloc[i, 2]

    ir = glob(f"{data_dir}/**/{data}.png")[0]
    rgb = glob(f"{data_dir}/**/{data}.jpg")[0]

    image1_text.set(f"{rgb}")
    image2_text.set(f"{ir}")

    try:
        if process_image():
            index.insert(0, i)
        else:
            cnt = -1
            index.insert(0, f"{i}: ERROR!")
    except FileNotFoundError as FE:
        log(f"{FE}: {i}")
        traceback.print_exc()
        cnt = -2
    except IndexError as IE:
        log(f"{IE}: {i}, FINISHED")
        traceback.print_exc()
        i -= idx
        # save()
    except TypeError as TE:
        log(f"{TE}: {i}")
        traceback.print_exc()
    count_text.set(f"{cnt}")
    previous.insert(0, auto_cnt)

    if AUTO_RUN == 1:  next(e)

def detect_keypress(e):
    log(e.keysym, e.keysym == 'a')
    log(e.char)
    log(e)

def change_count(cnt):
    global new_label
    new_label = count_text.get()+cnt
    count_text.set(new_label)

def get_count():
    cnt = count_text.get()
    dataset.iloc[i,2] = cnt
    dataset.iloc[i,1] = dataset.iloc[i,1].astype(np.uint8)

def get_mouse_xy(e):
    x, y = e.x, e.y
    log(f"{x}, {y}")
    return x, y

################################################################ EVENT

def open_bin():
    with open(ir, 'rb') as file:
        byte = binascii.hexlify(file.read(1))
        byte = binascii.hexlify(file.read(1))+byte
        i = 0
        while byte:
            cell = int(byte, 16)
            # offset[i] = cell
            if i not in mins or mins[i] > cell:
                mins[i] = cell
            if i not in maxs or maxs[i] < cell:
                maxs[i] = cell
            i += 1
            byte = binascii.hexlify(file.read(1))
            byte = binascii.hexlify(file.read(1))+byte


def open_foler():
    global dataset
    path = filedialog.askdirectory()
    ir_list = glob(f"{path}/*.png", recursive=False)
    rgb_list = glob(f"{path}/*.jpg", recursive=False)
    ir_li = sorted(ir_list)
    rgb_li = sorted(rgb_list)
    dataset = pd.DataFrame(list(zip(ir_li, rgb_li)), columns=['ir', 'rgb'])
    dataset['label'] = 0
    dataset['count'] = 0

def open_csv():
    global dataset
    file = filedialog.askopenfile()
    win.title(f"{file.name}")
    dataset = pd.read_csv(file)
    # convert_to_format()

def save():
    dataset.to_csv(f"label_{i}.csv", index=False)
    messagebox.showinfo("Information", "saved succesfully")

def move(e):
    global i
    i = int(index.get())
    change(e, 0)

def next(e):
    get_count()
    change(e, 1)

def prev(e):
    get_count()
    change(e, -1)

def increase(e):
    global resized_image1
    change_count(1)
    resized_image1 = draw_txt(img1, new_label, (resized_image1.shape[0]-x1, y1))

def decrease(e):
    global resized_image1
    change_count(-1)
    resized_image1 = draw_txt(img1, new_label, (resized_image1.shape[0]-x1, y1))

def draw_circle(e):
    global img2
    x, y = get_mouse_xy(e)
    img2 = circle(resized_image2, x-1, y, size)

def remove_circle(e):
    global img2
    x, y = get_mouse_xy(e)
    img2 = uncircle(resized_image2, x, y)

def close_win(e):
    df.to_csv(save_csv_path, index=False)
    win.destroy()

################################################################ WINDOW

win = Tk()
win.title(f"labeler")
if BG == 1:  win.geometry("1924x1200")
else:  win.geometry("1604x1000")

select_frame = Frame(win, width=0, height=0, bg="white")
# select_frame.grid_propagate(False)

################################################################ TEXT

image1_text = StringVar()
image1_text.set('')
image2_text = StringVar()
image2_text.set('')
count_text = IntVar()
count_text.set(0)

label0 = Label(win, textvariable=image1_text)
label1 = Label(win, textvariable=image2_text)
LEFT_IMG = Label(win)
RIGHT_IMG = Label(win)
label2 = Label(win, text="previous: ", padx=20, pady=10)
count = Label(win, textvariable=count_text)

index = Entry(win, width=10, justify='center', borderwidth=3, bg="yellow")
previous = Entry(win, width=10, justify='center', borderwidth=3, bg="white")
# previous = Label(win, textvariable=count_text)

open_csv_button = Button(select_frame, text="select csv_file", command=open_csv)
open_folder_button = Button(select_frame, text="select folder", command=open_foler)
open_bin_button = Button(select_frame, text="select bin_file", command=open_bin)

save = Button(win, text="Save", bg='red', fg='blue', padx=10, pady=10, command=save)

################################################################ GRID

label0.grid(row=0, column=0)
label1.grid(row=0, column=1)

LEFT_IMG.grid(row=1, column=0)
RIGHT_IMG.grid(row=1, column=1)

index.grid(row=2, column=0)
count.grid(row=2, column=1, sticky='nsew')

label2.grid(row=3, column=0)
previous.grid(row=3, column=1)

select_frame.grid(row=4, column=0)
open_csv_button.grid(row=0, column=1)
open_folder_button.grid(row=0, column=2)
open_bin_button.grid(row=0, column=3)

save.grid(row=4, column=1)

################################################################ BIND

win.bind('<Return>', move)

win.bind('<Right>', next)
win.bind('<Left>', prev)

win.bind('<Up>', increase)
win.bind('<Down>', decrease)

win.bind('<Escape>', close_win)

win.bind("<Button 1>", draw_circle)
win.bind("<Button 3>", remove_circle)

################################################################ CONFIG

# Grid.columnconfigure(win, index=0, weight=1)
# Grid.columnconfigure(win, index=1, weight=1)
# Grid.rowconfigure(win, index=1, weight=1)

# grid_list = [LEFT_IMG, RIGHT_IMG]
# grd = 0
# for i in grid_list:
#     Grid.rowconfigure(win, index=grd, weight=1)
#     grd += 1

################################################################

win.mainloop()
