# import csv
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


BG = 2
SAVE = 0
SHOW = 1
AUTO_RUN = 0

df = pd.DataFrame()
base_dir = f"../model/v1.2/out"
save_csv_path = f"{base_dir}/output.csv"

x1,y1,x2,y2 = 120, 70, 0, 70
h, w = 80, 80
if BG == 1:  size = 12
else:  size = 10
re_size1, re_size2 = (w*size, h*size), (w*size, h*size)
min, mid, max = 0, 128, 255
error = 239
cnt_thresh = 1
i, j = 0, 0
bgq = []
bgq_length = 30

win = Tk()
win.title(f"labeler")
if BG == 1:  win.geometry("1924x1200")
else:  win.geometry("1604x1000")

################################################################ FUNCTION

def drop_err(csv):
    img_list = glob(f"{base_dir}/*.png")
    df = pd.read_csv(save_csv_path)
    for i in range(len(df)):
        if f"out/{df.iloc[i, 0]}" not in img_list:
            df.iloc[i] = np.nan
    df = df.dropna()
    df.iloc[:, 1] = df.iloc[:, 1].astype(int)
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


def crop_img(arr, W, H, X, Y, a, b, R):
    new_arr = arr
    for x in range(80):
        for y in range(80):
            if (((X-x)**2)/a**2)+(((Y-y)**2)/b**2) > R**2:
                new_arr[y, x] = 0

    new_arr = new_arr[:, W//5:W-W//5]
    new_arr = new_arr[H//4:, :]

    return new_arr


def bg_maker(target):
    bg = np.zeros([80, 80], dtype=int)
    for i in bgq:
        bg += i
    bg //= len(bgq)

    # img1 = np.copy(target)
    img = target-bg

    low = 16
    img -= low
    img[img < low] = 0

    high = 255
    img *= high//img.max()
    img[img > high] = high
    # img = img1 * img
    # img[abs(img) < thresh] = img.min()
    # img[img < 5] = 0
    # img[img > 223] = 0
    # img -= img.min()
    # histogramer(img)

    # bg_img = cv2.cvtColor(bg_img)
    # clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(2, 2))
    # bg_img = clahe.apply(bg_img)

    return img


def show_bg(img, cnt):
    global bgq
    img_arr = img

    error1 = len(img_arr[img_arr > 239])
    error2 = len(img_arr[img_arr < 1])
    if error1 > 512 or error2 > 256:
        print(f"{i}: white-{error1}, black-{error2}")
        return 0

    bgq.insert(0, img_arr)
    if len(bgq) > bgq_length:  bgq.pop(-1)
    else:  pass

    bg = bg_maker(img_arr)

    return bg


def convert_to_format():
    global images
    images['count'] = images['count'].fillna(0)
    df = pd.read_csv(file)
    df1 = df.replace({'.png': '.jpg'}, regex=True)
    df1 = df1.rename(columns={'img_name': 'rgb_name'})
    df = df.drop(df.columns[1], axis=1)
    images = pd.concat([df, df1], axis=1)
    images["count"] = ""


def find_ir_error(img1, img2):
    arr1 = np.array(img1, int)
    arr2 = np.array(img2, int)

    e1 = arr1[arr1 > 239]
    e2 = arr2[arr2 > 239]
    er1, er2 = len(e1), len(e2)

    if er1 < 512 or er2 < 512:  return 1  # TRUE
    else:  return 0  # FALSE


def show_images(image_list):
    global base_name, df
    ir, rgb, auto_cnt, cnt = image_list[0], image_list[1], image_list[2], image_list[3]

    image1 = Image.open(rgb)
    image2 = Image.open(ir)
    image1 = np.array(image1, np.uint8)
    image2 = np.array(image2, np.uint8)

    # image1 = cv2.imread(rgb, 1)
    # image2 = cv2.imread(ir, 0)
    # image1 = image1[:, :, ::-1]

    num = find_ir_error(image1, image2)
    if num == 1:
        if BG == 1:
            bg = show_bg(image2, auto_cnt)
            image1 = bg
        elif BG == 2:
            bg = show_bg(image2, auto_cnt)
            image2 = bg

        if SAVE == 1:
            base_name = os.path.basename(ir)
            save_img_path = f"{base_dir}/{base_name}"
            # new_row = {'img': base_name, 'cnt': auto_cnt}
            # df = df.append(new_row, ignore_index=True)
            new_row = pd.DataFrame({'img':[base_name], 'cnt':[auto_cnt]})
            df = pd.concat([df, new_row], axis=0, ignore_index=True, sort=False)
            bg.save(save_img_path)

        img1_H, img1_W = image1.shape[0], image1.shape[1]
        img2_H, img2_W = image2.shape[0], image2.shape[1]
        image1 = crop_img(image1, img1_H, img1_W, img1_H//2, img1_W, 16, 38, 40)
        image2 = crop_img(image2, img2_H, img2_W, img2_H//2, img2_W, 3.3, 7.7, 8)

        interpolation = cv2.INTER_NEAREST  ## CUBIC LINEAR AREA
        image11 = cv2.resize(image1, (re_size1), interpolation=interpolation)
        image22 = cv2.resize(image2, (re_size2), interpolation=interpolation)

        image11 = Image.fromarray((image11).astype(np.uint8))
        image22 = Image.fromarray((image22).astype(np.uint8))

        if SHOW == 1:
            image111 = ImageTk.PhotoImage(image11)
            image222 = ImageTk.PhotoImage(image22)
            panelA.configure(image=image111)
            panelA.image111 = image111
            panelB.configure(image=image222)
            panelB.image222 = image222
        return num
    else: return num

################################################################ EVENT

def open_foler():
    global images
    path = filedialog.askdirectory()

    ir_list = glob(f"{path}/*.png", recursive=False)
    rgb_list = glob(f"{path}/*.jpg", recursive=False)
    ir_li = sorted(ir_list)
    rgb_li = sorted(rgb_list)
    images = pd.DataFrame(list(zip(ir_li, rgb_li)), columns=['ir', 'rgb'])
    images['cnt'] = 0
    images['count'] = 0


def open_csv():
    global images
    file = filedialog.askopenfile()

    win.title(f"{file.name}")
    images = pd.read_csv(file)
    # convert_to_format()


def change(e, idx):
    global i
    clear()
    i += idx

    row = images.iloc[i]
    ir, rgb, auto_cnt, cnt = images.iloc[i,0], images.iloc[i,1], images.iloc[i,2], images.iloc[i,3]

    image1_text.set(f"{rgb}")
    image2_text.set(f"{ir}")
    try:
        if show_images(row):
            index.insert(0, i)
        else:
            cnt = -1
            index.insert(0, f"{i}: ERROR!")
    except FileNotFoundError as FE:
        print(f"{FE}: {i}")
        traceback.print_exc()
        cnt = -2
    except IndexError as IE:
        print(f"{IE}: {i}, FINISHED")
        traceback.print_exc()
        i -= idx
        # save()
    except TypeError as TE:
        print(f"{TE}: {i}")
        traceback.print_exc()
    count_text.set(f"{cnt}")
    previous.insert(0, auto_cnt)

    if AUTO_RUN == 1:  next(e)


def save():
    images.to_csv(f"label_{i}.csv", index=False)
    messagebox.showinfo("Information", "saved succesfully")


def clear():
    count_text.set(0)
    index.delete(0, END)
    previous.delete(0, END)


def set(e, idx=0):
    global i
    i = int(index.get())
    change(e, idx)


def increase(e):
    count_text.set(count_text.get()+1)


def decrease(e):
    count_text.set(count_text.get()-1)


def get_count():
    cnt = count_text.get()
    images.iloc[i, 3] = cnt
    images['count'] = images['count'].astype(int)

def next(e, idx=1):
    get_count()
    change(e, idx)


def prev(e, idx=-1):
    get_count()
    change(e, idx)


def close_win(e):
    df.to_csv(save_csv_path, index=False)
    win.destroy()

################################################################ TEXT

image1_text = StringVar()
image1_text.set('')
image2_text = StringVar()
image2_text.set('')
count_text = IntVar()
count_text.set(0)

label0 = Label(win, textvariable=image1_text)
label1 = Label(win, textvariable=image2_text)
panelA = Label(win)
panelB = Label(win)
label2 = Label(win, text="previous: ", padx=20, pady=10)
count = Label(win, textvariable=count_text)

index = Entry(win, width=10, justify='center', borderwidth=3, bg="yellow")
previous = Entry(win, width=10, justify='center', borderwidth=3, bg="white")

open_csv_button = Button(win, text="select csv_file", command=open_csv)
open_folder_button = Button(win, text="select folder", command=open_foler)
save = Button(win, text="Save", bg='red', fg='blue', padx=10, pady=10, command=save)

################################################################ GRID

label0.grid(row=0, column=0, rowspan=1, columnspan=1)
label1.grid(row=0, column=1, rowspan=1, columnspan=1)

panelA.grid(row=1, column=0, rowspan=1, columnspan=1)
panelB.grid(row=1, column=1, rowspan=1, columnspan=1)

index.grid(row=2, column=0, rowspan=1, columnspan=1)
label2.grid(row=2, column=1, rowspan=1, columnspan=1)

count.grid(row=3, column=0, rowspan=1, columnspan=1)
previous.grid(row=3, column=1, rowspan=1, columnspan=1)

open_csv_button.grid(row=4, column=0, rowspan=1, columnspan=1)
save.grid(row=4, column=1, rowspan=1, columnspan=1)

open_folder_button.grid(row=5, column=0, rowspan=1, columnspan=1)

################################################################ BIND

win.bind('<Return>', set)

win.bind('<Right>', next)
win.bind('<Left>', prev)

win.bind('<Up>', increase)
win.bind('<Down>', decrease)

win.bind('<Escape>', close_win)

################################################################

win.mainloop()
