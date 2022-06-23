# import csv
# import os
from glob import glob
from tkinter import *
from tkinter import filedialog, messagebox
import traceback

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image, ImageTk


BG = 1

h, w = 80, 80
min, mid, max = 0, 128, 255
error = 239
cnt_thresh = 1
i, j = 0, 0
bgq = []
bgq_length = 30

win = Tk()
win.title(f"labeler")
win.geometry("1604x1000")

################################################################ FUNCTION

def histogramer(img):
    im = img.flatten()
    counts, bins = np.histogram(im, range=(0,255))
    # plot histogram centered on values 0..255
    # plt.bar(bins[:-1]-0.5, counts, width=1, edgecolor='none')
    plt.bar(bins[:-1]-0.5, counts, width=1, edgecolor='none')
    # plt.xlim([-0.5, 255.5])
    # plt.hist(im, 32, histtype='bar')
    plt.show()


def crop_img(arr):
    new_arr = arr
    X, Y = 40, 79
    a, b = 3.3, 7.9
    Radius = 8
    for x in range(80):
        for y in range(80):
            if (((X-x)**2)/a**2)+(((Y-y)**2)/b**2) > Radius**2:
                new_arr[y, x] = 0

    # lr, tb = w//4, h//4
    new_arr = new_arr[:, w//5:w-w//5]
    new_arr = new_arr[h//4:, :]

    return new_arr


def bg_maker(target):
    bg = np.zeros([80, 80], dtype=int)
    for i in bgq:
        bg += i
    bg //= len(bgq)

    # img1 = np.copy(target)
    img = target-bg

    img = crop_img(img)
    # thresh = 30
    # img[img < thresh] = 0
    # img[img > thresh] = 1
    # img = img1 * img

    # histogramer(img)
    # clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(2,2))
    # img = clahe.apply(img)

    img = np.round(img)

    img[img < 5] = 0
    img[img > 223] = 0
    # img -= img.min()
    # img = img * 255/img.max()
    img = np.round(img)
    img = img.astype(int)

    result = img.astype(np.uint8)
    bg_img = Image.fromarray(result)

    return bg_img


def show_bg(img, cnt):
    global bgq
    img_arr = np.array(img, int)

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

    if er1 < 512 or er2 < 512:
        return 1  # TRUE
    else:
        return 0  # FALSE


def show_images(image_list):
    ir, rgb, auto_cnt, cnt = image_list[0], image_list[1], image_list[2], image_list[3]

    image1 = Image.open(rgb)
    image2 = Image.open(ir)

    if BG == 1:
        bg = show_bg(image2, auto_cnt)
        image1 = bg

    image2 = np.array(image2, int)
    image2 = crop_img(image2)
    image2 = image2.astype(np.uint8)
    image2 = Image.fromarray(image2)

    image11 = image1.resize((800, 800))
    image111 = ImageTk.PhotoImage(image11)
    image22 = image2.resize((800, 800))
    image222 = ImageTk.PhotoImage(image22)
    panelA.configure(image=image111)
    panelA.image111 = image111
    panelB.configure(image=image222)
    panelB.image222 = image222

    num = find_ir_error(image1, image2)
    return num

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

# TODO 1.put empty template images in queue
# TODO 2.cosine distance to find the closest template
# TODO 3.use 2 to subtract from now data
