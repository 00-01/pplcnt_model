# import csv
from glob import glob
# import os
from tkinter import *
from tkinter import filedialog, messagebox

# import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageTk


win = Tk()
win.title(f"labeler")
win.geometry("1604x1000")

i, j = 0, 0
h, w = 80, 80
min, mid, max = 0, 128, 255
error = 239

################################################################ FUNCTION

def crop_img(arr):
    new_arr = arr
    lr, tb = w//4, h//4
    croped_arr = new_arr[:, w//4:w-w//4]
    croped_arr = croped_arr[h//4:, :]

    return croped_arr


def img1_minus_img2(img1, img2):
    # img3 = np.round_(255-((img1_arr-img2_arr)+255)/2).astype(np.uint8)
    img_dist = img1-img2

    img_dist = crop_img(img_dist)

    img_norm = np.round_(255-(img_dist+255))  # inverse
    img_norm[img_norm < 15] = 0  # low cut

    result = img_norm.astype(np.uint8)  # dtype to uint8

    return result


# def minus_mask_finder(img1_cnt, img2_cnt):
#     if img1_cnt == 0:
#         prev = img1_path
        # prev_cnt = img1_cnt


def convert_to_format():
    global images
    images['count'] = images['count'].fillna(0)
    df = pd.read_csv(file)
    df1 = df.replace({'.png': '.jpg'}, regex=True)
    df1 = df1.rename(columns={'img_name': 'rgb_name'})
    df = df.drop(df.columns[1], axis=1)
    images = pd.concat([df, df1], axis=1)
    images["count"] = ""


def find_ir_error(arr1, arr2):
    e1 = arr1[arr1 > 239]
    e2 = arr2[arr2 > 239]
    er1, er2 = len(e1), len(e2)

    if er1 < 512 or er2 < 512:
        return 1  # TRUE
    else:
        return 0  # FALSE


def show_images(image_list):
    # global i#, image1, image2
    image1 = Image.open(image_list[1])
    image1_arr = np.array(image1, int)
    image1 = image1.resize((800, 800))
    image1 = ImageTk.PhotoImage(image1)

    image2 = Image.open(image_list[0])
    image2_arr = np.array(image2, int)
    image2 = image2.resize((800, 800))
    image2 = ImageTk.PhotoImage(image2)

    panelA.configure(image=image1)
    panelA.image1 = image1
    panelB.configure(image=image2)
    panelB.image2 = image2

    num = find_ir_error(image1_arr, image2_arr)
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
    image1_text.set(f"{images.iloc[i,1]}")
    image2_text.set(f"{images.iloc[i,0]}")
    try:
        if show_images(images.iloc[i]):
            index.insert(0, i)
        else:
            images.iloc[i,3] = -1
            index.insert(0, f"{i}: ERROR!")
    except FileNotFoundError as E0:
        print(E0)
        images.iloc[i, 3] = -2
    except IndexError as E1:
        print(E1)
        print(f"{i}, finished")
        i -= idx
        save()
    count_text.set(f"{images.iloc[i, 3]}")
    previous.insert(0, images.iloc[i, 2])


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
label2 = Label(win, text="previous: ", padx=20, pady=10)
panelA = Label(win)
panelB = Label(win)

index = Entry(win, width=10, borderwidth=3, bg="yellow")
previous = Entry(win, width=10, borderwidth=3, bg="white")

open_folder_button = Button(win, text="select folder", command=open_foler)
open_csv_button = Button(win, text="select csv_file", command=open_csv)
save = Button(win, text="Save", bg='red', fg='blue', padx=10, pady=10, command=save)

count = Label(win, textvariable=count_text)

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

# win.bind('<BackSpace>', prev)
win.bind('<Return>', set)

win.bind('<Right>', next)
win.bind('<Left>', prev)

win.bind('<Up>', increase)
win.bind('<Down>', decrease)

win.bind('<Escape>', close_win)

################################################################

win.mainloop()
