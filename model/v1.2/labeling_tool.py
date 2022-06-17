import csv
from glob import glob
# import os
from pprint import pprint
from tkinter import *
from tkinter import filedialog, messagebox
import pyautogui

# import cv2
# import numpy as np
import numpy as np
import pandas as pd
from PIL import Image, ImageTk


csv_output_name = f"output.csv"

win = Tk()
win.title("ppcnt_labeler")
win.geometry("1604x1000")
main_lst = []

prev = ""
panelA, panelB = None, None
i, j = 0, 0
step = 1

h, w = 80, 80
min, mid, max = 0, 128, 255
error = 239


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

################################################################ FUNCTION

def select_folder():
    global images
    path = filedialog.askdirectory()

    ir_list = glob(f"{path}/*.png", recursive=False)
    rgb_list = glob(f"{path}/*.jpg", recursive=False)
    images = ir_list + rgb_list
    images = sorted(images)


def select_label():
    global images
    file = filedialog.askopenfile()

    df = pd.read_csv(file)
    df1 = df.replace({'.png': '.jpg'}, regex=True)
    df1 = df1.rename(columns={'img_name': 'rgb_name'})
    df = df.drop(df.columns[1], axis=1)
    # result = pd.concat([df, df1], axis=1)
    # images = result.values.tolist()
    images = pd.concat([df, df1], axis=1)
    images["count"] = ""


def show_images(image_list):
    global panelA, panelB

    image1 = Image.open(image_list[1])
    print(image1)
    image1_arr = np.array(image1, int)
    image1 = image1.resize((800, 800))
    image1 = ImageTk.PhotoImage(image1)

    image2 = Image.open(image_list[0])
    image2_arr = np.array(image2, int)
    image2 = image2.resize((800, 800))
    image2 = ImageTk.PhotoImage(image2)

    e1 = image1_arr[image1_arr > 239]
    e2 = image2_arr[image2_arr > 239]
    er1, er2 = len(e1), len(e2)

    if er1 < 512 or er2 < 512:
        # result = img1_minus_img2(im1_arr, im2_arr)

        if panelA is None or panelB is None:
            panelA = Label(image=image1)
            panelA.image1 = image1
            panelA.grid(row=1, column=0, rowspan=1, columnspan=1)

            panelB = Label(image=image2)
            panelB.image2 = image2
            panelB.grid(row=1, column=1, rowspan=1, columnspan=1)
        else:
            panelA.configure(image=image1)
            panelA.image1 = image1

            panelB.configure(image=image2)
            panelB.image2 = image2

    else:
        pyautogui.press("enter")


def print_label_on_entry():
    pass


# def show_list():
    # print(len(main_lst))
    # print(main_lst)


################################################################ EVENT

def save():
    # images = images.values.tolist()
    with open(csv_output_name, "w") as f:
        Writer = csv.writer(f)
        Writer.writerow(["ir", "rgb", "previous", "count"])
        Writer.writerows(images.values.tolist())
        messagebox.showinfo("Information", "saved succesfully")
        clear()


def clear():
    count.delete(0, END)
    previous.delete(0, END)


def close_win(e):
    win.destroy()


def prev(e):
    global i
    clear()
    i -= step
    print(i)
    try:
        # show_images(images[i])
        # image1_text.set(f"{images[i][1]}")
        # image2_text.set(f"{images[i][0]}")
        # count.insert(0, images[i][3])
        # previous.insert(0, images[i][2])
        show_images(images.iloc[i])
        image1_text.set(f"{images.iloc[i,1]}")
        image2_text.set(f"{images.iloc[i,0]}")
        count.insert(0, images.iloc[i,3])
        previous.insert(0, images.iloc[i,2])
    except IndexError:
        print(f"{i}, beginning")
        i += step


def next(e):
    global i
    clear()
    i += step
    images.iloc[i,3] = count.get()
    # images.insert([i][3], count.get())
    print(f"{i} {images.iloc[i]}")
    # print(f"{i} {images[i]}")
    try:
        # show_images(images[i])
        # image1_text.set(f"{images[i][1]}")
        # image2_text.set(f"{images[i][0]}")
        # count.insert(0, images[i][3])
        # previous.insert(0, images[i][2])
        show_images(images.iloc[i])
        image1_text.set(f"{images.iloc[i,1]}")
        image2_text.set(f"{images.iloc[i,0]}")
        count.insert(0, images.iloc[i,3])
        previous.insert(0, images.iloc[i,2])
    except IndexError:
        print(f"{i}, finished")
        i -= step

################################################################ SET

listbox = Listbox(win)

image1_text = StringVar()
image1_text.set("-")

image2_text = StringVar()
image2_text.set("-")

index_text = StringVar()
index_text.set(i)

################################################################ LABEL

label0 = Label(win, textvariable=image1_text)
label01 = Label(win, textvariable=image2_text)
label02 = Label(win, textvariable=index_text)

label1 = Label(win, text="count: ", padx=20, pady=10)
label2 = Label(win, text="previous: ", padx=20, pady=10)

################################################################ ENTRY

count = Entry(win, width=30, borderwidth=3, bg="white")
previous = Entry(win, width=30, borderwidth=3, bg="white")

################################################################ BUUTON

select_folder_button = Button(win, text="select folder", command=select_folder)
select_label_button = Button(win, text="select csv_file", command=select_label)

save = Button(win, text="Save", bg='red', fg='blue', padx=10, pady=10, command=save)

# ################################################################ GRID
select_folder_button.grid(row=5, column=0, rowspan=1, columnspan=1)
select_label_button.grid(row=6, column=0, rowspan=1, columnspan=1)

label0.grid(row=0, column=0, rowspan=1, columnspan=1)
label01.grid(row=0, column=1, rowspan=1, columnspan=1)

label1.grid(row=3, column=0, rowspan=1, columnspan=1)
label2.grid(row=3, column=1, rowspan=1, columnspan=1)

count.grid(row=4, column=0, rowspan=1, columnspan=1)
previous.grid(row=4, column=1, rowspan=1, columnspan=1)

label02.grid(row=5, column=1, rowspan=1, columnspan=1)
save.grid(row=6, column=1, rowspan=1, columnspan=1)

listbox.grid(row=0, column=2, sticky=NS, rowspan=6, columnspan=1)

# cnt.grid(row=6, column=1, rowspan=1, columnspan=1)

################################################################ BIND

# count.bind("<FocusIn>", clear)
# previous.bind("<FocusIn>", clear)

win.bind('<Return>', next)
win.bind('<BackSpace>', prev)

win.bind('<Right>', next)
win.bind('<Left>', prev)

win.bind('<Escape>', close_win)

################################################################

win.mainloop()
