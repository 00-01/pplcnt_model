import os
import shutil
import tkinter
from tkinter import *

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import six


tmpDirPath = 'tmpData'

canvasSize = (2, 3)

if os.path.exists(tmpDirPath):
    shutil.rmtree(tmpDirPath)
os.mkdir(tmpDirPath)

global CWD, df, menu_click, canvas, label_ir_arr, label_ir, dataIndex, preClickIndex, viewFileName
CWD = os.getcwd().replace('\\', '/')+'/'
df = None
preClickIndex = None
viewFileName = None
dataIndex = -1


def searchDirecotry(path):
    dir = []
    file = []
    for f in os.listdir(path):
        if os.path.isdir(path+'/'+f):
            dir.append(f)
        else:
            file.append(f)
    dir.sort()
    file.sort()
    return {'path': path, 'dir': dir, 'file': file}


window = tkinter.Tk()
window.title("IRBOX_data SHOW")
window.geometry("1000x800+100+100")

txt_cwd = Label(window, text=CWD)
txt_cwd.pack()

menu_click = 0

menu = Frame(window)
menu.pack(fill=X)


def click_menu(click):
    global menu_click
    if click == 0:
        menu_click = 0
        menu_btn_show.config(bg='yellow')
        menu_btn_label.config(bg='gray')
        listbox_click(None)
    elif click == 1:
        menu_click = 1
        menu_btn_show.config(bg='gray')
        menu_btn_label.config(bg='yellow')
        listbox_click(None)


menu_btn_show = Button(menu, text='show', command=lambda: click_menu(0), bg='yellow')
menu_btn_show.pack(side=RIGHT)
menu_btn_label = Button(menu, text='label', command=lambda: click_menu(1), bg='gray')
menu_btn_label.pack(side=RIGHT)

frame = Frame(window)
frame.pack(fill=BOTH, expand=True)

# photo = tkinter.PhotoImage(file="../CookPython/GIF/pic1.gif")
# pLabel = tkinter.Label(window, image=photo)
# pLabel.pack(expand=1, anchor=CENTER)

lb_dir = Listbox(frame, selectmode='single', width=40)
lb_dir.yview()


def render_mpl_table(data, col_width=3.0, row_height=0.625, font_size=14, header_color='#40466e', row_colors=['#f1f1f2', 'w'],
                     edge_color='w', bbox=[0, 0, 1, 1], header_columns=0, ax=None, **kwargs):
    if ax is None:
        size = (np.array(data.shape[::-1])+np.array([0, 1]))*np.array([col_width, row_height])
        fig, ax = plt.subplots(figsize=size)
        ax.axis('off')

    mpl_table = ax.table(cellText=data.values, bbox=bbox, colLabels=data.columns, **kwargs)

    mpl_table.auto_set_font_size(False)
    mpl_table.set_fontsize(font_size)

    for k, cell in six.iteritems(mpl_table._cells):
        cell.set_edgecolor(edge_color)
        if k[0] == 0 or k[1] < header_columns:
            cell.set_text_props(weight='bold', color='w')
            cell.set_facecolor(header_color)
        else:
            cell.set_facecolor(row_colors[k[0]%len(row_colors)])
    return ax


def view():
    global CWD, df, menu_click, canvas, label_ir_arr, label_ir, entry_ir, dataIndex, preClickIndex, viewFileName
    # plt.clf()
    # plt.plot(range(len([28.05, 27.7, 28.34, 29, 29.54, 30.71, 31.55, 32.55, 33.18])),
    #          [28.05, 27.7, 28.34, 29, 29.54, 30.71, 31.55, 32.55, 33.18])
    # plt.plot(range(len([25.4, 26.46, 27.04, 27.53, 28, 28.94, 29.7, 30.7, 31.31])),
    #          [25.4, 26.46, 27.04, 27.53, 28, 28.94, 29.7, 30.7, 31.31])
    # plt.plot(range(len([24.54, 25.35, 26.1, 26.91, 27.32, 28.12, 28.96, 29.59, 30.68])),
    #          [24.54, 25.35, 26.1, 26.91, 27.32, 28.12, 28.96, 29.59, 30.68])
    filePath = CWD+viewFileName
    df = pd.read_csv(filePath)
    canvasMaxSize = min(frame1.winfo_width()//2, frame1.winfo_height()//2)

    for canv1 in canvas:
        for canv2 in canv1:
            canv2.destroy()
    canvas = []
    for row in range(canvasSize[0]):
        canvasrow = []
        for col in range(canvasSize[1]):
            canvasTmp = Frame(frame1)
            canvasTmp.grid(row=row, column=col)
            canvasrow.append(canvasTmp)
        canvas.append(canvasrow)
    if menu_click == 0:
        # 선택가능하도록 entry추가
        if dataIndex == -1:
            dataIndex = np.random.randint(len(df))
        elif dataIndex >= len(df):
            dataIndex = len(df)-1
        entry_dataInfo.delete(0, 'end')
        entry_dataInfo.insert(0, str(dataIndex))
        txt_dataInfo.config(text='전체 건수 : '+str(len(df))+'('+str(dataIndex)+')')

        if 'Distance' in df.columns:
            img = df.iloc[dataIndex, -64:].values
            data_distanceR_mean = round(df.iloc[:, 1].mean(), 1)
            data_distanceR = df.iloc[dataIndex, 1]
            data_distanceL_mean = round(df.iloc[:, 1].mean(), 1)
            data_distanceL = df.iloc[dataIndex, 1]
        else:
            img = df.iloc[dataIndex, -64:].values
            data_distanceR_mean = round(df.iloc[:, 1].mean(), 1)
            data_distanceR = df.iloc[dataIndex, 1]
            data_distanceL_mean = round(df.iloc[:, 2].mean(), 1)
            data_distanceL = df.iloc[dataIndex, 2]
        imgData = cv2.flip(img.reshape(8, 8), 0)

        # canvas[0][0]
        plt.clf()
        dftmp = pd.DataFrame()
        dftmp[' '] = ['csvAVG', 'IRBOX', 'REF', 'DIF(IR-REF)']
        data_temp_mean = round(df.iloc[:, 2].mean(), 1)
        data_temp = df.iloc[dataIndex, 2]
        try:
            ref_distance = float(viewFileName.split('_')[0][1:])
        except:
            ref_distance = 0
        try:
            ref_temp = float(viewFileName.split('_')[2][1:])
        except:
            ref_temp = 0
        if 'background' in viewFileName:
            dif_distanceR_meandata = 0
            dif_distanceL_meandata = 0
            dif_temp_meandata = 0
        else:
            dif_distanceR_meandata = round(data_distanceR-ref_distance, 1)
            dif_distanceL_meandata = round(data_distanceL-ref_distance, 1)
            dif_temp_meandata = round(data_temp-ref_temp, 1)
        dftmp['DistanceR'] = [data_distanceR_mean, data_distanceR, ref_distance, dif_distanceR_meandata]
        dftmp['DistanceL'] = [data_distanceL_mean, data_distanceL, ref_distance, dif_distanceL_meandata]
        dftmp['AirTemp'] = [data_temp_mean, data_temp, ref_temp, dif_temp_meandata]
        render_mpl_table(dftmp, header_columns=0, col_width=2.0)
        plt.savefig(tmpDirPath+'/'+'tmpIRDataTable.png')
        cv2.imwrite(tmpDirPath+'/'+'tmpIRDataTable.png',
                    cv2.resize(cv2.imread(tmpDirPath+'/'+'tmpIRDataTable.png'), (canvasMaxSize, canvasMaxSize)))
        dataTableimg = PhotoImage(file=tmpDirPath+'/'+'tmpIRDataTable.png')
        img0 = Label(canvas[0][0])
        img0.configure(image=dataTableimg)
        img0.image = dataTableimg
        img0.pack(fill=BOTH, expand=True)
        plt.close()

        # canvas[0][1]
        plt.clf()
        sns.heatmap(imgData, fmt='.2f', vmin=0, vmax=40, annot=True)

        # 물체 온도
        try:
            plt.title('ObjTemp : '+str(float(viewFileName.split('_')[1][1:])))
        except:
            pass

        plt.savefig(tmpDirPath+'/'+'tmpIRData.png')
        cv2.imwrite(tmpDirPath+'/'+'tmpIRData.png',
                    cv2.resize(cv2.imread(tmpDirPath+'/'+'tmpIRData.png'), (canvasMaxSize, canvasMaxSize)))
        irimg = PhotoImage(file=tmpDirPath+'/'+'tmpIRData.png')
        img1 = Label(canvas[0][1])
        img1.configure(image=irimg)
        img1.image = irimg
        img1.pack(fill=BOTH, expand=True)
        plt.close()

        # canvas[1][0]
        plt.clf()
        irhist = plt.hist(df.iloc[:, 4:].values.reshape(-1), bins=100)
        histrange = range(int(np.min(irhist[1]))-1, int(np.max(irhist[1]))+2, 1)
        plt.xticks(histrange)
        if 'background' in viewFileName:
            plt.title('Avg IR 8x8 : '+str(round(np.mean(df.iloc[:, 4:].values.reshape(-1)), 2)))
        plt.savefig(tmpDirPath+'/'+'tmpIRavghistData.png')
        cv2.imwrite(tmpDirPath+'/'+'tmpIRavghistData.png',
                    cv2.resize(cv2.imread(tmpDirPath+'/'+'tmpIRavghistData.png'),
                               (canvasMaxSize, canvasMaxSize)))
        iravghistimg = PhotoImage(file=tmpDirPath+'/'+'tmpIRavghistData.png')
        img2 = Label(canvas[1][0])
        img2.configure(image=iravghistimg)
        img2.image = iravghistimg
        img2.pack(fill=BOTH, expand=True)
        plt.close()

        # canvas[1][1]
        plt.clf()
        plt.hist(df.iloc[dataIndex, 4:].values.reshape(-1), bins=100)
        plt.xticks(histrange)
        plt.savefig(tmpDirPath+'/'+'tmpIRhistData.png')
        cv2.imwrite(tmpDirPath+'/'+'tmpIRhistData.png',
                    cv2.resize(cv2.imread(tmpDirPath+'/'+'tmpIRhistData.png'), (canvasMaxSize, canvasMaxSize)))
        irhistimg = PhotoImage(file=tmpDirPath+'/'+'tmpIRhistData.png')
        img3 = Label(canvas[1][1])
        img3.configure(image=irhistimg)
        img3.image = irhistimg
        img3.pack(fill=BOTH, expand=True)
        plt.close()

    elif menu_click == 1:
        if 'center_desk' not in viewFileName:
            return

        backgroundFile = None
        for ls_file in os.listdir(CWD):
            if '_'.join(viewFileName.split('_')[:1]) in ls_file:
                if 'background' in ls_file:
                    backgroundFile = ls_file
                    break
        if not backgroundFile:
            txt_dataInfo.config(text='targetFile : '+viewFileName+'||| backgroundFile : NULL')
        txt_dataInfo.config(text='targetFile : '+viewFileName+'||| backgroundFile : '+backgroundFile)
        backgroundPath = CWD+backgroundFile
        df = pd.read_csv(backgroundPath)
        if 'Distance' in df.columns:
            background_Mean_Pixel = df.iloc[:, 4:].mean().values
        else:
            background_Mean_Pixel = df.iloc[:, 5:].mean().values

        df = pd.read_csv(filePath)
        if 'Distance' in df.columns:
            target_Mean_Pixel = df.iloc[:, 4:].mean().values
        else:
            target_Mean_Pixel = df.iloc[:, 5:].mean().values

        Minus = target_Mean_Pixel-background_Mean_Pixel

        # canvas[0][0]
        plt.clf()
        sns.heatmap(cv2.flip(background_Mean_Pixel.reshape(8, 8), 0), fmt='.2f', vmin=0, vmax=40, annot=True)
        plt.title('Background')
        plt.savefig(tmpDirPath+'/'+'tmpLabelBackground.png')
        cv2.imwrite(tmpDirPath+'/'+'tmpLabelBackground.png',
                    cv2.resize(cv2.imread(tmpDirPath+'/'+'tmpLabelBackground.png'), (canvasMaxSize, canvasMaxSize)))
        labelbgimg = PhotoImage(file=tmpDirPath+'/'+'tmpLabelBackground.png')
        img0 = Label(canvas[0][0])
        img0.configure(image=labelbgimg)
        img0.image = labelbgimg
        img0.pack(fill=BOTH, expand=True)
        plt.close()

        # canvas[0][1]
        plt.clf()
        sns.heatmap(cv2.flip(target_Mean_Pixel.reshape(8, 8), 0), fmt='.2f', vmin=0, vmax=40, annot=True)
        plt.title('Target')
        plt.savefig(tmpDirPath+'/'+'tmpLabelTarget.png')
        cv2.imwrite(tmpDirPath+'/'+'tmpLabelTarget.png',
                    cv2.resize(cv2.imread(tmpDirPath+'/'+'tmpLabelTarget.png'), (canvasMaxSize, canvasMaxSize)))
        labeltargetimg = PhotoImage(file=tmpDirPath+'/'+'tmpLabelTarget.png')
        img1 = Label(canvas[0][1])
        img1.configure(image=labeltargetimg)
        img1.image = labeltargetimg
        img1.pack(fill=BOTH, expand=True)
        plt.close()

        # canvas[1][0]
        plt.clf()
        sns.heatmap(cv2.flip(Minus.reshape(8, 8), 0), fmt='.2f', vmin=-10, vmax=10, annot=True)
        plt.title('Minus')
        plt.savefig(tmpDirPath+'/'+'tmpLabelMinus.png')
        cv2.imwrite(tmpDirPath+'/'+'tmpLabelMinus.png',
                    cv2.resize(cv2.imread(tmpDirPath+'/'+'tmpLabelMinus.png'), (canvasMaxSize, canvasMaxSize)))
        labelminusimg = PhotoImage(file=tmpDirPath+'/'+'tmpLabelMinus.png')
        img2 = Label(canvas[1][0])
        img2.configure(image=labelminusimg)
        img2.image = labelminusimg
        img2.pack(fill=BOTH, expand=True)
        plt.close()

        # canvas[1][1]
        tmp_frame = Frame(canvas[1][1])
        tmp_frame.pack(fill=X, expand=True)

        label_ir = []
        label_ir_arr = np.zeros((8, 8))

        def click_label_reset(event):
            global label_ir_arr, label_ir
            for r in range(8):
                for c in range(8):
                    label_ir_arr[r, c] = 0
                    label_ir[r*8+c].config(bg='gray', text='%.2f'%(cv2.flip(Minus.reshape(8, 8), 0)[r, c]))

        def click_label_apply(event):
            global label_ir_arr, entry_ir
            for r in range(8):
                for c in range(8):
                    entry_ir[r*8+c].delete(0, 'end')
                    if label_ir_arr[r, c] == 0:
                        entry_ir[r*8+c].insert(0, '0')
                    else:
                        entry_ir[r*8+c].insert(0, '%.2f'%(cv2.flip(target_Mean_Pixel.reshape(8, 8), 0)[r, c]))

        btn_reset = Button(tmp_frame, text='reset')
        btn_reset.bind('<Button-1>', click_label_reset)
        btn_reset.pack(side=RIGHT)
        btn_apply = Button(tmp_frame, text='apply')
        btn_apply.bind('<Button-1>', click_label_apply)
        btn_apply.pack(side=RIGHT)
        tmp_frame = Frame(canvas[1][1])
        tmp_frame.pack(fill=BOTH, expand=True)

        def click_label_ir(event, key, r, c):
            global label_ir_arr, label_ir
            if label_ir_arr[r, c] == 0:
                if key == 1:
                    label_ir[r*8+c].config(bg='yellow')
                elif key == 2:
                    label_ir[r*8+c].config(bg='red')
                label_ir_arr[r, c] = key
            elif label_ir_arr[r, c] == 1:
                if key == 1:
                    label_ir[r*8+c].config(bg='gray')
                    label_ir_arr[r, c] = 0
                elif key == 2:
                    label_ir[r*8+c].config(bg='red')
                    label_ir_arr[r, c] = 2
            elif label_ir_arr[r, c] == 2:
                if key == 1:
                    label_ir[r*8+c].config(bg='yellow')
                    label_ir_arr[r, c] = 1
                elif key == 2:
                    label_ir[r*8+c].config(bg='gray')
                    label_ir_arr[r, c] = 0

        for r in range(8):
            for c in range(8):
                canvasTmp = Button(tmp_frame, text='%.2f'%(cv2.flip(Minus.reshape(8, 8), 0)[r, c]), width=3, height=2, bg='gray')
                canvasTmp.bind('<Button-1>', lambda event, key=1, rr=r, cc=c: click_label_ir(event, key, rr, cc))
                canvasTmp.bind('<Button-3>', lambda event, key=2, rr=r, cc=c: click_label_ir(event, key, rr, cc))
                canvasTmp.grid(row=r, column=c)
                label_ir.append(canvasTmp)

        # canvas[1][2]
        tmp_frame1 = Frame(canvas[1][2])
        tmp_frame1.pack(fill=X, expand=True)

        entry_ir = []

        def click_entry_reset(event):
            global label_ir_arr, entry_ir
            for r in range(8):
                for c in range(8):
                    entry_ir[r*8+c].delete(0, 'end')
                    entry_ir[r*8+c].insert(0, '%.2f'%(cv2.flip(target_Mean_Pixel.reshape(8, 8), 0)[r, c]))

        def click_label_save(event):
            global label_ir_arr, label_ir, viewFileName
            '''
            1차원 데이터[71]
            [3]실측거리(ref), 실측물체온도(ref), 실측대기온도(ref)
            [4]측정거리R(ir), 측정거리L(ir), 측정대기온도(ir), 측정습도(ir)
            [64]values[1~64]
            '''
            values = []
            for r in range(8):
                for c in range(8):
                    values.append(entry_ir[r*8+c].get())
            values = np.array(values)

            ref_distance = float(viewFileName.split('_')[0][1:])
            ref_object = float(viewFileName.split('_')[1][1:])
            ref_temp = float(viewFileName.split('_')[2][1:])
            if 'Distance' in df.columns:
                csvValues = np.concatenate([df.iloc[:, 1:2].mean().values, df.iloc[:, 1:4].mean().values])
            else:
                csvValues = df.iloc[:, 1:5].mean().values
            saveData = np.concatenate([[ref_distance], [ref_object], [ref_temp], csvValues, values])

            labelPath = 'label'
            filePath = CWD+'../'+labelPath
            if not os.path.exists(filePath):
                os.mkdir(filePath)
            npyFilePath = filePath+'/'+viewFileName.replace('.csv', '.npy')

            np.save(npyFilePath, saveData.astype(np.float64))

        btn_reset1 = Button(tmp_frame1, text='reset')
        btn_reset1.bind('<Button-1>', click_entry_reset)
        btn_reset1.pack(side=RIGHT)
        btn_save = Button(tmp_frame1, text='save')
        btn_save.bind('<Button-1>', click_label_save)
        btn_save.pack(side=RIGHT)
        tmp_frame1 = Frame(canvas[1][2])
        tmp_frame1.pack(fill=BOTH, expand=True)

        for r in range(8):
            for c in range(8):
                canvasTmp = Entry(tmp_frame1, width=5)
                canvasTmp.insert(0, '%.2f'%(cv2.flip(target_Mean_Pixel.reshape(8, 8), 0)[r, c]))
                canvasTmp.grid(row=r, column=c)
                entry_ir.append(canvasTmp)


def listbox_click(event):
    global CWD, df, menu_click, canvas, label_ir_arr, label_ir, entry_ir, dataIndex, preClickIndex, viewFileName
    clickIndex = lb_dir.curselection()
    if lb_dir.get(clickIndex) == ():
        return
    elif lb_dir.get(clickIndex) == '..':
        txt_cwd.config(text=CWD)
    else:
        txt_cwd.config(text=CWD+lb_dir.get(clickIndex))
    if '.csv' in lb_dir.get(clickIndex):
        viewFileName = lb_dir.get(clickIndex)
        view()


def listbox_dclick(event):
    global CWD
    clickIndex = lb_dir.curselection()
    if CWD == '/':
        CWD = lb_dir.get(clickIndex)+'/'
        viewListBox(lb_dir)
    elif lb_dir.get(clickIndex) == '..':
        CWD = '/'.join(CWD.split('/')[:-2])+'/'
        viewListBox(lb_dir)
    elif os.path.isdir(CWD+lb_dir.get(clickIndex)):
        CWD = CWD+lb_dir.get(clickIndex)+'/'
        viewListBox(lb_dir)


def viewListBox(lb):
    lb.delete(0, lb.size())
    if CWD == '/':
        lb.insert(lb.size(), 'C:')
        lb.itemconfig(lb.size()-1, bg='gray')
        lb.insert(lb.size(), 'D:')
        lb.itemconfig(lb.size()-1, bg='gray')
        txt_cwd.config(text=CWD)
    else:
        dic = searchDirecotry(CWD)
        lb.insert(lb.size(), '..')
        lb.itemconfig(lb.size()-1)
        if 'dir' in dic:
            for i in range(len(dic['dir'])):
                lb.insert(lb.size(), dic['dir'][i])
                lb.itemconfig(lb.size()-1, bg='gray')
        if 'file' in dic:
            for i in range(len(dic['file'])):
                lb.insert(lb.size(), dic['file'][i])
        txt_cwd.config(text=CWD+lb.get(1))

    lb.bind("<<ListboxSelect>>", listbox_click)
    lb.bind("<Double-Button-1>", listbox_dclick)


viewListBox(lb_dir)
lb_dir.pack(side=LEFT, fill=Y)
frame_info = Frame(frame)
frame_info.pack()
txt_dataInfo = Label(frame_info, text='')
txt_dataInfo.pack(side=LEFT)

entry_dataInfo = Entry(frame_info, width=5)
entry_dataInfo.pack(side=RIGHT)
entry_dataInfo.insert(0, '0')


def click_DataNum(event, x):
    global dataIndex
    if x == -1:
        dataIndex = int(entry_dataInfo.get())
    elif x == 1:
        dataIndex += 1
    elif x == 2:
        dataIndex -= 1
        if dataIndex < 0:
            dataIndex = 0
    view()


entry_dataInfo.bind('<Return>', lambda event, x=-1: click_DataNum(event, x))
entry_dataInfo.bind('<Up>', lambda event, x=1: click_DataNum(event, x))
entry_dataInfo.bind('<Down>', lambda event, x=2: click_DataNum(event, x))


def click_randomDataNum(event):
    global dataIndex
    dataIndex = np.random.randint(len(df))
    view()


btn_randomDataNum = Button(frame_info, text='random')
btn_randomDataNum.pack(side=LEFT)
btn_randomDataNum.bind('<Button-1>', click_randomDataNum)

frame1 = Frame(frame)
frame1.pack(fill=BOTH, expand=True)

canvas = []
for row in range(canvasSize[0]):
    canvasrow = []
    for col in range(canvasSize[1]):
        canvasTmp = Frame(frame1)
        canvasTmp.grid(row=row, column=col)
        canvasrow.append(canvasTmp)
    canvas.append(canvasrow)

window.mainloop()



