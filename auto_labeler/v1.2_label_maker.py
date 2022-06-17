import csv
import pandas as pd
from glob import glob


machine = ""
base = f"{machine}/media/z/0/MVPC10/DATA/v1.1/RAW"
device = "device_03"
datapath = f"{base}/{device}"
csvfile = f"{datapath}/output_new_path.csv"
image_list = glob(f"{datapath}/**/*.png", recursive=False)

df = pd.read_csv(f'{csvfile}')
df1 = df['image_name']


def none_adder(full_list, df):
    li = []
    for i in full_list:
        val = i in df.values
        if val == False:
            li.append([i, 0])

    return li


## concate save filename and cnt
def boxLabel_to_cntLabel(df):
    li = []
    cnt = 1
    previous = ""
    for index, row in df.iteritems():
        current = row
        if previous != current:
            if previous != "":
                li.append([previous, cnt])
            cnt = 1
            previous = current
        else:
            cnt += 1

    return li


def column_sorter(column):
    col = column.sort()
    return col


def key(elem):
    return elem[0]


none_li = none_adder(image_list, df1)

li = boxLabel_to_cntLabel(df1)

li += none_li

li.sort(key=key)
# column_sorter()

## save to csv
csv_out = f"{datapath}/v1.2_output.csv"
fields = ['img_name', 'cnt']
with open(csv_out, 'w') as f:
    write = csv.writer(f)

    write.writerow(fields)
    write.writerows(li)
