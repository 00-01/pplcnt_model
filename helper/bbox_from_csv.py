import cv2
import pandas as pd


SAVE = 0
SHOW = 1
r, g, b = (0,0,255), (0,255,0), (255,0,0)

csv_file = f"input/output.csv"
csv = pd.read_csv(csv_file)

img_name = ""
boxes = []
for i in range(len(csv)):
    if csv.iloc[i,0] != img_name:
        img_name = csv.iloc[i,0]
        img = cv2.imread(f"input/{img_name}", cv2.COLOR_BGR2RGB)
        for box in boxes:
            cv2.rectangle(img, (box[0], box[2]), (box[1], box[3]), r, 1)
            cv2.putText(img, box[4], (box[0]-2, box[2]-1), cv2.FONT_HERSHEY_SIMPLEX, 0.2, g, 1)

        if SHOW == 1:
            window_name = f"out"
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
            cv2.imshow(window_name, img)
            key = cv2.waitKey()
            if key == 27:
                cv2.destroyAllWindows()

        if SAVE == 1:
            out_path = f"output/{img_name}"
            cv2.imwrite(out_path, img)

        boxes.clear()

    elif csv.iloc[i,0] == img_name:
        x_min, x_max, y_min, y_max = csv.iloc[i, 1], csv.iloc[i, 2], csv.iloc[i, 3], csv.iloc[i, 4]
        score = f"{csv.iloc[i,6]:.2f}"
        boxes.append([x_min, x_max, y_min, y_max, score])
