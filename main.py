import tkinter as tk
from tkinter.filedialog import *
from PIL import Image, ImageTk
import Recognition
import Cut
import cv2
import img_math
import Predict
import numpy as np

window = tk.Tk()
window.title('车牌识别')
window.geometry('900x500')  # 这里的乘是小x
pic_path=""

def open_pic():
    global pic_path
    pic_path = askopenfilename(title="选择识别图片")
    if pic_path:
        img = Image.open(pic_path)
        photo = ImageTk.PhotoImage(img)
        img = img.resize((680, 500), Image.ANTIALIAS)
        photo=ImageTk.PhotoImage(img)
        label = tk.Label(window,anchor="nw")
        label.config(image=photo)
        label.image=photo
        label.place(x=0,y=0)

def xzdingwei():
    lable3 = tk.Label(window, text="形状定位如下", font=('Arial', 15))
    lable3.place(x=750, y=0)
    Recognition.find_car_num_brod(pic_path)
    img = Image.open("tmp/chepai_img1.jpg")
    photo = ImageTk.PhotoImage(img)
    img = img.resize((160, 40), Image.ANTIALIAS)
    photo = ImageTk.PhotoImage(img)
    label2 = tk.Label(window,anchor="ne")
    label2.config(image=photo)
    label2.image=photo
    label2.place(x=735,y=30)

def ysdingwei():
    lable4 = tk.Label(window, text="颜色定位如下", font=('Arial', 15))
    lable4.place(x=750, y=100)
    filename = img_math.img_read(pic_path)
    oldimg = filename
    img_contours = oldimg
    pic_hight, pic_width = img_contours.shape[:2]
    lower_blue = np.array([100, 110, 110])
    upper_blue = np.array([130, 255, 255])
    lower_yellow = np.array([15, 55, 55])
    upper_yellow = np.array([50, 255, 255])
    lower_green = np.array([50, 50, 50])
    upper_green = np.array([100, 255, 255])
    hsv = cv2.cvtColor(filename, cv2.COLOR_BGR2HSV)
    mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)
    mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)
    mask_green = cv2.inRange(hsv, lower_yellow, upper_green)
    output = cv2.bitwise_and(hsv, hsv, mask=mask_blue + mask_yellow + mask_green)
    # 根据阈值找到对应颜色
    output = cv2.cvtColor(output, cv2.COLOR_BGR2GRAY)
    Matrix = np.ones((20, 20), np.uint8)
    img_edge1 = cv2.morphologyEx(output, cv2.MORPH_CLOSE, Matrix)
    img_edge2 = cv2.morphologyEx(img_edge1, cv2.MORPH_OPEN, Matrix)
    card_contours = img_math.img_findContours(img_edge2)
    card_imgs = img_math.img_Transform(card_contours, oldimg, pic_width, pic_hight)
    colors, car_imgs = img_math.img_color(card_imgs)
    cv2.imwrite('tmp/chepai_img1.jpg', card_imgs[0])
    print(colors[0])
    img = Image.open("tmp/chepai_img1.jpg")
    photo = ImageTk.PhotoImage(img)
    img = img.resize((160, 40), Image.ANTIALIAS)
    photo = ImageTk.PhotoImage(img)
    label2 = tk.Label(window,anchor="ne")
    label2.config(image=photo)
    label2.image=photo
    label2.place(x=735,y=130)

def shibie():
    lable5 = tk.Label(window, text="识别到的车牌号码为", font=('Arial', 15))
    lable5.place(x=710, y=350)
    Cut.del_file("./test_images")
    Cut.predict("tmp/chepai_img1.jpg")
    lable6 = tk.Label(window,font=('Arial', 15))
    Predict.pre()
    lable6.config(text=Predict.pre())
    lable6.place(x=750, y=390)


menubar = tk.Menu(window)
filemenu = tk.Menu(menubar, tearoff=0)
menubar.add_cascade(label='文件', menu=filemenu)
filemenu.add_command(label='打开图片', command=open_pic)
filemenu.add_separator()  # 添加一条分隔线
filemenu.add_command(label='退出', command=window.quit)  # 用tkinter里面自带的quit()函数
editmenu = tk.Menu(menubar, tearoff=0)
menubar.add_cascade(label='运行', menu=editmenu)
editmenu.add_command(label='形状定位', command=xzdingwei)
editmenu.add_separator()  # 添加一条分隔线
editmenu.add_command(label='颜色定位', command=ysdingwei)
editmenu.add_separator()  # 添加一条分隔线
editmenu.add_command(label='字符识别', command=shibie)
window.config(menu=menubar)
window.mainloop()