#!/usr/bin/env python 
# -*- coding:utf-8 -*-
import os
import cv2
from PIL import Image


def del_file(path):
    for i in os.listdir(path):
        path_file = os.path.join(path, i)
        if os.path.isfile(path_file):
            os.remove(path_file)
        else:
            del_file(path_file)

def find_car_num_brod(path):
    watch_cascade = cv2.CascadeClassifier('cascade.xml')
    # 先读取图片
    image = cv2.imread(path)
    resize_h = 1000
    height = image.shape[0]
    scale = image.shape[1] / float(image.shape[0])
    image = cv2.resize(image, (int(scale * resize_h), resize_h))
    image_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    watches = watch_cascade.detectMultiScale(image_gray, 1.2, 2, minSize=(36, 9), maxSize=(36 * 40, 9 * 40))
    print("检测到车牌数", len(watches))
    k=len(watches)
    for (x, y, w, h) in watches:
            #cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 1)
            cut_img = image[y:y + h, x+3:x + w+3]  # 裁剪坐标为[y0:y1, x0:x1]
            cv2.imwrite("tmp/chepai_img"+str(k)+".jpg", cut_img)
            im = Image.open("tmp/chepai_img"+str(k)+".jpg")
            size = 720, 180
            mmm = im.resize(size, Image.ANTIALIAS)
            mmm.save("tmp/chepai_img"+str(k)+".jpg", "JPEG", quality=95)
            k=k-1

if __name__ == '__main__':
    path = "./tmp"
    del_file(path)
    find_car_num_brod()