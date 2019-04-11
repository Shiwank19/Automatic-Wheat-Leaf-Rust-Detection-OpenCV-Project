# -*- coding: utf-8 -*-
"""
Created on Sun Jan 20 17:16:36 2019

@author: fdshiwank
"""

import cv2
import numpy as np
def doNothing():
    pass

def printMouse(event,x,y,flags,params):
    if event==cv2.EVENT_LBUTTONDBLCLK:
        print(res2[x,y])

sample=cv2.imread("C:\\Users\\user\\Desktop\\Python\\Wheat Diseases\\Leaf Rust\\lr04.png")
sample_hsv=cv2.cvtColor(sample,cv2.COLOR_BGR2HSV)
hue=sample_hsv[:,:,0]
s=sample_hsv[:,:,1]
v=sample_hsv[:,:,2]
Z=hue.reshape((-1,1))
Z=np.float32(Z)
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
K=16
ret,label,center=cv2.kmeans(Z,K,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)
center = np.uint8(center)
res = center[label.flatten()]
res2 = res.reshape((hue.shape))

Y=s.reshape((-1,1))
Y=np.float32(Y)
ret_s,label_s,center_s=cv2.kmeans(Y,K,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)
center_s=np.uint8(center_s)
res_s=center_s[label_s.flatten()]
res_s2=res_s.reshape((s.shape))

ter,shades=cv2.threshold(res2,18,255,cv2.THRESH_BINARY_INV)
ter_s,shades_s=cv2.threshold(res_s2,180,255,cv2.THRESH_BINARY)
anded=cv2.bitwise_and(shades,shades_s)
#shades=cv2.GaussianBlur(shades,(3,3),0)
img2, contours, hierarchy = cv2.findContours(anded, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours(sample, contours, -1, (255, 0, 0), 1)
#cv2.adaptiveThreshold(img,255,cv.ADAPTIVE_THRESH_GAUSSIAN_C,cv.THRESH_BINARY,11,2)
cv2.namedWindow("Shades")
cv2.namedWindow("Shades_s")
cv2.namedWindow("Clustered")
#cv2.createTrackbar('shades_s',"Shades_s",0,255,doNothing)
cv2.namedWindow("s")
cv2.createTrackbar("shades","Shades",0,255,doNothing)
cv2.createTrackbar("shades_s","Shades_s",0,255,doNothing)
cv2.setMouseCallback("Clustered",printMouse)
while True:
    th=cv2.getTrackbarPos("shades","Shades")
    th_s=cv2.getTrackbarPos("shades_s","Shades_s")
    ter_s,shades_s=cv2.threshold(res_s2,222,255,cv2.THRESH_BINARY)
    ter,shades=cv2.threshold(res2,19,255,cv2.THRESH_BINARY_INV)
    anded=cv2.bitwise_and(shades,shades_s)
    #shades=cv2.GaussianBlur(shades,(3,3),0)
    img2, contours, hierarchy = cv2.findContours(anded, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(sample, contours, -1, (255, 0, 0), 1)
    cv2.imshow("h",hue)
    cv2.imshow("s",s)
    cv2.imshow("v",v)
    cv2.imshow("Clustered",res2)
    cv2.imshow("Shades",shades)
    cv2.imshow("Original",sample)
    cv2.imshow("Shades_s",shades_s)
    cv2.imshow("Anded",anded)
    if cv2.waitKey(1)==27:
        break
cv2.destroyAllWindows()
white_count=0
black_count=0
width,height=anded.shape[:2]
for i in range(0,width):
    for j in range(0,height):
        if anded[i][j]==255:
            white_count+=1
        else:
            black_count+=1
rust_percent=(white_count/(white_count+black_count))*100
print(rust_percent,"%")
