import cv2
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from PIL import Image
import PIL.ImageOps
import os,ssl,time



 #Setting up and HTTps Context
if (not os.environ.get('PYTHONHTTPSVERIFY','')and
    getattr(ssl,'_create_unverified_context',None)):
    ssl.create_default_context=ssl._create_unverified_context


X,y = fetch_openml('mnist_784',version = 1, return_X_y= True)
print(pd.Series(y).value_counts())
classes = ['A','B','C','D','E','F','G','H','I','J']
nclasses = len(classes)

X_train, X_test, y_train, y_test = train_test_split(X,y,train_size = 7500,test_size = 2500,random_state = 9)

X_train_scale = X_train/255.0
X_test_scale = X_test/255.0

clf = LogisticRegression(solver = 'saga',multi_class = 'multinomial').fit(X_train_scale,y_train)

y_pred = clf.predict(X_test_scale)
print("accuracy = ", accuracy_score(y_test, y_pred))


#start the camera
cap=cv2.VideoCapture(0)

while(True):
    try:
        ret,frame=cap.read()

        gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

        height,width=gray.shape
        upper_left = (int(width / 2 - 56), int(height / 2 - 56))
        bottom_right = (int(width / 2 + 56), int(height / 2 + 56))
        cv2.rectangl(gray.upper_left,bottom_right,(0,255,0),2)

        #Region of Interest , to only consider that area
        roi=gray[upper_left[1]:bottom_right[1],upper_left[0]:bottom_right[0]]


        im_pil=Image.fromarray(roi) 

        image_bw=im_pil.convert('L')
        image_bw_resized=image_bw_resized((28,28), Image.ANTIALIAS)

        image_bw_resized_inverted = PIL.ImageOps.invert(image_bw_resized)
        pixel_filter = 20
        min_pixel = np.percentile(image_bw_resized_inverted, pixel_filter)
        image_bw_resized_inverted_scaled = np.clip(image_bw_resized_inverted-min_pixel, 0, 255)
        max_pixel = np.percentile(image_bw_resized_inverted, pixel_filter)
        image_bw_resized_inverted_scaled = np.asarray(image_bw_resized_inverted_scaled)/max_pixel
        test_sample=np.array(image_bw_resized_inverted_scaled)
        test_pred=clf.predict(test_sample)
        print("Predicted class is:",test_pred)

        cv2.imshow('frame',gray)
        if cv2.waitKey(1) & 0xFF==ord('q'):
            break
    except Exception as e :
        pass

#when everything is done close all the files 
cap.release()
cv2.destroyAllWindows()