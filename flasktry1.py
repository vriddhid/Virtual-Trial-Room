from flask import Flask, render_template, request
import json
from flask_cors import CORS
import numpy as np
import cv2                              # Library for image processing
from math import floor
import os
import time
app = Flask(__name__)
CORS(app)

@app.route('/')
def index():
    return render_template('index.html')
@app.route('/men.html')
def plot():
    return render_template('men.html')
@app.route('/women.html')
def ploty():
    return render_template('women.html')
@app.route('/predict', methods=['GET','POST'])
def predict():
    shirtno = int(request.form["shirt"])
    imgarr=["shirt1.jpeg","shirt2.jpeg","","shirt6.png","11.jpeg","9.png","2.png"]
    imgshirt = cv2.imread(imgarr[shirtno-1],1)
    musgray = cv2.cvtColor(imgshirt,cv2.cv2.COLOR_BGR2GRAY) #grayscale conversion
    ret, orig_mask = cv2.threshold(musgray,150 , 255, cv2.THRESH_BINARY)
    orig_mask_inv = cv2.bitwise_not(orig_mask)
    origshirtHeight, origshirtWidth = imgshirt.shape[:2]
    face_cascade=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    #cap=cv2.VideoCapture(0)
    camera_port = 0
    cap = cv2.VideoCapture(camera_port, cv2.CAP_DSHOW)
    ret,img=cap.read()
    img_h, img_w = img.shape[:2]
    while True:
            ret,img=cap.read()
            gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
            faces=face_cascade.detectMultiScale(gray,1.3,5)

            for (x,y,w,h) in faces:
                cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)

                face_w = w
                face_h = h
                face_x1 = x
                face_x2 = face_x1 + face_h
                face_y1 = y
                face_y2 = face_y1 + face_h

        # set the shirt size in relation to tracked face
                shirtWidth = 3 * face_w
                shirtHeight = int(shirtWidth * origshirtHeight / origshirtWidth)


                shirt_x1 = face_x2 - int(face_w/2) - int(shirtWidth/2) #setting shirt centered wrt recognized face
                shirt_x2 = shirt_x1 + shirtWidth
                shirt_y1 = face_y2 + 5 # some padding between face and upper shirt. Depends on the shirt img
                shirt_y2 = shirt_y1 + shirtHeight

        # Check for clipping
                if shirt_x1 < 0:
                    shirt_x1 = 0
                if shirt_y1 < 0:
                    shirt_y1 = 0
                if shirt_x2 > img_w:
                    shirt_x2 = img_w
                if shirt_y2 > img_h:
                    shirt_y2 = img_h

                shirtWidth = shirt_x2 - shirt_x1
                shirtHeight = shirt_y2 - shirt_y1
                if shirtWidth < 0 or shirtHeight < 0:
                    continue

        # Re-size the original image and the masks to the shirt sizes
                shirt = cv2.resize(imgshirt, (shirtWidth,shirtHeight), interpolation = cv2.INTER_AREA) #resize all,the masks you made,the originla image,everything
                mask = cv2.resize(orig_mask, (shirtWidth,shirtHeight), interpolation = cv2.INTER_AREA)
                mask_inv = cv2.resize(orig_mask_inv, (shirtWidth,shirtHeight), interpolation = cv2.INTER_AREA)

        # take ROI for shirt from background equal to size of shirt image
                roi = img[shirt_y1:shirt_y2, shirt_x1:shirt_x2]


        # roi_bg contains the original image only where the shirt is not
        # in the region that is the size of the shirt.
                roi_bg = cv2.bitwise_and(roi,roi,mask = mask_inv)
                roi_fg = cv2.bitwise_and(shirt,shirt,mask = mask)
                dst = cv2.add(roi_bg,roi_fg)
                img[shirt_y1:shirt_y2, shirt_x1:shirt_x2] = dst


                break
            cv2.imshow('img',img)
            #time.sleep(40)
            if cv2.waitKey(100000) == ord('q'):
                break;
            cap.release() # Destroys the cap object
            cv2.destroyAllWindows() # Destroys all the windows created by imshow
            #time.sleep(40)
            return render_template('index.html')
            #return predict()
            

if __name__ == '__main__':
    app.run(host='0.0.0.0',debug=True,port=5000)
