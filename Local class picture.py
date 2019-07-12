# import the necessary packages
import numpy as np
import argparse
import imutils
import pickle
import cv2
import os
import json
import requests
import time
import PySimpleGUI as sg

layout = [
          [sg.Text('Professor you will process picture of class use this with local class image')], 
          [sg.Text('Please complete the required')],    
          [sg.Text('Plese select Material : ', size=(15, 1)), sg.InputCombo(('Antenna','Electronics','Field','Maths','Digital','Micro','Dsp','Communication','C++','Fiber'), size=(40, 5))],
          [sg.Text('Choose A image', size=(35, 1))],
          [sg.Text('Your Folder', size=(15, 1), auto_size_text=False, justification='right'),
          sg.InputText('Default Folder'), sg.FileBrowse()],
          [sg.Submit(), sg.Cancel()]      
         ]

window = sg.Window('Simple data entry window').Layout(layout)         
button, values = window.Read()

print(button, values[0],values[1])
window.Close()
# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
##ap.add_argument("-i", "--image", required=True,
##	help="path to input image")
##ap.add_argument("-d", "--detector", required=True,
##	help="path to OpenCV's deep learning face detector")
##ap.add_argument("-m", "--embedding-model", required=True,
##	help="path to OpenCV's deep learning face embedding model")
##ap.add_argument("-r", "--recognizer", required=True,
##	help="path to model trained to recognize faces")
##ap.add_argument("-l", "--le", required=True,
##	help="path to label encoder")
ap.add_argument("-c", "--confidence", type=float, default=0.5,
	help="minimum probability to filter weak detections")
args = vars(ap.parse_args())
##*** define array to take material and date****##
date_time = time.asctime(time.localtime(time.time()))
##print("*************************************************************************************")
##print("_______ This is the material list Select Your material please__________",end='\n') 
##print(" 1 = digitl,  2 = electronics , 3=micro ,4 =antenna,5 = communiactions, 6 = maths , 7 = fields , 8 = c++ , 9 = DSP , 10 = Fiber ")
##print("++++++++++++++++++++++++++++++++++++++++++++++++++++++ ")
##option = input("professor please select your material from above: ",)
##material_list = ["" ,"digital","electronics","micro","antenna","communiactions","maths","fields","c++","dsp","fiber"]
##selected= int(option)
material = values[0]
##print("profeesor you selected  {} :: ".format (material))
##if material >"10" :
##        print("please enter a correct selection restart app and select from our options")      
## array to hold the attendance##
attendance=[{"name":"none"},{"name":"null"}]
##**********###make the first element of array contain the date and time
date_time = time.asctime(time.localtime(time.time()))
attendance=[{"name":date_time}]

# load our serialized face detector from disk
print("[INFO] loading face detector...")
protoPath = "face_detection_model/deploy.prototxt"
modelPath = "face_detection_model/res10_300x300_ssd_iter_140000.caffemodel"
detector = cv2.dnn.readNetFromCaffe(protoPath, modelPath)

# load our serialized face embedding model from disk
print("[INFO] loading face recognizer...")
embedder = cv2.dnn.readNetFromTorch("openface_nn4.small2.v1.t7")

# load the actual face recognition model along with the label encoder
recognizer = pickle.loads(open("output/recognizer.pickle", "rb").read())
le = pickle.loads(open("output/le.pickle", "rb").read())
# load the image, resize it to have a width of 600 pixels (while
# maintaining the aspect ratio), and then grab the image dimensions
image = cv2.imread("{}".format(values[1]))
image = cv2.resize(image,(3264,2448))
(h, w) = image.shape[:2]
for n in range(5):
  #
    for t in range(4):
        
        image = image[0:h,0:w]
        image_edited = imutils.resize(image,width=600,height=600)
        # construct a blob from the image
        imageBlob = cv2.dnn.blobFromImage(
                cv2.resize(image_edited, (300, 300)), 1.0, (300, 300),
                (104.0, 177.0, 123.0), swapRB=False, crop=False)

        # apply OpenCV's deep learning-based face detector to localize
        # faces in the input image
        detector.setInput(imageBlob) 
        detections = detector.forward()
        # loop over the detections
        for i in range(0, detections.shape[2]):
                # extract the confidence (i.e., probability) associated with the
                # prediction
                confidence = detections[0, 0, i, 2]

                # filter out weak detections
                if confidence > args["confidence"]:
                        # compute the (x, y)-coordinates of the bounding box for the
                        # face
                        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                        (startX, startY, endX, endY) = box.astype("int")

                        # extract the face ROI
                        face = image[startY:endY, startX:endX]
                        (fH, fW) = face.shape[:2]

                        # ensure the face width and height are sufficiently large
                        if fW < 20 or fH < 20:
                                continue
                        # construct a blob for the face ROI, then pass the blob
                        # through our face embedding model to obtain the 128-d
                        # quantification of the face
                        faceBlob = cv2.dnn.blobFromImage(face, 1.0 / 255, (96, 96),
                                (0, 0, 0), swapRB=True, crop=False)
                        embedder.setInput(faceBlob)
                        vec = embedder.forward()

                        # perform classification to recognize the face
                        preds = recognizer.predict_proba(vec)[0]
                        j = np.argmax(preds)
                        proba = preds[j]
                        name = le.classes_[j]
                        # draw the bounding box of the face along with the associated
                        # probability
                        text = "{}: {:.2f}%".format(name, proba * 100)
                        y = startY - 10 if startY - 10 > 10 else startY + 10
                        cv2.rectangle(image, (startX, startY), (endX, endY),
                                (0, 0, 255), 2)
                        cv2.putText(image , text, (startX, y),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
                        #### ****** update attendance arrayfrom recognized images *****###
                        attend = {"name":name}
                        if not any(d["name"] == name for d in attendance):
                                attendance.append(attend.copy())                                
        w = w-70       
    h = h-550
attendance[0] = {"material":material}
attendance[1] = {"date":date_time}
print("all the attended students are :")
print("_____________attendance____________")
print("all the attended students are :",end='\n')
print(attendance)
#print(len(attendance))
###**** json server ****###
data_json = json.dumps(attendance)
headers = {'Content-type': 'application/json'}
#payload = {'json_payload': data_json}
###**** json request ****###
r = requests.post('http://192.168.43.118:8080/api/camera', headers = headers , data=data_json)
##print("+++++++++++++ The JSON result ++++++++++++++")
print(r)

## *** end of json files *** ##

# show the output image
#output = cv2.resize(image,(1500,1000))
#cv2.imshow("Image", output )
cv2.waitKey(0)
cv2.destroyAllWindows()
    
