import cv2
import numpy as np
from keras.models import load_model
import sys

arguments = sys.argv[1:]
count = len(arguments)
inputFile = ""

if count == 2:
    if sys.argv[1] == "-p":
        if str(sys.argv[2]) != "":
            inputFile = sys.argv[2]
        else:
            print("Usage - python Recognize_Video.py -p abc.mov")
            sys.exit(0)
    else:
        print("No video file provided for prediction")
        sys.exit(0)
else:
    print("Usage - python Recognize_Video.py -p abc.mov")
    sys.exit(0)


windowName = "Video Input" # window name
windowNameBG = "Background Model" # window name
windowNameFG = "Foreground Objects" # window name
windowNameFGP = "Foreground Probabiity" # window name

cv2.namedWindow(windowName, cv2.WINDOW_NORMAL)
cv2.namedWindow(windowNameBG, cv2.WINDOW_NORMAL)
cv2.namedWindow(windowNameFG, cv2.WINDOW_NORMAL)
cv2.namedWindow(windowNameFGP, cv2.WINDOW_NORMAL)

model = load_model('Gesture_Model.h5')


gesture = {
    0: "Dad",
    1: "Mom",
    2: "Yes",
    3: "No",
    4: "Happy",
    5: "Sad"
}


def predict(hand):
    img = cv2.resize(hand, (50,50) )
    img = np.array(img)
    img = img.reshape( (1,50,50,1) )
    img = img/255.0
    res = model.predict( img ) 
    max_ind = res.argmax()
    #return gesture[ max_ind ]
    return gesture[ max_ind ], "{0}-{1:.5f}, {2}-{3:.5f}, {4}-{5:.5f}, {6}-{7:.5f}, {8}-{9:.5f}".format(gesture[0], res[0][0], gesture[1], res[0][1], gesture[2], res[0][2], gesture[3], res[0][3], gesture[4], res[0][4])

mog = cv2.createBackgroundSubtractorMOG2(history=2000, varThreshold=16, detectShadows=True)

vc = cv2.VideoCapture(inputFile)



old_pred_text = ""
pred_text = ""
count_frames = 0
tot_string = ""

while (vc.isOpened()):
    rval, frame = vc.read()
    
    if frame is not None: 
        
        frame = cv2.flip(frame, 1)
        fgmask = mog.apply(frame)
        fgthres = cv2.threshold(fgmask.copy(), 200, 255, cv2.THRESH_BINARY)[1]
        fgdilated = cv2.dilate(fgthres, kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3)), iterations = 3)
        bgmodel = mog.getBackgroundImage()
        
        old_pred_text = pred_text
        
        pred_text, fullText = predict(fgdilated)
        
        if old_pred_text == pred_text:
            count_frames += 1
        else:
            count_frames = 0
        
        if count_frames > 10 and pred_text != "":
            tot_string = fullText
            count_frames = 0

        cv2.putText(frame, "Predicted intent - {0}".format(pred_text), (220, 90), cv2.FONT_HERSHEY_DUPLEX, 2, (102, 0, 51), 3)
        cv2.putText(frame, tot_string, (220, 130), cv2.FONT_HERSHEY_DUPLEX, 1, (102, 0, 51), 2)

        cv2.imshow(windowName,frame)
        cv2.imshow(windowNameFG,fgdilated)
        cv2.imshow(windowNameFGP,fgmask)
        cv2.imshow(windowNameBG, bgmodel)
        
    keypress = cv2.waitKey(1)
    if keypress == ord('q'):
        break
    if rval == False:
        break

vc.release()
cv2.destroyAllWindows()
cv2.waitKey(1)

