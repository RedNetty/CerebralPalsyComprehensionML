import cv2
import numpy as np
from keras.models import load_model


windowName = "Live Camera Input" # window name
windowNameBG = "Background Model" # window name
windowNameFG = "Foreground Objects" # window name
windowNameFGP = "Foreground Probabiity" # window name

cv2.namedWindow(windowName, cv2.WINDOW_NORMAL)
cv2.namedWindow(windowNameBG, cv2.WINDOW_NORMAL)
cv2.namedWindow(windowNameFG, cv2.WINDOW_NORMAL)
cv2.namedWindow(windowNameFGP, cv2.WINDOW_NORMAL)

model = load_model('Hand_Gesture_Recognize.h5')


gesture = {
    0: "Left hand up",
    1: "Hands down",
    2: "Right hand up"
}


def predict(hand):
    img = cv2.resize(hand, (50,50) )
    img = np.array(img)
    img = img.reshape( (1,50,50,1) )
    img = img/255.0
    res = model.predict( img ) 
    max_ind = res.argmax()
    return gesture[ max_ind ]

mog = cv2.createBackgroundSubtractorMOG2(history=2000, varThreshold=16, detectShadows=True)

vc = cv2.VideoCapture(0)
rval, frame = vc.read()


old_pred_text = ""
pred_text = ""
count_frames = 0
tot_string = ""

while True:
    
    if frame is not None: 
        
        frame = cv2.flip(frame, 1)
        fgmask = mog.apply(frame)
        fgthres = cv2.threshold(fgmask.copy(), 200, 255, cv2.THRESH_BINARY)[1]
        fgdilated = cv2.dilate(fgthres, kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3)), iterations = 3)
        bgmodel = mog.getBackgroundImage()
        
        old_pred_text = pred_text
        
        pred_text = predict(fgdilated)
        
        if old_pred_text == pred_text:
            count_frames += 1
        else:
            count_frames = 0
        
        if count_frames > 10 and pred_text != "":
            tot_string = "Predicted gesture - " + pred_text
            count_frames = 0
            
        cv2.putText(frame, tot_string, (10, 50), cv2.FONT_HERSHEY_TRIPLEX, 1, (102, 0, 51))

        cv2.imshow(windowName,frame)
        cv2.imshow(windowNameFG,fgdilated)
        cv2.imshow(windowNameFGP,fgmask)
        cv2.imshow(windowNameBG, bgmodel)
        
    rval, frame = vc.read()
    keypress = cv2.waitKey(1)
    
    if keypress == ord('q'):
        break

vc.release()
cv2.destroyAllWindows()
cv2.waitKey(1)

