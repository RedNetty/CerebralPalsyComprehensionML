import numpy as np
import cv2
import os
import pandas as pd

windowName = "Live Camera Input" # window name
windowNameBG = "Background Model" # window name
windowNameFG = "Foreground Objects" # window name
windowNameFGP = "Foreground Probabiity" # window name

cv2.namedWindow(windowName, cv2.WINDOW_NORMAL)
cv2.namedWindow(windowNameBG, cv2.WINDOW_NORMAL)
cv2.namedWindow(windowNameFG, cv2.WINDOW_NORMAL)
cv2.namedWindow(windowNameFGP, cv2.WINDOW_NORMAL)


df = pd.read_excel('Videos_Intention.xlsx')
df.loc[df['Left Forearm '] == 2]

vc = cv2.VideoCapture('./gesture1.MOV')
rval, frame = vc.read()
pic_no = 0
total_pic = 1200
path = './gesture/1'

flag_capturing = True
vc.set(cv2.CAP_PROP_POS_AVI_RATIO,1)
milliSec = vc.get(cv2.CAP_PROP_POS_MSEC)
milliSecToSkip = 0.1*milliSec

#################### Setting up parameters ################

seconds = 0.001*milliSecToSkip
fps = vc.get(cv2.CAP_PROP_FPS) # Gets the frames per second
skipNumberFrames = int(fps * seconds)
#skipNumberFrames = 1
print(fps)
print(skipNumberFrames)
vc.release()
frameCount = 0
vc = cv2.VideoCapture('./gesture1.MOV')
rval, frame = vc.read()

mog = cv2.createBackgroundSubtractorMOG2(history=2000, varThreshold=16, detectShadows=True)

while True:
	
	if frame is not None:

		frame = cv2.flip(frame, 1)
		fgmask = mog.apply(frame)
		fgthres = cv2.threshold(fgmask.copy(), 200, 255, cv2.THRESH_BINARY)[1]
		fgdilated = cv2.dilate(fgthres, kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3)), iterations = 3)
		bgmodel = mog.getBackgroundImage()

		cv2.imshow(windowName,frame)
		cv2.imshow(windowNameFG,fgdilated)
		cv2.imshow(windowNameFGP,fgmask)
		cv2.imshow(windowNameBG, bgmodel)
                
		if flag_capturing:
			frameCount+=1
			print(frameCount)
			if frameCount >= skipNumberFrames:
			    pic_no += 1
			    save_img = cv2.resize( fgdilated, (50,50) )
			    save_img = np.array(save_img)
			    cv2.imwrite(path + "/" + str(pic_no) + ".jpg", save_img)

	rval, frame = vc.read()
	keypress = cv2.waitKey(1)
    
	# if pic_no == total_pic:
	# 	flag_capturing = False
	# 	break
    
	if keypress == ord('q'):
		break
	elif keypress == ord('c'):
		flag_capturing = True

vc.release()
cv2.destroyAllWindows()
cv2.waitKey(1)

