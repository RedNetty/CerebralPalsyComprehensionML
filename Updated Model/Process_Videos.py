import numpy as np
import cv2
import os
import sys

arguments = sys.argv[1:]
count = len(arguments)
silentMode = True

if count != 0:
	if sys.argv[1] == "-s":
		silentMode = True
	else:
		silentMode = False
else:
	silentMode = False

windowName = "Video Input" # window name
windowNameBG = "Background Model" # window name
windowNameFG = "Foreground Objects" # window name
windowNameFGP = "Foreground Probabiity" # window name

if not silentMode:
	cv2.namedWindow(windowName, cv2.WINDOW_NORMAL)
	cv2.namedWindow(windowNameBG, cv2.WINDOW_NORMAL)
	cv2.namedWindow(windowNameFG, cv2.WINDOW_NORMAL)
	cv2.namedWindow(windowNameFGP, cv2.WINDOW_NORMAL)

dataPath = './movies/'			# path where each gesture video is placed 
picturePath = './pictures/'		# path where the extracted images will be placed

folders = os.listdir(dataPath)

dataFiles, dataFolders = [], []
picNo = 0

# Get all the folders(class) name and list of all the data files
for folderName in folders:
	movieFiles = os.listdir(dataPath + folderName)
	for movie in movieFiles:
		movieFullPath = dataPath + folderName + '/' + movie
		dataFiles.append(movieFullPath)
	dataFolders.append(folderName)

mog = cv2.createBackgroundSubtractorMOG2(history=2000, varThreshold=16, detectShadows=True)

for item in dataFiles:
	
	temp = item[len(dataPath):]
	className = temp[:temp.find('/')]	# class name
	print("Processing {0}".format(item))

	vc = cv2.VideoCapture(item)

	while(vc.isOpened()):
		ret, frame = vc.read()

		if frame is not None:
			frame = cv2.flip(frame, 1)
			fgMask = mog.apply(frame)
			fgThres = cv2.threshold(fgMask.copy(), 200, 255, cv2.THRESH_BINARY)[1]
			fgDilated = cv2.dilate(fgThres, kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3)), iterations = 3)
			bgmodel = mog.getBackgroundImage()

			if not silentMode:
				cv2.imshow(windowName,frame)
				cv2.imshow(windowNameFG,fgDilated)
				cv2.imshow(windowNameFGP,fgMask)
				cv2.imshow(windowNameBG, bgmodel)
					
			saveImg = cv2.resize( fgDilated, (50,50) )
			saveImg = np.array(saveImg)
			picNo += 1
			
			if not os.path.exists(picturePath + '/' + className):
				os.makedirs(picturePath + '/' + className)
			
			cv2.imwrite(picturePath + '/' + className + '/' + str(picNo) + ".jpg", saveImg)	
			print("created file - {0}".format(picturePath + className + '/' + str(picNo) + ".jpg", saveImg))
		
		cv2.waitKey(1)
		if ret == False:
			break
	vc.release()
cv2.destroyAllWindows()
cv2.waitKey(1)

