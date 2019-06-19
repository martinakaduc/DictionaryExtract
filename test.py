from imutils.object_detection import non_max_suppression
import numpy as np
import pytesseract
import argparse
import cv2 as cv

img = cv.imread('TD-0019.jpg')
(H, W) = img.shape[:2]
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
blur = cv.GaussianBlur(gray,(3,3),0)
ret3, th3 = cv.threshold(blur,0,255,cv.THRESH_BINARY+cv.THRESH_OTSU)


dst = cv.erode(th3, kernel=np.ones((6, 30)))
cont, hier = cv.findContours(dst, mode=cv.RETR_LIST, method=cv.CHAIN_APPROX_SIMPLE)

cropped = []
rectList = []
for box in cont:
    x, y, width, height = cv.boundingRect(box)
    rectList.append((x,y,width,height))

rectList = list(filter(lambda rect: rect[2] > 100 and rect[3] > 35, rectList))

for i in range(4):
    rectColumn = list(filter(lambda rect: rect[0] < W/(4-i), rectList))
    rectColumn.sort(key = lambda rect: rect[1])
    for rect in rectColumn:
        cv.rectangle(img, rect, (0,255,0), 3)
        x, y, width, height = rect
        cropped.append(th3[y:y+height, x:x+width])
    rectList = list(filter(lambda rect: rect not in rectColumn, rectList))

config = ("-l vie --oem 0 --psm 7")
outputText = []
for crop in cropped:
    outputText.append(pytesseract.image_to_string(crop, config=config))
    # outputText.append(pytesseract.image_to_data(crop, config=config))
    # a = pytesseract.image_to_pdf_or_hocr(crop, extension='hocr', config=config)
    # with open("result.xml", "wb") as wf:
    #     wf.write(a)

for text in outputText:
    print(text)
cv.imshow('image',cv.resize(img, (1136,640)))

k = cv.waitKey(0)
if k == 27:
    cv.destroyAllWindows()
