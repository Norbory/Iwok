import cv2
import mediapipe as mp
import time
import HandTM as htm


pTime=0
cTime = 0
cap=cv2.VideoCapture(0)
detector = htm.handDetector()

while True:
    success, img= cap.read()
    img=detector.findHands(img)
    lmList=detector.findPosition(img)
    if len(lmList)!=0:
        print(lmList[4])

    cTime= time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime

    cv2.putText(img, f'FPS: {int(fps)}', (20,20), cv2.FONT_HERSHEY_COMPLEX, 0.6, (255,0,0), 1)

    cv2.imshow("Manos",img)
    cv2.waitKey(1)