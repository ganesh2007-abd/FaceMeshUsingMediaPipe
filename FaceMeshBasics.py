import cv2 as cv
import mediapipe as mp
import time

cap = cv.VideoCapture(r"C:\Users\HP\Downloads\2_facedetecting.mp4")

ptime=0

mpDraw = mp.solutions.drawing_utils
mpfaceMesh = mp.solutions.face_mesh
faceMesh = mpfaceMesh.FaceMesh()

Drawspecs = mpDraw.DrawingSpec(thickness=1,circle_radius=2)

while True:
    success,img = cap.read()
    if not success:
        break
    img = cv.resize(img, (640, 480))
    # img = cv.resize(img,None,fx=0.25,fy=0.25,interpolation=cv.INTER_AREA)
    imgRGB = cv.cvtColor(img,cv.COLOR_BGR2RGB)
    results = faceMesh.process(imgRGB)
    if results.multi_face_landmarks:
        for facelm in results.multi_face_landmarks:
            mpDraw.draw_landmarks(img,facelm,mpfaceMesh.FACEMESH_CONTOURS,Drawspecs,Drawspecs)
            
    cv.imshow("Image",img)


    if cv.waitKey(1) == ord('q'):
        break

cap.release()
cv.destroyAllWindows()