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

    ctime = time.time()
    fps = 1/(ctime-ptime)
    ptime=ctime
    cv.putText(img,str(int(fps)),(20,70),cv.FONT_HERSHEY_DUPLEX,2,(0,255,0),2)
    
    results = faceMesh.process(imgRGB)
    if results.multi_face_landmarks:
        for facelm in results.multi_face_landmarks:
            mpDraw.draw_landmarks(img,facelm,mpfaceMesh.FACEMESH_CONTOURS,Drawspecs,Drawspecs)
            for id,lm in enumerate(facelm.landmark):
                ih,iw,ic = img.shape
                x,y = int(lm.x*iw),int(lm.y*ih)
                print(id,x,y)

    cv.imshow("Image",img)


    if cv.waitKey(1) == ord('q'):
        break

cap.release()
cv.destroyAllWindows()