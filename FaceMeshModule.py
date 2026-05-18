import cv2 as cv
import mediapipe as mp
import time

class FaceMeshDetector():
    def __init__(self,staticmode=False,maxFaces=2,minDetectioncon=0.5,minTrackingcon=0.5):
        self.staticmode=staticmode
        self.maxFaces=maxFaces
        self.minDetectioncon=minDetectioncon
        self.minTrackingcon=minTrackingcon

        self.mpDraw = mp.solutions.drawing_utils
        self.mpfaceMesh = mp.solutions.face_mesh
        self.faceMesh = self.mpfaceMesh.FaceMesh(static_image_mode=self.staticmode,max_num_faces=self.maxFaces,min_detection_confidence=self.minDetectioncon,min_tracking_confidence=self.minTrackingcon)

        self.Drawspecs = self.mpDraw.DrawingSpec(thickness=1,circle_radius=1)


    def findMesh(self,img,draw=True):
        self.imgRGB = cv.cvtColor(img,cv.COLOR_BGR2RGB)
        self.results = self.faceMesh.process(self.imgRGB)
        faces=[]
        if self.results.multi_face_landmarks:
            for facelm in self.results.multi_face_landmarks:
                self.mpDraw.draw_landmarks(img,facelm,self.mpfaceMesh.FACEMESH_TESSELATION,self.Drawspecs,self.Drawspecs)
                face=[]
                for id,lm in enumerate(facelm.landmark):
                    ih,iw,ic = img.shape
                    x,y = int(lm.x*iw),int(lm.y*ih)

                    face.append(lm)

                faces.append(face)


        return img,faces
                    

def main():
    cap = cv.VideoCapture(r"C:\Users\HP\Downloads\2_facedetecting.mp4")
    ptime=0
    detector=FaceMeshDetector()
    while True:
        success,img = cap.read()
        if not success:
            break
        img = cv.resize(img, (640, 480))
        img,faces = detector.findMesh(img)

        ctime = time.time()
        fps = 1/(ctime-ptime)
        ptime=ctime
        cv.putText(img,str(int(fps)),(20,70),cv.FONT_HERSHEY_DUPLEX,2,(0,255,0),2)

        cv.imshow("Image",img)

        if cv.waitKey(1) == ord('q'):
            break

    cap.release()
    cv.destroyAllWindows()

if __name__ == "__main__":
    main()