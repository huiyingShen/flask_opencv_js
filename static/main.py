import numpy as np
import cv2
import cv2.aruco as aruco
from homography import Homography

class HomographyCv(Homography):
    def loadImageTest(self):
        self.im0 = cv2.imread("market_tmap.png")
        self.corners, self.ids, _ = aruco.detectMarkers(self.im0,aruco.Dictionary_get(cv2.aruco.DICT_4X4_250))
        return self

    def getCornersTest(self):
        self.im = cv2.imread("marker_test.jpg")
        corners, ids, _ = aruco.detectMarkers(self.im,aruco.Dictionary_get(cv2.aruco.DICT_4X4_250))
        return corners, ids
    
    def drawMarker(self,bgr,corners):
        assert len(corners)%4 == 0
        nMarker = int(len(corners)/4)
        for n in range(nMarker):
            for i in range(4):
                i1 = (i+3)%4
                x0,y0 = corners[n*4 + i1]
                x1,y1 = corners[n*4 + i]
                cv2.line(bgr, (int(x0),int(y0)), (int(x1),int(y1)), (0, 255, 0), 2)

    def drawPt(self,bgr,x,y):
        cv2.circle(bgr, (x, y), 4, (0, 0, 255), -1)
    
    def getHomographyCv(self,vCn1,vCn2):
        self.h, status = cv2.findHomography(np.array(vCn1),np.array(vCn2))
        # print("getHomographyCv(): \n",self.h)

    def test0(self):
        self.loadImageTest()
        corners,ids = self.getCornersTest()
        vCn1,vCn2 = self.getCornerPairs(corners,ids)
        self.getHomographyCv(vCn1,vCn2 )
        self.verify(vCn1,vCn2)
        self.drawMarker(self.im0,vCn2)
        cv2.imshow("im0",self.im0)
        self.drawMarker(self.im,vCn1)
        cv2.imshow("im",self.im)
        self.getHomographyCv(vCn1,vCn2 )
        cv2.waitKey(0)

    def test1(self):

        self.loadImageTest()
        cv2.imshow("im0",self.im0)
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 800)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 600)
        while True:
            cv2.waitKey(10)
            ret, self.im = cap.read()
            corners, ids, _ = aruco.detectMarkers(self.im,aruco.Dictionary_get(cv2.aruco.DICT_4X4_250))
            cv2.imshow("im",self.im)
            if ids is None or len(ids) < 3: continue
            vCn1,vCn2 = self.getCornerPairs(corners,ids)
            im0 = self.im0.copy()
            self.drawMarker(im0,vCn2)

            self.drawMarker(self.im,vCn1)
            self.getHomography(vCn1,vCn2 )
            self.verify(vCn1,vCn2)
            x0,y0 = 400,400
            xy = self.warp_func(x0,y0)
            x,y = int(xy[0]),int(xy[1])
            cv2.circle(self.im, (x0, y0), 4, (0, 0, 255), -1)
            cv2.circle(im0, (x, y), 4, (0, 0, 255), -1)
            cv2.imshow("im",self.im)
            cv2.imshow("im0",im0)
            cv2.waitKey(100)


if __name__ == '__main__':
    HomographyCv().test1()

