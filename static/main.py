import numpy as np
import cv2
import cv2.aruco as aruco
import base64
from homography import Homography

def data_uri_to_cv2_img(data_uri):
    data_uri += "=" * ((4 - len(data_uri) % 4) % 4) # Base64 needs a string with length multiple of 4. If the string is short, it is padded with 1 to 3 =.
    print("data_uri[:100] = ",data_uri[:100])
    nparr = np.frombuffer(base64.b64decode(data_uri), np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    print(img.shape)
    return img
class HomographyCv(Homography):
    def __init__(self):
        super().__init__()
        self.im0 = None
        self.im = None
        self.dct = aruco.Dictionary_get(cv2.aruco.DICT_4X4_250)

    def loadMapImage(self,im0):
        self.im0 = im0
        self.corners, self.ids, _ = aruco.detectMarkers(self.im0,self.dct)
        cns = []
        for cn in self.corners:
            cns.extend(cn[0])
        self.drawMarker(im0,cns)

    def loadMapDataUri(self,data_uri):
        self.loadMapImage(data_uri_to_cv2_img(data_uri))


    def tryGetHomography(self,im):
        self.im = im
        corners, ids, _ = aruco.detectMarkers(self.im,self.dct)
        if ids is None or len(ids) < 3: return False
        vCn1,vCn2 = self.getCornerPairs(corners,ids)
        self.drawMarker(self.im,vCn1)
        if len(vCn1) < 12: return False
        self.getHomography(vCn1,vCn2)
        return True
    
    def tryGetHomographyDataUri(self,data_uri):
        return self.tryGetHomography(data_uri_to_cv2_img(data_uri))

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
        self.loadMapImage(cv2.imread("market_tmap.png"))
        self.im = cv2.imread("marker_test.jpg")
        corners, ids, _ = aruco.detectMarkers(self.im,self.dct)
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
        self.loadMapImage(cv2.imread("market_tmap.png"))
        cv2.imshow("im0",self.im0)
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 800)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 600)
        while True:
            cv2.waitKey(10)
            ret, im = cap.read()
            cv2.imshow("im",im)
            hasH = self.tryGetHomography(im)
            if not hasH: continue
            im0 = self.im0.copy()
            x0,y0 = 400,400
            xy = self.warp_func(x0,y0)
            x,y = int(xy[0]),int(xy[1])
            cv2.circle(im, (x0, y0), 4, (0, 0, 255), -1)
            cv2.circle(im0, (x, y), 4, (0, 0, 255), -1)
            cv2.imshow("im",im)
            cv2.imshow("im0",im0)
            cv2.waitKey(100)


if __name__ == '__main__':
    HomographyCv().test1()

