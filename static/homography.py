import base64
import numpy as np
import cv2
import cv2.aruco as aruco

def data_uri_to_cv2_img(data_uri):
    data_uri += "=" * ((4 - len(data_uri) % 4) % 4) # Base64 needs a string with length multiple of 4. If the string is short, it is padded with 1 to 3 =.
    # print("data_uri[:100] = ",data_uri[:100])
    nparr = np.frombuffer(base64.b64decode(data_uri), np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    print(img.shape)
    return img

class Homography:
    def __init__(self):
        self.ids = []
        self.corners = []
        
    def findId(self,id):
        for i in range(len(self.ids)):
            if id[0] == self.ids[i][0]:
                return i
        return -1
    
    def getCornerPairs(self,corners, ids):
        if ids is None: return
        vCn1,vCn2 = [],[]
        for i,id in enumerate(ids):
            indx = self.findId(id)
            if indx != -1:
                vCn1.extend(corners[i][0])
                vCn2.extend(self.corners[indx][0])
        return vCn1,vCn2
        
    def getHomography(self,vCn1,vCn2):
        assert len(vCn1) == len(vCn2) and len(vCn1) >= 4
        A = []
        for i in range(len(vCn1)):
            x, y = vCn1[i]
            u, v = vCn2[i]
            A.extend([
                [-x, -y, -1, 0, 0, 0, u*x, u*y, u],
                [0, 0, 0, -x, -y, -1, v*x, v*y, v]
            ])
        A = np.array(A)
        
        _, _, V = np.linalg.svd(A)
        self.h = V[-1].reshape(3, 3)
        # print("getHomography(): \n",self.h)

    def verify(self,vCn1,vCn2):
         for i in range(len(vCn1)):
            x, y = vCn1[i]
            u1,v1 = self.warp_func(x,y)
            u, v = vCn2[i]
            print(f'{u:5.1f} {v:5.1f} {u1:5.1f} {v1:5.1f} ') 

    def warp_func(self,x,y):
        xy_homogeneous = np.array([x, y, 1])
        uvw = np.dot(self.h, xy_homogeneous)
        return [uvw[0] / uvw[2], uvw[1] / uvw[2]]

class HomographyCv(Homography):
    def __init__(self):
        super().__init__()
        self.im0 = None
        self.im = None
        #print(dir(aruco))
        self.detector = aruco.ArucoDetector(aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_250))

    def loadMapImage(self,im0):
        self.im0 = im0
        self.corners, self.ids, _ = self.detector.detectMarkers(self.im0)


    def loadMapDataUri(self,dataURL):
        self.loadMapImage(data_uri_to_cv2_img(dataURL))


    def tryGetHomography(self,im):
        self.im = im
        corners, ids, _ = self.detector.detectMarkers(self.im)
        if ids is None or len(ids) < 3: return False
        vCn1,vCn2 = self.getCornerPairs(corners,ids)
        self.drawMarker(self.im,vCn1)
        if len(vCn1) < 12: return False
        self.getHomography(vCn1,vCn2)
        return True
    
    def tryGetHomographyDataUri(self,testDataURL):
        return self.tryGetHomography(data_uri_to_cv2_img(testDataURL))


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

