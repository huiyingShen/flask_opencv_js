import numpy as np
import cv2
import cv2.aruco as aruco

class Homography:
    def loadImage(self):
        im0 = cv2.imread("market_tmap.png")
        self.corners, self.ids, _ = aruco.detectMarkers(im0,aruco.Dictionary_get(cv2.aruco.DICT_4X4_250))
        return self
    
    def findId(self,id):
        for i in range(len(self.ids)):
            if id[0] == self.ids[i][0]:
                return i
        return -1
    
    def getCorners(self):
        im = cv2.imread("marker_test.jpg")
        corners, ids, _ = aruco.detectMarkers(im,aruco.Dictionary_get(cv2.aruco.DICT_4X4_250))
        return corners, ids
    
    def getCornerPairs(self,corners, ids):
        if ids is None: return
        vCn1,vCn2 = [],[]
        for i,id in enumerate(ids):
            indx = self.findId(id)
            if indx != -1:
                # print(self.ids[indx][0])
                vCn1.extend(corners[i][0])
                vCn2.extend(self.corners[indx][0])
        return vCn1,vCn2

    def verify(self,vCn1,vCn2):
         for i in range(len(vCn1)):
            x, y = vCn1[i]
            u1,v1 = self.warp_func(x,y)
            u, v = vCn2[i]
            print(f'{u:.1f} {v:.1f} {u1:.2f} {v1:.2f} ') 

    def warp_func(self,x,y):
        xy_homogeneous = np.array([x, y, 1])
        uvw = np.dot(self.h, xy_homogeneous)
        return [uvw[0] / uvw[2], uvw[1] / uvw[2]]
    
    def getHomographyCv(self,vCn1,vCn2):
        self.h, status = cv2.findHomography(np.array(vCn1),np.array(vCn2))
        print("getHomographyCv(): \n",self.h)
        self.verify(vCn1,vCn2)

    def getHomography(self,vCn1,vCn2):
        # Check that there are at least 4 corresponding points
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
        print("getHomography(): \n",self.h)
        self.verify(vCn1,vCn2)


if __name__ == '__main__':
    hm = Homography().loadImage()
    corners,ids = hm.getCorners()
    vCn1,vCn2 = hm.getCornerPairs(corners,ids)
    hm.getHomographyCv(vCn1,vCn2)
    hm.getHomography(vCn1,vCn2)
