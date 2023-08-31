import numpy as np
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