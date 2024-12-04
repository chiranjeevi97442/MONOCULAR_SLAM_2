
import numpy as np
import cv2
#import matplotlib.pyplot as plt
import os

from skimage.measure import ransac
from skimage.transform import FundamentalMatrixTransform
from skimage.transform import EssentialMatrixTransform



class Frame(object):
    def __init__(self, mapp, img, K):
        self.K = K
        self.Kinv = np.linalg.inv(self.K)
        self.pose = IRt

        self.id = len(mapp.frames)
        mapp.frames.append(self)
        def add_ones(x):

            return np.concatenate([x, np.ones((x.shape[0], 1))], axis=1)

        def normalize(Kinv, pts):

            return np.dot(Kinv, add_ones(pts).T).T[:, 0:2]
            

        def extract(img):
            orb = cv2.ORB_create()
            
            # Convert to grayscale
            gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # Detection
            pts = cv2.goodFeaturesToTrack(gray_img, 8000, qualityLevel=0.01, minDistance=10)

            if pts is None:
                return np.array([]), None

            # Extraction
            kps = [cv2.KeyPoint(f[0][0], f[0][1], 20) for f in pts]
            kps, des = orb.compute(gray_img, kps)

            return np.array([(kp.pt[0], kp.pt[1]) for kp in kps]), des   

        pts, self.des = extract(img)

        
        if self.des.any()!=None:
            self.pts = normalize(self.Kinv, pts)


class Process_frame(object):
      def __init__(self,IMG_LIST):
          self.IMG_1=IMG_LIST[0]
          self.IMG_2=IMG_LIST[1]
          self.IMG_1_gray=cv2.cvtColor(self.IMG_1,cv2.COLOR_BGR2GRAY)
          self.IMG_2_gray=cv2.cvtColor(self.IMG_2,cv2.COLOR_BGR2GRAY)
          self.PROCESSED_IMG1=self.IMG_1_gray
          self.PROCESSED_IMG2=self.IMG_2_gray
          self.detect=cv2.ORB_create()
          self.relative_poses=[]
          self.output=None


      def feature_mapping(self,img):
        orb=cv2.ORB_create()
        
        pts=cv2.goodFeaturesToTrack(img,1000, qualityLevel=0.01, minDistance=7)
        key_pts=[cv2.KeyPoint(x=f[0][0], y=f[0][1], size=20) for f in pts]
        key_pts,descriptors=orb.compute(img,key_pts)
        
        return np.array([(kp.pt[0],kp.pt[1]) for kp in key_pts]),descriptors,orb    

      def PROCESS(self):
          key_pts_1,des_1,orb=self.feature_mapping(self.IMG_1_gray)
    
          key_pts_2,des_2,orb=self.feature_mapping(self.IMG_2_gray)

          def array_to_keypoints(points):
                return [cv2.KeyPoint(x=int(pt[0]), y=int(pt[1]), size=1) for pt in points]
    
    
          def convert_to_dmatches(matches):
                #print("matches",matches[0][0],matches[1][0])
                
                return [cv2.DMatch(_queryIdx=match[0], _trainIdx=match[1], _distance=0.0) for match in matches] 
    
    
          def extract_RT(F):
                """
                check this link:https://www-users.cse.umn.edu/~hspark/CSci5980/Lec12_CameraPoseEstimation.pdf
                if we use cv2.recoverpose the essential
                or fundamentqalmatrix along with the recover pose we can compute
                the r,t value internally  it also do the same as we did here

                Parameters
                ----------
                model.params : TYPE
                    DESCRIPTION.

                Returns
                -------
                R,T value based on the given fundamental matrix 

                """
                #chairalitty of pose recovery
                W=np.mat([[0,-1,0],[1,0,0],[0,0,1]],dtype=float)
                Z=np.array([[0,1,0],[-1,0,0],[0,0,0]])
                
                U,d,Vt=np.linalg.svd(F)
                if np.linalg.det(Vt)<0:
                    Vt*=-1.0
                    
                R=np.dot(np.dot(U,W),Vt)
                t=U[:,2]
                
                t1=(U@Z.T)@U.T
                
                ret=np.eye(4)
                ret[:3,:3]=R
                ret[:3,3]=t
                
                return ret
          
          def GOOD_MATCHES(KP_1,KP_2,des1,des2):
                bf=cv2.BFMatcher(cv2.NORM_HAMMING)
                matches=bf.knnMatch(des1,des2,k=2)
                GOOD=[]
                ret=[]
                x1,x2=[],[]
                #this is lowe's Ratio test
                for m,n in matches:
                    if m.distance < 0.75*n.distance:
                        GOOD.append([m])
                        pts1=key_pts_1[m.queryIdx]
                        pts2=key_pts_2[m.trainIdx]
                        # travel less than 10% of diagonal and within orb distance 32
                        if np.linalg.norm((pts1-pts2)) < 0.1*np.linalg.norm([1920,1080]) and m.distance<32:
                            if m.queryIdx not in x1 and m.trainIdx not in x2:
                                x1.append(m.queryIdx)
                                x2.append(m.trainIdx)
                                
                                ret.append((pts1,pts2))
                                
                assert(len(x1))==len(x1)
                assert(len(x2))==len(x2)

                assert len(ret)>=8
                ret=np.array(ret).astype(np.float32)
                x1=np.array(x1).astype(np.float32)
                x2=np.array(x2).astype(np.float32)

                return GOOD, ret[:,0],ret[:,1]
          
          def _form_transf(R, t):
                    """
                    Makes a transformation matrix from the given rotation matrix and translation vector
                
                    Parameters
                    ----------
                    R (ndarray): The rotation matrix
                    t (list): The translation vector
                
                    Returns
                    -------
                    T (ndarray): The transformation matrix
                    """
                    T = np.eye(4, dtype=np.float64)
                    
                    T[:3, :3] = R
                    T[:3, 3] = t
                    return T
          
          def FIND_RT(pts1,pts2):
            """
            computing the R,t by taking the matches and camera intrinsic parameters
            
            """
            
            
            RT_ACTUAL=np.eye(4)
            K=np.array([[500,0,960],[0,500,540],[0,0,1]])
            E,mask=cv2.findEssentialMat(pts1, pts2, K)
            RT=extract_RT(E)
            _,R,t, _= cv2.recoverPose(E, pts1, pts2, K)
            RT_ACTUAL =  _form_transf(R,np.squeeze(t))
            
            
                
            return RT_ACTUAL   
        
          def draw_MATCHES_MANUEL(img1,key_pts_1,img2,key_pts_2,matches):
                
                img1=cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
                img2=cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)
                
                IMAGE_WHOLE=np.concatenate((img1,img2),axis=1)
                New_key_pts_2=np.zeros([len(key_pts_2),2])
                New_key_pts_2[:,1]+=1920
                for i in range(len(matches)):
                    pt1=key_pts_1[i]
                    pt2=key_pts_1[i]+New_key_pts_2[i]
                    #print("pt1",pt1)
                    #print("pt2",pt2)
                    color=(0,0,255)
                    cv2.line(IMAGE_WHOLE, (int(pt1[1]),int(pt1[0])), (int(pt2[1]),int(pt2[0])), color)
                return IMAGE_WHOLE
          
          good_matches,X_1,X_2=GOOD_MATCHES(key_pts_1,key_pts_2,des_1, des_2)

          R_T=FIND_RT(X_1, X_2)

          RT_ACTUAL_INV = np.linalg.inv(R_T)
          self.relative_poses.append(RT_ACTUAL_INV)

          K_PT_1=array_to_keypoints(key_pts_1)
          K_PT_2=array_to_keypoints(key_pts_2)
          
          #output=cv2.drawMatches(I1, K_PT_1, I2, K_PT_2, good_matches[:10], None, flags=2)
          self.output=draw_MATCHES_MANUEL(self.IMG_1_gray,key_pts_1,self.IMG_2_gray,key_pts_2,good_matches[:10])
            
          self.PROCESSED_IMG1=cv2.drawKeypoints(self.IMG_1_gray,K_PT_1, self.IMG_1_gray)
          self.PROCESSED_IMG2=cv2.drawKeypoints(self.IMG_2_gray,K_PT_2, self.IMG_2_gray)

          return self.PROCESSED_IMG1,self.PROCESSED_IMG2,self.output


def main():
     IMG_LIST=[]
     CAP=cv2.VideoCapture(r"test_countryroad.mp4")

     while True:
          ret,frame=CAP.read()
          if ret:
               IMG_LIST.append(frame)
               if len(IMG_LIST)==2:
                    PF=Process_frame(IMG_LIST)
                    F1,F2,output=PF.PROCESS()
               else:
                    continue 
               IMG_LIST.pop(0)
               cv2.imshow("keypoints_image_1",cv2.resize(F1, (960,540)))
               cv2.imshow("keypoints_image_2",cv2.resize(F2,(960,540)))
               cv2.imshow("outputimages", cv2.resize(output,(1920,1080)))
                #cv2.imshow("DISPLAYING_IMAGE",frame)
                
               if cv2.waitKey(1)==ord('q'):
                    break
     CAP.release()
     cv2.destroyAllWindows()   

if __name__=="__main__":
         F= int(os.getenv("F","500"))
         W, H = 1920//2, 1080//2
         K = np.array([[F,0,W//2],[0,F,H//2],[0,0,1]])
         IRt = np.eye(4)


         main()            

 

