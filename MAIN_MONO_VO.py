# -*- coding: utf-8 -*-
"""
Created on Tue Oct 29 07:51:35 2024

@author: chiranjeevi

"""

import numpy as np
import cv2
#import matplotlib.pyplot as plt
import os


def BF_FeatureMatcher(des1,des2):
    brute_force = cv2.BFMatcher(cv2.NORM_HAMMING,crossCheck=True)
    no_of_matches = brute_force.match(des1,des2)
    #no_of_matches = brute_force.knnMatch(des1,des2,k=2)
 
    # finding the humming distance of the matches and sorting them
    no_of_matches = sorted(no_of_matches,key=lambda x:x.distance)
    return no_of_matches


    
def feature_mapping(img):
    orb=cv2.ORB_create()
    
    pts=cv2.goodFeaturesToTrack(img,1000, qualityLevel=0.01, minDistance=7)
    key_pts=[cv2.KeyPoint(x=f[0][0], y=f[0][1], size=20) for f in pts]
    key_pts,descriptors=orb.compute(img,key_pts)
    
    return np.array([(kp.pt[0],kp.pt[1]) for kp in key_pts]),descriptors,orb
            
     

def process_frames(img_list):
    
     ##ORB_descreptors_and_Feature_matchinig
    detect=cv2.ORB_create()
    #print("lenght of image list",len(img_list))
    
    
    I1 = cv2.cvtColor(img_list[0], cv2.COLOR_BGR2GRAY)
    I2 = cv2.cvtColor(img_list[1], cv2.COLOR_BGR2GRAY)
    
    key_pt_1,desc_1=detect.detectAndCompute(I1,None)
    key_pt_2,desc_2=detect.detectAndCompute(I2,None)
    
    
    
    good_matches=BF_FeatureMatcher(desc_1, desc_2)
    #print("good_matches",good_matches[:10])
    tot_good_matches=len(good_matches)
    print("key_pt_1",key_pt_1)
    out_put_image=cv2.drawMatches(I1, key_pt_1, I2, 
            key_pt_2, good_matches[:10], None, flags=2)
    #print("length of key_pt_1 and key_pt_2",len(key_pt_1),len(key_pt_2))     
    I1=cv2.drawKeypoints(I1,key_pt_1, I1)
    I2=cv2.drawKeypoints(I2,key_pt_2, I2)
    #print("total number of matches",tot_good_matches)
    
    return I1,I2,out_put_image

def process_frames_1(img_list):
   
    I1 = cv2.cvtColor(img_list[0], cv2.COLOR_BGR2GRAY)
    I2 = cv2.cvtColor(img_list[1], cv2.COLOR_BGR2GRAY)
    key_pts_1,des_1,orb=feature_mapping(I1)
    
    key_pts_2,des_2,orb=feature_mapping(I2)
    #print(len(key_pts_1),len(key_pts_2))
    #print(len(des_1),len(des_2))
    
    
    def GOOD_MATCHES(des1,des2):
        bf=cv2.BFMatcher(cv2.NORM_HAMMING)
        matches=bf.knnMatch(des1,des2,k=2)
        GOOD=[]
        for m,n in matches:
            if m.distance < 0.75*n.distance:
                GOOD.append([m])
                                
        return GOOD
    
    good_matches=GOOD_MATCHES(des_1, des_2)
    #print("good_matches",len(good_matches))
    #print("lenght of key_pts_1",len(key_pts_1))
    #print("lenght of key_pts_2",len(key_pts_2))
    print("key_pt_1",key_pts_1)
    output=cv2.drawMatches(I1, key_pts_1, I2, key_pts_2, good_matches[:10], None, flags=2)
    
    I1=cv2.drawKeypoints(I1,key_pts_1, I1)
    I2=cv2.drawKeypoints(I2,key_pts_2, I2)
    
    
    return I1,I2,output
    




def main():
    IMG_LIST=[]
    CAP=cv2.VideoCapture(r"test_countryroad.mp4")
    
    while True:
        ret,frame=CAP.read()
        if ret:
            IMG_LIST.append(frame)
            if len(IMG_LIST)==2:
                F1,F2,output=process_frames_1(IMG_LIST)
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
    main()
    



