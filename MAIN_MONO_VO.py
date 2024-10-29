# -*- coding: utf-8 -*-
"""
Created on Tue Oct 29 07:51:35 2024

@author: chiranjeevi

"""

import numpy as np
import cv2
import matplotlib.pyplot as plt




def process_frames(img_list):
    
    I1 = cv2.cvtColor(img_list[0], cv2.COLOR_BGR2GRAY)
    I2 = cv2.cvtColor(img_list[1], cv2.COLOR_BGR2GRAY)






def main():
    IMG_LIST=[]
    CAP=cv2.VideoCapture("SPECIFY_THE_PATH")
    
    while True:
        ret,frame=CAP.read()
        if ret:
            IMG_LIST.append(frame)
            if len(IMG_LIST)==2:
                process_frames(IMG_LIST)
            else:
                continue
            IMG_LIST.pop(0)
            cv2.imshow("DISPLAYING_IMAGE",frame)
            
            if cv2.waitKey(1)==ord('q'):
                break
    CAP.release()
    cv2.destroyAllWindows()       
      



if __name__=="__main__":
    main()
    



