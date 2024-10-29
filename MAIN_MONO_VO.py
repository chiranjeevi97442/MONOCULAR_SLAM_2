# -*- coding: utf-8 -*-
"""
Created on Tue Oct 29 07:51:35 2024

@author: chiranjeevi
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt

def main():
    IMG_LIST=[]
    CAP=cv2.VideoCapture("SPECIFY_THE_PATH")
    while True:
        ret,frame=CAP.read()
        if ret:
            IMG_LIST.append(frame)
            cv2.imshow("DISPLAYING_IMAGE",frame)
            if cv2.waitKey(1)==ord('q'):
                break
    CAP.release()
    cv2.destroyAllWindows()       
      



if __name__=="__main__":
    main()
    



