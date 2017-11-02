'''
480x288
360x288
1920x1080
960x1080
720x540
'''

import cv2
import sys

import numpy as np


CROP = 0.3333

def init(fname):
    global rea, wri, CROP
    rea = cv2.VideoCapture(fname)
    CROP = int(rea.get(cv2.CAP_PROP_FRAME_HEIGHT)*CROP)
    fourcc = cv2.VideoWriter_fourcc('H','2','6','4') #int(rea.get(cv2.CAP_PROP_FOURCC))#cv2.VideoWriter_fourcc('X','2','6','4') #int(rea.get(cv2.CAP_PROP_FOURCC))d v
    fps = rea.get(cv2.CAP_PROP_FPS)
    framesize = (int(rea.get(cv2.CAP_PROP_FRAME_WIDTH)-CROP*2), int(rea.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    print('aaaaaaaa')
    sys.stdout.flush()
    wri = cv2.VideoWriter('crop.mp4',fourcc,fps,framesize)
    print('bbbbbb')
    sys.stdout.flush()

def main(fname):
    global rea, wri
    init(fname)
    count = rea.get(cv2.CAP_PROP_FRAME_COUNT)
    while(1):
        p = rea.get(cv2.CAP_PROP_POS_FRAMES)
        print("%d / %d, %.2f%%"%(p,count,100*p/count))
        q = rea.read()
        if q[0]:
           tmp = q[1][:,CROP:-CROP,:]
           #tmp = cv2.resize(tmp, (720,540), interpolation=cv2.INTER_AREA)
           wri.write(tmp)
           #cv2.imshow('a',tmp)
           #cv2.waitKey(1)
        else:
           break
    rea.release()
    wri.release()

def combine(fname1, fname2):
    rea1 = cv2.VideoCapture(fname1)
    rea2 = cv2.VideoCapture(fname2)
    fourcc = int(rea1.get(cv2.CAP_PROP_FOURCC))
    fps = rea1.get(cv2.CAP_PROP_FPS)
    width = int(max(rea1.get(cv2.CAP_PROP_FRAME_WIDTH), rea2.get(cv2.CAP_PROP_FRAME_WIDTH)))
    h1 = int(rea1.get(cv2.CAP_PROP_FRAME_HEIGHT))
    h2 = int(rea2.get(cv2.CAP_PROP_FRAME_HEIGHT))
    height = int(rea1.get(cv2.CAP_PROP_FRAME_HEIGHT) + rea2.get(cv2.CAP_PROP_FRAME_HEIGHT))
    framesize = (width, height)
    wri = cv2.VideoWriter('combine.mp4',fourcc,fps,framesize)
    pic = np.zeros((height, width, 3), 'uint8')
    count = rea1.get(cv2.CAP_PROP_FRAME_COUNT)
    while(1):
        p = rea1.get(cv2.CAP_PROP_POS_FRAMES)
        print("%d / %d, %.2f%%"%(p,count,100*p/count))
        q1 = rea1.read()
        q2 = rea2.read()
        if q1[0] and q2[0]:
            pic[:h1,:,:] = q1[1]
            pic[h1:,:,:] = q2[1]
            wri.write(pic)
        else:
            break

if __name__ == '__main__':
    if(len(sys.argv) < 2):
        print("usage: %s [video.mp4]"%sys.argv[0])
    elif len(sys.argv) == 2:
        main(sys.argv[1])
    elif len(sys.argv) == 3:
        combine(sys.argv[1], sys.argv[2])

