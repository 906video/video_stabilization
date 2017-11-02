import sys
from matplotlib import pyplot as plt
import numpy as np
import cv2

def calc(fname):
    global cur,pre,rea
    fig = plt.figure(1)
    ax = []
    num = len(fname)
    data = []
    res = []
    for i in range(num):
        rea = cv2.VideoCapture(fname[i])
        pre = rea.read()[1]
        data = []
        count = rea.get(cv2.CAP_PROP_FRAME_COUNT)
        while(1):
            cur = rea.read()
            if not cur[0]:
                break
            p = rea.get(cv2.CAP_PROP_POS_FRAMES)
            print("%d / %d, %.2f%%"%(p,count,100*p/count))

            cur = cur[1]
            data.append((cur-pre).var())
            pre = cur

        X = [no for no,i in enumerate(data)]
        ax = plt.subplot(100*num+11+i)
        ax.plot(X,data)
        res.append(10*np.log10(255*255/np.mean(data)))
    for nn, rr in zip(fname,res):
        print("%s:%f:%f"%(nn,rr,rr/count))
        
    plt.show()



def read(fname):
    NO = []
    X = []
    Y = []
    A = []
    DIFF = []
    pre = 0
    with open(fname,'r') as f:
        while(1):
            string = f.readline()
            if string == '':
                break
            tmp = string.split(' ')
            #print(tmp)
            NO.append(int(tmp[0]))
            X.append(float(tmp[1]))
            Y.append(float(tmp[2]))
            A.append(float(tmp[3])*200)
            cur = X[-1]+Y[-1]+A[-1]
            DIFF.append(abs(pre-cur))
            pre = cur
    return (NO, X, Y, A, DIFF)


def test(fname):
    fig = plt.figure(1)
    ax = []
    num = len(fname)
    data = []
    for i in range(num):
        data = read(fname[i])
        ax = plt.subplot(100*num+11+i)
        ax.plot(data[0],data[4]) #diff
        #ax.plot(data[0], data[1]) #x
        #ax.plot(data[0], data[2]) #Y
        #ax.plot(data[0], data[3]) #A
        print(i, sum(data[4])/len(data[4]))
    plt.show()

if __name__ == '__main__':
   test(sys.argv[1:])
   #calc(sys.argv[1:])
