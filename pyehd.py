# python ehd module
import numpy as np
from PIL import Image

def findehd(img):
    r,c,m = np.shape(img)
    if m==3:
        img = Image.open(img).convert('LA') # convert RGB img to grayscale img
    M = 4*np.ceil(r/4) 
    N = 4*np.ceil(c/4)
    img = Image.fromarray(img)
    img.resize(size=(M, N))
    AllBins = np.zeros((17, 5))
    p = 1
    L = 0
    for i in range(1,4,1):
        K = 0
        for j in range(1,4,1):
            block = img[K+1:K+M/4, L+1:L+N/4]
            AllBins[p,:] = getbins(double(block))
            K = K + (M/4)
            p = p + 1
        L = L + (N/4)
    GlobalBin = np.mean(AllBins)
    AllBins[17,:]= np.round(GlobalBin)
    ehd = np.reshape(np.transpose(AllBins),[1,85])
    return ehd


# function for getting Bin values for each block
def getbins(imgb):
    M,N = imgb.shape
    M = 2*np.ceil(M/2)
    N = 2*np.ceil(N/2)
    imgb = np.reshape(imgb,M,N) # Making block dimension divisible by 2
    bins = np.zeros(1,5) # initialize Bin
    """Operations, define constant"""
    V = np.array([[1,-1],[1,-1]]) # vertical edge operator
    H = np.array([[1,1],[-1,-1]]) # horizontal edge operator
    D45 = np.array([[1.414,0],[0,-1.414]])# diagonal 45 edge operator
    D135 = np.array([0,1.414],[-1.414,0]) # diagonal 135 edge operator
    Isot = np.array([2,-2],[-2,2]) # isotropic edge operator
    T = 50 # threshold
    
    nobr = M/2 # loop limits
    nobc = N/2 # loop limits
    L = 0

    """loops of operating"""
    for i in range(nobc):
        K = 0
        for j in range(nobr):
            block = imgb[K:K+1, L:L+1] # Extracting 2x2 block
            pv = np.abs(np.sum(np.sum(block*V))) # apply operators
            ph = np.abs(np.sum(np.sum(block*H)))
            pd45 = np.abs(np.sum(np.sum(block*D45)))
            pd135 = np.abs(np.sum(np.sum(block*D135)))
            pisot = np.abs(np.sum(np.sum(block*Isot)))
            parray = [pv,ph,pd45,pd135,pisot]
            index = np.argmax(parray) 
            value = parray[index]
            if value >= T:
                bins[index]=bins[index]+1 # update bins values
            K = K+2
        L = L+2
    return bins
