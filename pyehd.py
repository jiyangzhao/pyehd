# python ehd module
import numpy as np


# function to convert RBG image to grayscale image
def rgb2gray(rgb):
    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    return gray

# function to get EHD vector
def findehd(img):

    r,c,m = np.shape(img) # get the shape of image
    if m==3:
        img = rgb2gray(img) # convert RGB img to grayscale img
    M = 4*np.ceil(r/4) 
    N = 4*np.ceil(c/4)
    img = np.reshape(img,(int(M),int(N))) # Making image dim. divisible completely by 4
    AllBins = np.zeros((17, 5)) # initializing Bins
    p = 1
    L = 0
    for _ in range(4):
        K = 0
        for _ in range(4):
            block = img[K:K+int(M/4), L:L+int(N/4)] # Extracting (M/4,N/4) block
            AllBins[p,:] = getbins(np.double(block)) 
            K = K + int(M/4)
            p = p + 1
        L = L + int(N/4)
    GlobalBin = np.mean(AllBins) # getting global Bin
    AllBins[16,:]= np.round(GlobalBin)
    ehd = np.reshape(np.transpose(AllBins),[1,85])
    ehd = ehd[0,-5:]
    return ehd


# function for getting Bin values for each block
def getbins(imgb):
    M,N = imgb.shape
    M = 2*np.ceil(M/2)
    N = 2*np.ceil(N/2)
    # print(M)
    # print(N)
    imgb = np.reshape(imgb,(int(M),int(N))) # Making block dimension divisible by 2
    bins = np.zeros((1,5)) # initialize Bin
    """Operations, define constant"""
    V = np.array([[1,-1],[1,-1]]) # vertical edge operator
    H = np.array([[1,1],[-1,-1]]) # horizontal edge operator
    D45 = np.array([[1.414,0],[0,-1.414]])# diagonal 45 edge operator
    D135 = np.array([[0,1.414],[-1.414,0]]) # diagonal 135 edge operator
    Isot = np.array([[2,-2],[-2,2]]) # isotropic edge operator
    T = 50 # threshold
    
    nobr = int(M/2) # loop limits
    nobc = int(N/2) # loop limits
    L = 0

    """loops of operating"""
    for _ in range(nobc):
        K = 0
        for _ in range(nobr):
            block = imgb[K:K+2, L:L+2] # Extracting 2x2 block
            pv = np.abs(np.sum(np.sum(block*V))) # apply operators
            ph = np.abs(np.sum(np.sum(block*H)))
            pd45 = np.abs(np.sum(np.sum(block*D45)))
            pd135 = np.abs(np.sum(np.sum(block*D135)))
            pisot = np.abs(np.sum(np.sum(block*Isot)))
            parray = [pv,ph,pd45,pd135,pisot]
            index = np.argmax(parray) # get the index of max value
            value = parray[index] # get the max value
            # print('value: '+str(value))
            if value >= T:
                bins[0,index]=bins[0,index]+1 # update bins values
            K = K+2
        L = L+2
    
    # print('bins is:')
    # print(bins)
    return bins
