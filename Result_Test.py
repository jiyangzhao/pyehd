import os
import random
from PIL import Image
from numpy import asarray
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import pyehd as ehd

def plotimg(image,label,index):
    # labelsList = ['Other','Flemish Stretch Bond','English Bond','Stretcher Bond','Other Brick Patterns']
    plt.imshow(image)
    plt.title('No. '+str(index)+' '+str(labelsList[label]))
    plt.show()

labelsList = ['Other','Flemish Stretch Bond','English Bond','Stretcher Bond','Other Brick Patterns']

with open('Images_ehd.npy','rb') as fread:
    Images = np.load(fread)
    fread.close()

with open('Labels_ehd.npy','rb') as fread:
    Labels = np.load(fread)
    fread.close()

imgs = Images

numList = range(0,len(Labels))

for i in random.sample(numList,2):
    # plotimg(Images[i],Labels[i],i)
    ehd_current = ehd.findehd(Images[i])
    print('Image type: ' + str(labelsList[Labels[i]]))
    print("ehd is: " + str(ehd_current))