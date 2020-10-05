import os
import random
from PIL import Image
from numpy import asarray
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import pyehd as ehd

def plotimg(image,label,index):
    labelsList = ['Other','Flemish Stretch Bond','English Bond','Stretcher Bond','Other Brick Patterns']
    plt.imshow(image)
    plt.title('No. '+str(index)+' '+str(labelsList[label]))
    plt.show()

with open('Images_ehd.npy','rb') as fread:
    Images = np.load(fread)
    fread.close()

with open('Labels_ehd.npy','rb') as f:
    Labels = np.load(f)
    f.close()


print(Images.shape)
print(Labels.shape)

numList = range(0,len(Labels))

# for i in random.sample(numList,1):
#     plotimg(Images[i],Labels[i],i)

img1 = Images[1]
img2 = Images[2]

ehd1 = ehd.findehd(img1)
ehd2 = ehd.findehd(img2)

fig, axs = plt.subplots(nrows=2, ncols=2)
axs[0,0].imshow(img1)
plt.title('Image 1')
axs[0,1].bar([1,2,3,4,5], axs[80:85])
axs.set_title('Global Bin of Image 1')
axs[1,0].imshow(img2)
plt.title('Image 2')
axs[1,1].bar([1,2,3,4,5], axs[80:85])
axs.set_title('Global Bin of Image 2')

plt.figure(2)
ehd_plot1, = plt.plot(ehd1,color='r')
ehd_plot2, = plt.plot(ehd2,color='b')
plt.title('Comparing EHD1 and EHD2')
plt.legend([ehd_plot1, ehd_plot2],["EHD1", "EHD2"])

# L2 Distance between EHD1 and EHD2
D2 = np.sqrt(np.sum((ehd1-ehd2)**2))
np.disp('L2 Distance = %1.2f' % D2)
# L1 Distance between EHD1 and EHD2
D1 = np.sum(np.abs(ehd1-ehd2))
np.disp('L1 Distance = %1.2f' % D1)