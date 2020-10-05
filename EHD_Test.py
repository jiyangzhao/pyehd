# Program to compare two images on the basis of Edge Histogram Descriptor (EHD)
import tkinter as tk
from tkinter import filedialog
import os
import matplotlib.pyplot as plt
import numpy as np
import pyehd as ehd
from PIL import Image

# Get file name for img1
root = tk.Tk()
root.withdraw()
file_path1 = filedialog.askopenfilename()
file_name1 = os.path.basename(file_path1)
# Reading Image 1
img1 = plt.imread(file_path1)
# Get file name for img2
root = tk.Tk()
root.withdraw()
file_path2 = filedialog.askopenfilename()
file_name2 = os.path.basename(file_path2)
img2 = plt.imread(file_path2)
# Function findehd() to get EHD vector

# Finding EHD1
ehd1 = ehd.findehd(img1)
# Finding EHD2
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