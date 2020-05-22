#!/usr/bin/env python3
"""
================================================================================
 Written by Robert Caddy.  Created on Fri May 22 14:49:13 2020

 plot and animate results from Advection/Riemann solver

 Dependencies:
     numpy
     timeit
     donemusic
     matplotlib

 Changelog:
     Version 1.0 - First Version
================================================================================
"""

import numpy as np
import matplotlib.pyplot as plt
from timeit import default_timer
import socket
import os
# import donemusic

plt.close('all')
start = default_timer()

def main():
    # Load file
    file = np.loadtxt("results.csv", delimiter=",", usecols=range(1000))
    
    # sim info
    length = 1.
    size = len(file[0,:])
    positions = np.linspace(0.,length,size)
    
    # Plotting info
    numFrames = 100
    stride = len(file[:,0])//numFrames
    
    # Find mins and maxes
    pad = np.max(np.abs([file.min(), file.max()])) * 0.05
    small = file.min() - pad
    large = file.max() + pad
    
    

    for i in range(0, len(file[:,0]), stride):
        plt.figure(1)
        plt.plot(positions,file[i,:],
                 linestyle='-',
                 color='blue')
        
        plt.ylim(small, large)
    
        plt.xlabel("Position")
        plt.ylabel("Value of a")
        plt.title("Solution to top hat")
        plt.tight_layout()
        
        plt.savefig(f'images/{i}.png',
                    bbox='tight',
                    dpi=150)
        plt.close('all')

    if (socket.gethostname()[7:] == "crc.pitt.edu" ):
        os.system("convert images/*.png animated.gif")

main()
print(f'\nTime to execute: {round(default_timer()-start,2)} seconds')
# donemusic.nonstop() #other option availableto the file? ~/')

