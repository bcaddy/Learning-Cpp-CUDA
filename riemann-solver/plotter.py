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
# import donemusic

plt.close('all')
start = default_timer()

def main():
    # Load file
    global file, size
    file = np.loadtxt("results.csv", delimiter=",", usecols=range(1000))
    
    # sim info
    length = 1.
    size = len(file[0,:])
    positions = np.linspace(0.,length,size)

    for i in range(0, len(file[:,0])):
        plt.figure(1)
        plt.plot(positions,file[i,:])
    
        plt.xlabel("Position")
        plt.ylabel("Value of a")
        plt.title("Solution to top hat")
        plt.tight_layout()
        
        plt.savefig(f'images/{i}.png',
                    bbox='tight',
                    dpi=150)






main()
print(f'\nTime to execute: {round(default_timer()-start,2)} seconds')
# donemusic.nonstop() #other option availableto the file? ~/')

