"""Prepare the centroids"""
import os
import mmcv
import numpy as np
from pathlib import Path
from matplotlib import pyplot as plt
try:
    import xml.etree.cElementTree as ET
except ImportError:
    import xml.etree.ElementTree as ET

curr_dir = Path(__file__).parent.absolute()
idx_file = os.path.expanduser(os.path.join(curr_dir, 'trainvaltest.txt'))
img_dir = os.path.expanduser(os.path.join(curr_dir, 'imgs'))
centroids_dir = os.path.expanduser(os.path.join(curr_dir, 'centroids'))    
cen_vis_val_dir = os.path.expanduser(os.path.join(curr_dir, 'cen_vis_val'))
if not os.path.exists(cen_vis_val_dir):
    os.makedirs(cen_vis_val_dir)

def main():    

    means = []
    stds = []
    # load image and mask paths
    with open(idx_file, "r") as lines:
        # print("lines:", lines)
        for line in lines:
            idx = line.rstrip('\n')
            _image = os.path.join(img_dir, idx + ".png")
            img = mmcv.imread(_image, flag='grayscale').astype(np.double)
            means.append(img.mean())
            stds.append(img.std())
            # break   
             
    # print(means)
    print("mean: ", np.array(means).mean())
    print("std: ", np.array(stds).mean())

if __name__ == '__main__':
    main()
