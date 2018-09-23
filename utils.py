
import matplotlib.pyplot as plt
from glob import glob
import pydicom
import numpy as np
import os, sys

OVPATH = '/home/ubuntu/datasets/chest/*/*/'
DPATH = '/home/ubuntu/datasets/chest/CBIS-DDSM/*'

def plot_raw(name, zoom=None):
    casefolders = glob(DPATH)
    assert len(casefolders) > 0
    matching = [fl for fl in casefolders if name in fl]
    root = sorted(matching, key=lambda ent: len(ent))[0]
    maskpaths = sorted(matching, key=lambda ent: len(ent))[1:]
    impath = glob(glob(glob(root+'/*')[0]+'/*')[0]+'/*')[0]
    # print(impath)
    plt.figure(figsize=(14, 14))
    with pydicom.dcmread(impath) as db:
        rawimg = db.pixel_array
    # print(impath)
    mask = np.zeros(rawimg.shape)
    # ddsm = OVPATH + 'case%s' % name.split('_')[0][1:]
    # print(ddsm)
    # print(glob(ddsm))
    # return 

    for pth in maskpaths:
        pth = glob(glob(glob(pth+'/*')[0]+'/*')[0]+'/*')[1]
        print(pth)
        with pydicom.dcmread(pth) as db:
            rawmask = db.pixel_array
            try:
                assert mask.shape == rawmask.shape
            except:
                print(mask.shape, rawmask.shape)
                assert False
            mask[rawmask > 0] = 1
    plt.subplot(1, 2, 1)
    if zoom is not None:
        xx, yy = zoom
        plt.imshow(rawimg[yy:yy+1000, xx:xx+1000])
    else:
        plt.imshow(rawimg)
    plt.subplot(1, 2, 2)
    if zoom is not None:
        xx, yy = zoom
        plt.imshow(rawimg[yy:yy+1000, xx:xx+1000], cmap='gray')
        plt.imshow(mask[yy:yy+1000, xx:xx+1000], alpha=0.5)
    else:
        plt.imshow(rawimg, cmap='gray')
        plt.imshow(mask, alpha=0.5)
    plt.show()
    plt.close()