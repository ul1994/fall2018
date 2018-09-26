
import matplotlib.pyplot as plt
from glob import glob
import pydicom
import numpy as np
import os, sys
from random import randint, shuffle
from scipy.ndimage import gaussian_filter as blur
import cv2
import json

CONTPATH = '/home/ubuntu/datasets/contours/'
OVPATH = '/home/ubuntu/datasets/chest/*/*/'
DPATH = '/home/ubuntu/datasets/chest/CBIS-DDSM/*'

def plot_raw(name, zoom=None, noplot=False):
    with open('.cbis_cache.json') as db:
        refs = json.load(db)

    ref = [ent for ent in refs if name == ent['sid']]
    assert len(ref) == 1
    ref = ref[0]

    casefolders = glob(DPATH)
    assert len(casefolders) > 0
    matching = [fl for fl in casefolders if name in fl]
    root = sorted(matching, key=lambda ent: len(ent))[0]
    impath = glob(glob(glob(root+'/*')[0]+'/*')[0]+'/*')[0]
    
    with pydicom.dcmread(impath) as db:
        rawimg = db.pixel_array
    mask = np.zeros(rawimg.shape)

    if not noplot: plt.figure(figsize=(14, 14))
    for pth in ref['masks']:
        with pydicom.dcmread(pth) as db:
            rawmask = db.pixel_array
            try:
                assert mask.shape == rawmask.shape
            except:
                hscale = mask.shape[0] / rawmask.shape[0]
                rawmask = cv2.resize(rawmask, (0,0), fx=hscale, fy=hscale)
                wscale = mask.shape[1] / rawmask.shape[1]
                rawmask = cv2.resize(rawmask, (0,0), fx=wscale, fy=1)
                assert mask.shape == rawmask.shape
            mask[rawmask > 0] = 1
    if not noplot: plt.subplot(1, 2, 1)
    if zoom is not None:
        xx, yy = zoom
        if not noplot: plt.imshow(rawimg[yy:yy+1000, xx:xx+1000])
    else:
        if not noplot: plt.imshow(rawimg)
    if not noplot: plt.subplot(1, 2, 2)
    if zoom is not None:
        xx, yy = zoom
        if not noplot: plt.imshow(rawimg[yy:yy+1000, xx:xx+1000], cmap='gray')
        if not noplot: plt.imshow(mask[yy:yy+1000, xx:xx+1000], alpha=0.5)
    else:
        if not noplot: plt.imshow(rawimg, cmap='gray')
        if not noplot: plt.imshow(mask, alpha=0.5)
    if not noplot: plt.show()
    if not noplot: plt.close()

    return rawimg, mask

def spread_contour_batch(batch, maxplots=5):
    imgs, noise, masks, refs = batch

    plt.figure(figsize=(14, 14))
    for ii in range(maxplots):
        plt.subplot(3, maxplots, ii+1)
        plt.imshow(imgs[ii, :, :, 0])

        plt.subplot(3, maxplots, maxplots+ii+1)
        plt.imshow(imgs[ii, :, :, 0])
        plt.imshow(masks[ii, :, :, 0], alpha=0.5)

        plt.subplot(3, maxplots, 2*maxplots+ii+1)
        plt.imshow(imgs[ii, :, :, 0])
        plt.imshow(noise[ii, :, :, 0], alpha=0.5)
    
    plt.show()
    plt.close()

def spread_contours(name, maxplots=7):
    hasmask = ['centered']
    labels = ['centered', 'healthy']
    for lbl in labels:
        found = glob('%s%s/%s*.npy' % (CONTPATH, lbl, name))
        found = sorted(found, key=lambda fname: fname.split('_')[-1])
        assert len(found) > 0

        if lbl in hasmask:
            found_masks = glob('%s%s/%s*.jpg' % (CONTPATH, lbl, name))
            found_masks = sorted(found_masks, key=lambda fname: fname.split('_')[-1])
            assert len(found_masks) > 0
        
        nplots = min(maxplots, len(found))
        plt.figure(figsize=(14, 7))
        for ii, fname in enumerate(found[:maxplots]):
            mat = np.load(fname)
            plt.subplot(1, nplots, ii+1)
            plt.gca().set_title(lbl)
            plt.imshow(mat)
            if lbl in hasmask:
                maskim = cv2.imread(found_masks[ii], 0)
                plt.imshow(maskim, alpha=0.5)
        plt.show()
        plt.close()


