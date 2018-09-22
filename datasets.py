
from configs import *
import os, sys
import cv2
import numpy as np
from glob import glob
from scipy.ndimage import gaussian_filter as blur
from random import shuffle, randint
import json

CPATH = '%s/tissue/cancers' % DATAPATH
HPATH = '%s/tissue/healthy'  % DATAPATH

def strip_sid(raw):
    sid = '_'.join(raw.split('_')[:-2])
    return sid

class TrainTest:
    def __init__(self, label, all, resnames):
        self.label = label
        self.train = []
        self.test = []
        for path in all:
            if strip_sid(path.split('/')[-1]) in resnames:
                self.test.append(path)
            else:
                self.train.append(path)

        assert len(self.train) > 0
        assert len(self.test) > 0

        self.tr = 0
        self.te = 0

    def next(self, bsize, mode='train'):
        if mode == 'train':
            if self.tr + bsize > len(self.train):
                self.tr = 0
                shuffle(self.train)

            return self.train[self.tr:self.tr+bsize]
        elif mode == 'test':
            if self.te + bsize > len(self.test):
                self.te = 0
                shuffle(self.test)

            return self.test[self.te:self.te+bsize]

    def summary(self):
        print(' [*] %s: %d / %d' % (self.label, len(self.train), len(self.test)))

def load_folder(fname, label, reserve):
    imgpaths = glob(fname + '/*.npy')
    assert len(imgpaths) > 0

    bycases = {}
    for samp in imgpaths:
        sid = strip_sid(samp)
        if sid not in bycases: bycases[sid] = []
        bycases[sid].append(samp)

    print(' [*] %s: %d unique cases' % (label, len(bycases)))

    cachename = '.%s_reserved.json' % label
    if not os.path.isfile(cachename):
        inds = list(range(len(bycases)))
        shuffle(inds)
        resnames = [list(bycases.keys())[ii] for ii in inds[:reserve]]
        resnames = [nm.split('/')[-1] for nm in resnames]
        with open(cachename, 'w') as fl:
            json.dump(resnames, fl)
    else:
        with open(cachename) as fl:
            resnames = json.load(fl)
    assert len(resnames) == reserve

    split = TrainTest(label, imgpaths, resnames)
    return split

def centercrop(ls, imsize, pad=64):
    return [img[pad:imsize+pad, pad:imsize+pad] for img in ls]

class Tissue:
    def __init__(self, reserve=512):
        

        self.sick = load_folder(CPATH, 'tissue_sick', reserve)
        self.healthy = load_folder(HPATH, 'tissue_healthy', reserve)

        self.sick.summary()
        self.healthy.summary()

        self.train_size = min(len(self.sick.train), len(self.healthy.train))
        self.test_size = min(len(self.sick.test), len(self.healthy.test))

    def gen(self, mode='train', imsize=256, bsize=64, set=None, sickonly=False, mask=False):
        bhalf = bsize//2
        while True:
            sick_imgs = self.sick.next(bhalf, mode=mode)
            if sickonly:
                healthy_imgs = self.sick.next(bhalf, mode=mode)
                lbls = np.zeros((bsize, 2))
                lbls[:, 1] = 1 # healthy .... sick
            else:
                healthy_imgs = self.healthy.next(bhalf, mode=mode)
                lbls = np.zeros((bsize, 2))
                lbls[:len(healthy_imgs), 0] = 1
                lbls[len(healthy_imgs):, 1] = 1 # healthy .... sick

            imgs = healthy_imgs+sick_imgs
            assert len(imgs) == len(lbls)

            batch = list(zip(imgs, lbls))
            shuffle(batch)
            imgs, lbls = zip(*batch)
            lbls = np.array(lbls)

            paths = imgs
            imgs = [np.load(path) for path in paths]
            imgs = [img.astype(np.float32) for img in imgs]

            # TODO: CROPPPING
            imgs = centercrop(imgs, imsize)
            

            imgs = np.array(imgs).reshape((bsize, imsize, imsize, 1))

            imgs /= 255**2 # uint16 range

            assert lbls.shape == (bsize, 2)

            # assert imgs[0].shape[:2] == (imsize, imsize)
            # imgs = np.array(imgs).reshape((bsize, imsize, imsize, 1))
            if not mask:
                yield imgs, lbls
            else:
                masks = []
                for pth in paths:
                    pth = pth.replace('.npy', '.jpg')
                    msk = cv2.imread(pth, 0).astype(np.float32)/255
                    masks.append(msk)
                masks = centercrop(masks, imsize)
                yield imgs, masks, lbls
