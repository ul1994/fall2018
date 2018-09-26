
from configs import *
import os, sys
import cv2
import numpy as np
from glob import glob
from scipy.ndimage import gaussian_filter as blur
from scipy.ndimage import rotate
from random import shuffle, randint
import json

CPATH = '%s/tissue/cancers' % DATAPATH
HPATH = '%s/tissue/healthy'  % DATAPATH

def strip_sid(raw):
    sid = '_'.join(raw.split('_')[:-2])
    return sid

class TrainTest:
    def __init__(self, label, all, resnames, get_id=strip_sid):
        self.label = label
        self.train = []
        self.test = []
        for path in all:
            if get_id(path.split('/')[-1]) in resnames:
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
            batch = self.train[self.tr:self.tr+bsize]
            self.tr += bsize
            return batch
        elif mode == 'test':
            if self.te + bsize > len(self.test):
                self.te = 0
                shuffle(self.test)
            batch = self.test[self.te:self.te+bsize]
            self.te += bsize
            return batch

    def summary(self):
        print(' [*] %s: %d / %d' % (self.label, len(self.train), len(self.test)))

def load_folder(fname, label, reserve, get_id=strip_sid):
    imgpaths = glob(fname + '/*.npy')
    assert len(imgpaths) > 0

    bycases = {}
    for samp in imgpaths:
        sid = get_id(samp)
        if sid not in bycases: bycases[sid] = []
        bycases[sid].append(samp)

    print(' [*] %s: %d unique cases (%d total)' % (label, len(bycases), len(imgpaths)))

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

    split = TrainTest(label, imgpaths, resnames, get_id=get_id)
    return split

def centercrop(ls, imsize, slack=32, cropspec=None, refs=None):

    history = []
    # return [img[pad:imsize+pad, pad:imsize+pad] for img in ls]
    cropped = []
    for imii, img in enumerate(ls):
        native = img.shape[0]
        try:
            assert native == 320
        except:
            print(img.shape)
            if refs is not None:
                print(imii, refs[imii])
            assert False
        if cropspec is None:
            pad = (native - imsize) // 2
            slack = min(pad, slack) # cant exceed pad
            rx, ry = randint(-slack, slack), randint(-slack, slack)
            x0, y0 = pad + rx, pad + ry # rx=0 would be centercrop
        else:
            x0, y0 = cropspec[imii]
        img = img[y0:y0+imsize, x0:x0+imsize]
        history.append((x0, y0)) # save this for later
        assert img.shape[:2] == (imsize, imsize)
        cropped.append(img)
    return history, cropped

class Tissue:
    def __init__(self, reserve=512):
        direct_id = lambda fname: fname.replace('.npy', '')

        self.sick = load_folder(CPATH, 'tissue_sick', reserve, get_id=direct_id)
        self.healthy = load_folder(HPATH, 'tissue_healthy', reserve)

        self.sick.summary()
        self.healthy.summary()

        self.train_size = min(len(self.sick.train), len(self.healthy.train))
        self.test_size = min(len(self.sick.test), len(self.healthy.test))

    def gen(self, mode='train', imsize=256, bsize=64, regionsize=16, set=None,
        sickonly=False,
        augment=True,
        labels=['masks']
    ):
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

            paths, refs = imgs, imgs
            imgs = [np.load(path) for path in paths]
            imgs = [img.astype(np.float32) for img in imgs]

            # rotate augment
            if augment:
                rotatespec = [[0, 90, 180, 270][randint(0, 3)] for ii in range(len(imgs))]
                imgs = [rotate(imgs[ii], rotatespec[ii], reshape=False) for ii in range(len(imgs))]
            # TODO: CROPPPING
            cropspec, imgs = centercrop(imgs, imsize, refs=refs)


            imgs = np.array(imgs).reshape((bsize, imsize, imsize, 1))

            imgs /= 255**2 # uint16 range

            assert lbls.shape == (bsize, 2)

            if labels == ['lbls']:
                yield imgs, lbls
            elif 'regions' in labels:
                regions = []
                for bii, pth in enumerate(paths):
                    canvas = np.zeros((320, 320, 2), dtype=int)
                    if np.argmax(lbls[bii]) == 1: #for sick
                        pth = pth.replace('.npy', '.jpg')
                        msk = cv2.imread(pth, 0).astype(np.float32)/255
                        canvas[:, :, 1] = msk
                        canvas[:, :, 0] = 1 - msk
                    else:
                        canvas[:, :, 0] = 1
                    regions.append(canvas.astype(np.float32))
                if augment:
                    regions = [rotate(regions[ii], rotatespec[ii], reshape=False) for ii in range(len(regions))]
                _, regions = centercrop(regions, imsize, cropspec=cropspec, refs=refs)
                # print(len(regions), regions[0].shape)
                scale = regionsize/imsize
                regions = [cv2.resize(
                    img, (0,0), fx=scale, fy=scale,
                    interpolation=cv2.INTER_NEAREST) for img in regions]
                regions = np.array(regions).reshape((bsize, regionsize, regionsize, 2))

                if 'lbls' in labels and 'refs' in labels:
                    yield imgs, [regions, lbls], refs
                elif 'lbls' in labels:
                    yield imgs, [regions, lbls]
                else:
                    yield imgs, regions

            elif 'masks' in labels:
                masks = []
                for bii, pth in enumerate(paths):
                    if np.argmax(lbls[bii]) == 1: #for sick
                        pth = pth.replace('.npy', '.jpg')
                        msk = cv2.imread(pth, 0).astype(np.float32)/255
                        masks.append(msk)
                    else:
                        masks.append(np.zeros((320, 320)))
                if augment:
                    masks = [rotate(masks[ii], rotatespec[ii], reshape=False) for ii in range(len(masks))]
                _, masks = centercrop(masks, imsize, cropspec=cropspec, refs=refs)
                # print(len(masks), masks[0].shape)
                masks = np.array(masks).reshape((bsize, imsize, imsize, 1))
                if labels == ['masks']:
                    yield imgs, masks
                elif 'lbls' in labels and 'refs' in labels:
                    yield imgs, masks, lbls, refs
                elif 'lbls' in labels:
                    yield imgs, masks, lbls


def sample_noise(mask, ratio=5, external=50):
    yy, xx = np.nonzero(mask)
    hh = np.max(yy) - np.min(yy)
    ww = np.max(xx) - np.min(xx)
    samps = int(max(np.sqrt(hh * ww) // ratio, 1))
    # print('Samples', samps)
    canvas = np.zeros(mask.shape)
    for ii in range(samps):
        ind = randint(0, len(yy) - 1)
        yi, xi = yy[ind], xx[ind]
        canvas[yi, xi] = 1

    for ii in range(external):
        rx, ry = randint(0, mask.shape[1] - 1), randint(0, mask.shape[0] - 1)
        canvas[ry, rx] = 1

    canvas = blur(canvas, 3)
    canvas /= np.max(canvas)
    return canvas

class Contours(Tissue):
    cpath = '%s/contours/centered' % DATAPATH
    hpath = '%s/contours/healthy' % DATAPATH
    # spath = '%s/contours/sick'  % DATAPATH # sick contours not used

    def __init__(self, reserve=512):
        get_coord_id = lambda fname: '_'.join(fname.split('_')[:-1])
        get_indexed_id = lambda fname: '_'.join(fname.split('_')[:-2])

        # with open('.cbis_detailed.json') as fl:
        #     self.metadata = json.load(fl)
        #     self.lbls_lookup = {}
        #     for ent in self.metadata:
        #         for det in ent['details']:
        #             # maskid = original folder where mask was found
        #             self.lbls_lookup[det['maskid']] = det['type']

        self.sick = load_folder(self.cpath, 'sick', reserve, get_id=get_indexed_id)
        self.healthy = load_folder(self.hpath, 'healthy', reserve, get_id=get_coord_id)

        self.sick.summary()
        self.healthy.summary()

        self.train_size = min(len(self.sick.train), len(self.healthy.train))
        self.test_size = min(len(self.sick.test), len(self.healthy.test))

    def gen(self, mode='train', imsize=512, bsize=32,
        native=1024,
        sickonly=False,
        augment=True,
        labels=['noise', 'masks', 'refs']
    ):
        imscale = imsize / native
        bhalf = bsize//2
        while True:
            lbls = np.zeros((bsize, 2))
            sick_imgs = self.sick.next(bhalf, mode=mode)
            if sickonly:
                healthy_imgs = self.sick.next(bhalf, mode=mode)
                lbls[:, 1] = 1
            else:
                healthy_imgs = self.healthy.next(bhalf, mode=mode)
                lbls[:len(healthy_imgs), 0] = 1
                lbls[len(healthy_imgs):, 1] = 1


            imgs = healthy_imgs+sick_imgs

            batch = list(zip(imgs, lbls))
            shuffle(batch)
            imgs, lbls = zip(*batch)
            lbls = np.array(list(lbls))

            refs = imgs
            imgs = [np.load(path) for path in refs]
            imgs = [img.astype(np.float32) for img in imgs]

            imgs = [cv2.resize(img, (0,0), fx=imscale, fy=imscale) for img in imgs]

            # rotate augment
            # if augment:
            #     rotatespec = [[0, 90, 180, 270][randint(0, 3)] for ii in range(len(imgs))]
            #     imgs = [rotate(imgs[ii], rotatespec[ii], reshape=False) for ii in range(len(imgs))]
            # TODO: CROPPPING
            # cropspec, imgs = centercrop(imgs, imsize, refs=refs)

            imgs = np.array(imgs).reshape((bsize, imsize, imsize, 1))
            imgs /= 255**2 # uint16 range

            masks = []
            noise = []
            for bii, pth in enumerate(refs):
                if self.cpath in refs[bii]: # is sick and has mask
                    pth = pth.replace('.npy', '.jpg')
                    msk = cv2.imread(pth, 0).astype(np.float32)/255
                    msk = cv2.resize(msk, (0,0), fx=imscale, fy=imscale)
                    masks.append(msk)
                    # noiseim = np.zeros(imgs[0].shape[:2])
                    noiseim = sample_noise(msk)
                    assert noiseim.shape == imgs[0].shape[:2]
                    noise.append(noiseim)
                else:
                    blank = np.zeros(imgs[0].shape[:2])
                    assert blank.shape == imgs[0].shape[:2]
                    masks.append(blank)
                    noise.append(blank)
                # if augment:
                #     masks = [rotate(masks[ii], rotatespec[ii], reshape=False) for ii in range(len(masks))]
                # _, masks = centercrop(masks, imsize, cropspec=cropspec, refs=refs)
                # print(len(masks), masks[0].shape)
            masks = np.array(masks).reshape((bsize, imsize, imsize, 1))
            noise = np.array(noise).reshape((bsize, imsize, imsize, 1))

            if 'masks' in labels and 'refs' in labels:
                yield imgs, masks, lbls, refs
            elif 'masks' in labels:
                yield imgs, masks, lbls
            else:
                yield imgs, lbls
