
# coding: utf-8

# In[32]:


import os, sys, cv2, pydicom
import matplotlib.pyplot as plt
import numpy as np


# In[33]:


DPATH = '/home/ubuntu/datasets/chest/CBIS-DDSM'
cases = os.listdir(DPATH)


# In[34]:


import json

with open('.cbis_cache.json') as fl:
    tosave = json.load(fl)
print(len(tosave))


# In[43]:


from random import randint

SAVEPATH = '/home/ubuntu/datasets/tissue/healthy'
__mismatch = 0

def sample_boxes(case, ratio=2000, maxout=50):
    global __mismatch
    sid = case['sid']
    with pydicom.dcmread(case['image']) as db:
        rawimg = db.pixel_array[:]

    
    badmask = False
    mask = np.zeros(rawimg.shape).astype(np.bool)
    coverage = np.zeros(rawimg.shape).astype(np.bool)
    for maskpath in case['masks']:
        with pydicom.dcmread(maskpath) as db:
            rawmask = db.pixel_array[:]
        try: assert rawimg.shape == rawmask.shape
        except:
            __mismatch += 1
            badmask = True
            break
        mask = np.logical_or(mask, rawmask.astype(np.bool))
    if badmask: return None, None
    
    ssize = 320
    ys = rawimg.shape[0] // ssize + 1
    xs = rawimg.shape[1] // ssize + 1
    num_samples = 0
    num_ignored = 0
    for yii in range(ys):
        for xii in range(xs):
            yy = min(yii * ssize, rawimg.shape[0] - ssize)
            xx = min(xii * ssize, rawimg.shape[1] - ssize)
            ipatch = rawimg[yy:yy+ssize, xx:xx+ssize]
            if np.mean(ipatch) > 500:
                mpatch = mask[yy:yy+ssize, xx:xx+ssize]
                if np.sum(mpatch) != 0: 
                    num_ignored += 1
                    continue # skip if there is mask here
                coverage[yy:yy+ssize, xx:xx+ssize] = True
                ref = '%s_%s_%s' % (case['sid'], yy, xx)
                np.save('%s/%s.npy' % (SAVEPATH, ref), ipatch)
                num_samples += 1
                
#     if num_samples > 150:
#         print(num_samples)
#         plt.figure(figsize=(14, 14))
#         plt.subplot(1, 2, 1)
#         plt.imshow(rawimg)
#         plt.subplot(1, 2, 2)
#         plt.imshow(coverage)
#         plt.show()
#         plt.close()
#         assert False
    return num_samples, num_ignored

sampcounts = []
skip = 100
for ii, case in enumerate(tosave[skip:]):
    nsamps, nskips = sample_boxes(case)
    if nsamps is None: continue
    sampcounts.append(nsamps)
    
    sys.stdout.write('%d/%d Samps: %.1f  Skips: %d   Miss: %d   \r' % (
        ii, len(tosave)-skip, np.mean(sampcounts), nskips, __mismatch))
    sys.stdout.flush()
#     break

