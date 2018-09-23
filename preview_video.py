
import cv2
import argparse
import os, sys

parser = argparse.ArgumentParser(description='')
parser.add_argument('--folder', required=True)
parser.add_argument('--tag', required=True)
args = parser.parse_args()


imgs = ['%s/%s' % (args.folder, name) for name in os.listdir(args.folder) if args.tag in name]

def skey(val):
	seq = val.split('.')[0].split('/')[-1]
	iter = seq.split('_')[-1]
	return int(iter)

imgs = sorted(imgs, key=skey)
print(len(imgs), imgs[0], imgs[-1])



from cv2 import imread

writer = None
decshape = None
for ii, im_path in enumerate(imgs):
	sys.stdout.write('%d/%d : %s\r' % (ii+1, len(imgs), im_path))
	sys.stdout.flush()

	im = cv2.cvtColor(imread(im_path), cv2.COLOR_BGR2RGB)
	
	if writer is None:
		writer = cv2.VideoWriter('preview_%s.avi' % args.tag,
						cv2.VideoWriter_fourcc(*'MJPG'),
						10, (im.shape[1], im.shape[0]), True)
		decshape = im.shape
	try:
		writer.write(im)
	except:
		print(im_path, decshape, im.shape)
		print()

writer.release()
print()

