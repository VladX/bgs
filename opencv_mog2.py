import numpy as np
import cv2
import argparse
import os
import time
import random

argparser = argparse.ArgumentParser(description='Draft implementation of the VIBE background subtraction algorithm.')

argparser.add_argument('-g', '--gt', help='Directory with ground-truth frames', required=True)
argparser.add_argument('-f', '--frames', help='Directory with input frames', required=True)
argparser.add_argument('-k', '--output', type=int, help='Calculate answer for K-th frame and output')
args = argparser.parse_args()

gt = map(lambda x: os.path.join(args.gt, x), os.listdir(args.gt))
gt.sort()
f = map(lambda x: os.path.join(args.frames, x), os.listdir(args.frames))
f.sort()

gt = np.uint8(map(lambda x: cv2.imread(x, cv2.IMREAD_GRAYSCALE), gt))
f = np.uint8(map(lambda x: cv2.imread(x, cv2.IMREAD_COLOR), f))

bgs = cv2.bgsegm.createBackgroundSubtractorMOG()

def MOG_step(frame):
	mask = bgs.apply(frame)
	return mask

if args.output is not None:
	for i in xrange(f.shape[0]):
		sec = time.time()
		out = MOG_step(f[i])
		print('Frame %d, %.3f sec.' % (i, time.time() - sec))
		if i >= args.output:
			cv2.imwrite('mog-frame.png', f[i])
			cv2.imwrite('mog-mask.png', out)
			cv2.imwrite('mog-gt.png', gt[i])
			break
else:
	for i in xrange(f.shape[0]):
		cv2.imshow('Frame', f[i])
		cv2.imshow('Ground-truth', gt[i])
		cv2.imshow('MOG OpenCV', MOG_step(f[i]))
		k = cv2.waitKey(0)
		if k == 27:
			break
