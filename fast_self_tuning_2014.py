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

gt = np.float32(map(lambda x: cv2.imread(x, cv2.IMREAD_GRAYSCALE), gt)) / 255.0
f = np.float32(map(lambda x: cv2.imread(x, cv2.IMREAD_COLOR), f)) / 255.0

K = 5
H, W = f[0].shape[:2]
bg_model = np.zeros((K,) + f[0].shape, np.float32)
efficacy = np.ones((K,H,W), np.int32)
long_fg = np.zeros_like(f[0])
long_fg_cnt = np.zeros((H,W))

def FST_init(frame, R=K/2):
	for i in xrange(H):
		for j in xrange(W):
			i0 = np.clip(i, R, H-R-1)
			j0 = np.clip(j, R, W-R-1)
			bg_model[0,i,j] = frame[i,j]
			for k in xrange(1, R):
				bg_model[k,i,j] = frame[i0+random.randrange(-R, R+1),j0+random.randrange(-R, R+1)]

FST_init(f[0])

def FST_step(frame, eps=0.2, alpha=0.05, theta=10):
	global efficacy
	global bg_model
	global long_fg_cnt

	candidates = np.int32((np.sum(np.abs(bg_model - frame), axis=3) < eps) & (efficacy > 0))
	efficacy -= candidates
	efficacy = np.maximum(efficacy, 0)
	ind = np.argmax((efficacy*candidates).transpose((1,2,0)), axis=2)
	for i in xrange(H):
		for j in xrange(W):
			k = ind[i,j]
			if candidates[k,i,j]:
				efficacy[k,i,j] += 2
	cand3 = np.zeros_like(bg_model, np.int32)
	cand3[:,:,:,0] = candidates
	cand3[:,:,:,1] = candidates
	cand3[:,:,:,2] = candidates

	bg_model -= cand3 * bg_model * alpha
	bg_model += cand3 * frame * alpha

	mask = 1-np.max(candidates, axis=0)

	long_fg_cnt += (np.int32(np.sum(np.abs(long_fg - frame), axis=2) < eps) * mask) * 2 - 1

	for i in xrange(H):
		for j in xrange(W):
			if long_fg_cnt[i,j] <= 0:
				long_fg_cnt[i,j] = 1
				long_fg[i,j] = frame[i,j]
			if long_fg_cnt[i,j] > theta:
				k = np.argmin(efficacy[:,i,j])
				efficacy[k,i,j] = 1
				bg_model[k,i,j] = long_fg[i,j]
				long_fg_cnt[i,j] = 0

	return mask

if args.output is not None:
	for i in xrange(f.shape[0]):
		sec = time.time()
		out = FST_step(f[i])
		print('Frame %d, %.3f sec.' % (i, time.time() - sec))
		if i >= args.output:
			cv2.imwrite('fst-frame.png', f[i] * 255)
			cv2.imwrite('fst-mask.png', out * 255)
			cv2.imwrite('fst-gt.png', gt[i] * 255)
			break
else:
	for i in xrange(f.shape[0]):
		cv2.imshow('Frame', f[i])
		cv2.imshow('Ground-truth', gt[i])
		cv2.imshow('Output', FST_step(f[i]))
		k = cv2.waitKey(0)
		if k == 27:
			break
