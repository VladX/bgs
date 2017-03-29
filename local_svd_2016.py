import numpy as np
import cv2
import argparse
import os
import time
import random
import maxflow

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

S = 20
H, W = f[0].shape[:2]
samples_int = np.zeros((S,) + f[0].shape, np.float32)
samples_lsbp = np.zeros((S,H,W,9), np.bool)
D = np.ones((S,H,W), np.float32) * 0.2
Racc = np.ones((H,W), np.float32) * 0.2
Tacc = np.ones((H,W), np.float32) * 0.08

def extract_LSBP(frame, R1=1, R2=1, tau=0.05):
	intensity = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	intens = np.zeros((H+2*R1, W+2*R1), np.float32)
	intens[R1:H+R1, R1:W+R1] = intensity
	g = np.zeros((H+2*R2, W+2*R2), np.float32)
	for i in xrange(H):
		for j in xrange(W):
			s = np.linalg.svd(intens[i:i+2*R1+1,j:j+2*R1+1], compute_uv=False)
			g[i+R2,j+R2] = (s[1] + s[2]) / s[0]
	lsbp = np.zeros((H, W, (2*R2+1)**2), np.bool)
	for i in xrange(H):
		for j in xrange(W):
			lsbp[i,j] = (np.abs(g[i:i+2*R2+1,j:j+2*R2+1] - g[i+R2,j+R2]) < tau).ravel()
	return lsbp

def SVD_init(frame, R=S/2):
	lsbp = extract_LSBP(frame)
	for i in xrange(H):
		for j in xrange(W):
			i0 = np.clip(i, R, H-R-1)
			j0 = np.clip(j, R, W-R-1)
			samples_int[0,i,j] = frame[i,j]
			samples_lsbp[0,i,j] = lsbp[i,j]
			for k in xrange(1, R):
				i1, j1 = i0+random.randrange(-R, R+1), j0+random.randrange(-R, R+1)
				samples_int[k,i,j] = frame[i1,j1]
				samples_lsbp[k,i,j] = lsbp[i1,j1]

SVD_init(f[0])

def SVD_step(frame, HR=4, threshold=2, Rscale=5, Rlr=0.1, Tlr=0.02):
	global Racc
	global Tacc
	lsbp = extract_LSBP(frame)
	Racc = (1.0 - Rlr) * Racc + Rlr * (np.mean(D, axis=0) * Rscale)
	mask = np.float32(np.sum((np.sum(np.abs(samples_int - frame), axis=3) < Racc) & (np.sum(samples_lsbp ^ lsbp, axis=3) < HR), axis=0) < threshold)
	Tacc = (1.0 - Tlr) * Tacc + Tlr * (1.0 - mask)
	Tacc = np.clip(Tacc, 0.05, 0.8)
	for i in xrange(H):
		for j in xrange(W):
			if mask[i,j] == 0.0:
				if random.random() < Tacc[i,j]:
					k = random.randrange(0, S)
					dist = np.sum(np.abs(samples_int[:,i,j] - frame[i,j]), axis=1)
					dist.sort()
					D[k,i,j] = dist[1]
					samples_int[k,i,j] = frame[i,j]
					samples_lsbp[k,i,j] = lsbp[i,j]
				if random.random() < Tacc[i,j]:
					k = random.randrange(0, S)
					i0, j0 = np.clip(i+random.randrange(-1,2), 0, H-1), np.clip(j+random.randrange(-1,2), 0, W-1)
					dist = np.sum(np.abs(samples_int[:,i0,j0] - frame[i,j]), axis=1)
					dist.sort()
					D[k,i0,j0] = dist[1]
					samples_int[k,i0,j0] = frame[i,j]
					samples_lsbp[k,i0,j0] = lsbp[i,j]
	return mask

def postprocessing(im, unary):
	unary = np.float32(unary)
	unary = cv2.GaussianBlur(unary, (9, 9), 0)

	g = maxflow.Graph[float]()
	nodes = g.add_grid_nodes(unary.shape)

	for i in xrange(im.shape[0]):
		for j in xrange(im.shape[1]):
			v = nodes[i,j]
			g.add_tedge(v, -unary[i,j], -1.0+unary[i,j])

	def potts_add_edge(i0, j0, i1, j1):
		v0, v1 = nodes[i0,j0], nodes[i1,j1]
		w = 0.1 * np.exp(-((im[i0,j0] - im[i1,j1])**2).sum() / 0.1)
		g.add_edge(v0, v1, w, w)

	for i in xrange(1,im.shape[0]-1):
		for j in xrange(1,im.shape[1]-1):
			potts_add_edge(i, j, i, j-1)
			potts_add_edge(i, j, i, j+1)
			potts_add_edge(i, j, i-1, j)
			potts_add_edge(i, j, i+1, j)

	g.maxflow()
	seg = np.float32(g.get_grid_segments(nodes))
	return seg

if args.output is not None:
	for i in xrange(f.shape[0]):
		sec = time.time()
		out = SVD_step(f[i])
		mrf = postprocessing(f[i], out)
		print('Frame %d, %.3f sec.' % (i, time.time() - sec))
		if i >= args.output:
			cv2.imwrite('svd-frame.png', f[i] * 255)
			cv2.imwrite('svd-mask.png', out * 255)
			cv2.imwrite('svd-gt.png', gt[i] * 255)
			cv2.imwrite('svd-mrf.png', mrf * 255)
			break
else:
	for i in xrange(f.shape[0]):
		cv2.imshow('Frame', f[i])
		cv2.imshow('Ground-truth', gt[i])
		out = SVD_step(f[i])
		cv2.imshow('Output', out)
		cv2.imshow('Output + MRF', postprocessing(f[i], out))
		k = cv2.waitKey(0)
		if k == 27:
			break
