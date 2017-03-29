# Experiments with background segmentation algorithms

This repository contains draft implementations of some state-of-the-art background segmentation algorithms.

All implementations are written in Python using OpenCV + Numpy.

Until now this repository has implementations of:

* ViBE [1]
* A Fast Self-tuning Background Subtraction Algorithm [2]
* Background Subtraction using Local SVD Binary Pattern [3]

### Test frame
Frame

![MOG](https://raw.githubusercontent.com/VladX/bgs/master/demos/mog-frame.png)

Ground-truth mask

![MOG](https://raw.githubusercontent.com/VladX/bgs/master/demos/mog-gt.png)

### Mixture of Gaussian from OpenCV
![MOG](https://raw.githubusercontent.com/VladX/bgs/master/demos/mog-mask.png)

### ViBe: A powerful random technique to estimate the background in video sequences (2009)
![VIBE](https://raw.githubusercontent.com/VladX/bgs/master/demos/vibe-mask.png)

### A Fast Self-tuning Background Subtraction Algorithm (2014)
![FST](https://raw.githubusercontent.com/VladX/bgs/master/demos/fst-mask.png)

### Background Subtraction using Local SVD Binary Pattern (2016)
![SVD](https://raw.githubusercontent.com/VladX/bgs/master/demos/svd-mask.png)

### Background Subtraction using Local SVD Binary Pattern + MRF postprocessing
![SVD](https://raw.githubusercontent.com/VladX/bgs/master/demos/svd-mrf.png)

### References

1. Barnich, Olivier; Van Droogenbroeck, Marc (2009). "ViBe: A powerful random technique to estimate the background in video sequences": 945–948. doi:10.1109/ICASSP.2009.4959741
2. B. Wang and P. Dudek. A fast self-tuning background subtraction algorithm. In IEEE Workshop on Change Detection, 2014.
3. L. Guo, D. Xu, and Z. Qiang. Background Subtraction using Local SVD Binary Pattern. International Conference on Computer Vision and Pattern Recognition, CVPR 2016, June 2016.
