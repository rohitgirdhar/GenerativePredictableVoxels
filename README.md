Testing code release for

R. Girdhar, D. F. Fouhey, M. Rodriguez and A. Gupta     
[Learning a Predictable and Generative Vector Representation for Objects](https://rohitgirdhar.github.io/GenerativePredictableVoxels/)     
In Proc. of European Conference on Computer Vision (ECCV), 2016 

## Pre-requisites
1. Caffe (trained and tested with [97f4536](https://github.com/BVLC/caffe/tree/97f4536d51c5ed464025179ea36798ebee8bf033), though should work with the latest version). Clone and install in `libs` dir.
2. Python libs `h5py`, `matplotlib`, `mayavi`.

## Download pre-trained models
Download all the models from [dropbox](https://www.dropbox.com/sh/ic5nb3e3gd4y99x/AABihSLOgeXxtZ81rbnHeq10a) to `models/` dir.

## Testing using the precomputed networks

```bash
$ python src/testing/reconst.py  # stores the prediction in output/ folder
```

Dataset splits, training code to come soon.

