import numpy as np
import os
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, 'libs/caffe/python/')
import caffe
sys.path.append('src/voxelize')
import transformAndVisVoxels
import h5py


def predModel_AlexNetDirect(net, in_):
  # shape for input (data blob is N x C x H x W), set data
  net.blobs['data'].reshape(1, *in_.shape)
  net.blobs['data'].data[...] = in_
  # run net and take argmax for prediction
  net.forward()
  return net.blobs['reconst'].data[0,...]


def predModel_3dnw(net, in_):
  return predModel_AlexNetDirect(net, in_)[0, ...]

im_ht = 227
im_wd = 227
transformer = caffe.io.Transformer({'data': (1,3,im_ht,im_wd)})
transformer.set_transpose('data', (2,0,1))
transformer.set_raw_scale('data', 255)  # the reference model operates on images in [0,255] range instead of [0,1]
transformer.set_channel_swap('data', (2,1,0))  # the reference model has channels in BGR order instead of RGB
transformer.set_mean('data', np.array([104.00699,116.66877,122.67892]))

netfile = 'models/deploy.prototxt'
modelfile = 'models/impart.caffemodel'
modelfile2 = 'models/autoenc.caffemodel'
inpath = 'data/chair.jpg'
outfpath = 'output/chair.h5'

net = caffe.Net(netfile, modelfile, caffe.TEST)
if len(modelfile2):
  net.copy_from(modelfile2)

im = plt.imread(inpath)
if np.shape(im)[2] > 3:
  im = im[:, :, :3]
if np.max(im) > 1:
  im = im / 255.0

in_ = transformer.preprocess('data', im)
out = predModel_3dnw(net, in_)

with h5py.File(outfpath, 'w') as f:
  f.create_dataset('reconst', data=out, compression='gzip', compression_opts=9)
