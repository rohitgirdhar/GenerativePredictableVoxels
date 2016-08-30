import sys
import os
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import scipy.misc


def saveVisSnapshotMayavi(data, outfpath=None, fig=None):
  import mayavi.mlab
  figWasNone = 0
  if fig is None:
    mayavi.mlab.options.offscreen = True
    fig = mayavi.mlab.figure(bgcolor=(1,1,1))
    figWasNone = 1
  else:
    mayavi.mlab.clf(fig)
  visualizeDenseMayavi(data, fig)
  mayavi.mlab.view(113.21283385785944,142.9695294105835,roll=93.37654235007402)
  I = mayavi.mlab.screenshot(fig)
  if figWasNone:
    mayavi.mlab.close(fig)
  if outfpath is not None:
    scipy.misc.imsave(outfpath, I)
  return I


def visualizeMayavi(data_sparse, fig, weights=None):
  import mayavi.mlab
  if weights is None:
    mayavi.mlab.points3d(data_sparse[0,:], data_sparse[1,:],
        data_sparse[2,:], scale_factor=1,
        color=(0.55,0,0), mode='cube', figure=fig)
  else:
    mayavi.mlab.points3d(data_sparse[0,:], data_sparse[1,:],
        data_sparse[2,:], weights, transparent=True, scale_factor=1,
        mode='cube', figure=fig)
  mayavi.mlab.view(-90,180)


# gen this once # TODO: Find better way to do this
VOXEL_DIM=20
_coords = []
for i in range(VOXEL_DIM):
  for j in range(VOXEL_DIM):
    for k in range(VOXEL_DIM):
      _coords.append(np.array([i,j,k]))
_coords = np.array(_coords)
def visualizeDenseMayavi(data, fig=None):
  import mayavi.mlab
  if not fig:
    fig2 = mayavi.mlab.figure(bgcolor=(1,1,1))
  else:
    fig2 = fig
  if data.dtype == 'bool':
    visualizeMayavi(binvox_rw.dense_to_sparse(data), fig2)
  else:
    values = np.reshape(data, (-1, 1), 'C')
    coords = _coords[...] 
    visualizeMayavi(coords.transpose(), fig2, values[:, 0])
  if not fig:
    mayavi.mlab.show()


# This ensures the output model is such that the 3rd dimension
# is the real depth
# so the closest parts are data[:,:,1] and farthest data[:,:,n].
# Also useful with visualizeDense_2d vis
def readModel(mpath):
  model = sio.loadmat(mpath, squeeze_me=True, struct_as_record=False)
  data = model['grid']
  data = np.rot90(data, 2)
  data = np.swapaxes(data, 1, 2)
  data = data[:, ::-1, :]
  return data
