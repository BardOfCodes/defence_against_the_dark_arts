#!/usr/bin/env python2

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import ctypes
# import torch

import random
import numpy
import os

import pkgutil
if pkgutil.find_loader("adversarial") is not None:
    # If adversarial module is created by pip install
    QUILTING_LIB = ctypes.cdll.LoadLibrary(os.path.join(os.path.dirname(__file__), "libquilting.so"))
else:
    try:
        QUILTING_LIB = ctypes.cdll.LoadLibrary('libquilting.so')
    except ImportError:
        raise ImportError("libquilting.so not found. Check build script")


def generate_patches(img, patch_size, overlap):
    # assert torch.is_tensor(img) and img.dim() == 3
    assert type(patch_size) == int and patch_size > 0
    assert type(overlap) == int and overlap > 0
    assert patch_size > overlap
    y_range = range(0, img.shape[1] - patch_size, patch_size - overlap)
    x_range = range(0, img.shape[2] - patch_size, patch_size - overlap)
    num_patches = len(y_range) * len(x_range)
    # patches = torch.FloatTensor(num_patches, 3 * patch_size * patch_size).zero_()
    patches = numpy.ascontiguousarray(numpy.zeros((num_patches, 3*patch_size*patch_size),dtype= 'float32'))
    #print(patches.shape, img.shape)
    QUILTING_LIB.generatePatches(
        ctypes.c_void_p(patches.ctypes.data),
        ctypes.c_void_p(img.ctypes.data),
        ctypes.c_uint(img.shape[1]),
        ctypes.c_uint(img.shape[2]),
        ctypes.c_uint(patch_size),
        ctypes.c_uint(overlap)
    )
    #print(patches.shape)
    #print(patches[0])
    return patches


def generate_quilted_images(neighbors, patch_dict, img_h, img_w, patch_size,
                            overlap, graphcut=False, random_stitch=False):
    #assert torch.is_tensor(neighbors) and neighbors.dim() == 1
    #assert torch.is_tensor(patch_dict) and patch_dict.dim() == 2
    assert type(img_h) == int and img_h > 0
    assert type(img_w) == int and img_w > 0
    assert type(patch_size) == int and patch_size > 0
    assert type(overlap) == int and overlap > 0
    assert patch_size > overlap

    result = numpy.ascontiguousarray(numpy.zeros((3, img_h, img_w),dtype = 'float32'))#.zero_()
    #print('here starts the error')
    #print(type(result),type(neighbors),type(patch_dict),type(img_h), type(img_w), type(patch_size), type(overlap), type(graphcut))
    #print(result.shape, neighbors.shape,patch_dict.shape, img_h, img_w, patch_size, overlap, graphcut)
    QUILTING_LIB.generateQuiltedImages(
        ctypes.c_void_p(result.ctypes.data),
        ctypes.c_void_p(neighbors.ctypes.data),
        ctypes.c_void_p(patch_dict.ctypes.data),
        ctypes.c_uint(img_h),
        ctypes.c_uint(img_w),
        ctypes.c_uint(patch_size),
        ctypes.c_uint(overlap),
        ctypes.c_bool(graphcut)
    )

    return result


def select_random_neighbor(neighbors):
    if len(neighbors.shape) == 1:
        # If only 1 neighbor per path is available then return
        return neighbors
    else:
        # Pick a neighbor randomly from top k neighbors for all queries
        nrows = neighbors.shape[0]
        ncols = neighbors.shape[1]
        random_patched_neighbors = numpy.zeros(nrows).astype('int')
        for i in range(0, nrows):
            col = random.randint(0, ncols - 1)
            random_patched_neighbors[i] = neighbors[i, col]
        return random_patched_neighbors


# main quilting function:
def quilting(img, faiss_index, patch_dict, patch_size=9, overlap=2,
             graphcut=False, k=1, random_stitch=False):

    # assertions:
    #assert torch.is_tensor(img)
    #assert torch.is_tensor(patch_dict) and patch_dict.dim() == 2
    assert type(patch_size) == int and patch_size > 0
    assert type(overlap) == int and overlap > 0
    assert patch_size > overlap
    img = numpy.ascontiguousarray(img,dtype = 'float32').transpose(2,0,1)/255.0

    # generate image patches
    patches = generate_patches(img, patch_size, overlap)

    # find nearest patches in faiss index:
    faiss_index.nprobe = 5
    # get top k neighbors of all queries
    _, neighbors = faiss_index.search(patches, k)
    neighbors = select_random_neighbor(neighbors)
    # neighbors = torch.LongTensor(neighbors).squeeze()
    neighbors = neighbors.astype('int')
    #print(neighbors.shape)
    if (neighbors == -1).any():
        print('WARNING: %d out of %d neighbor searches failed.' %
              ((neighbors == -1).sum(), neighbors.nelement()))

    # stitch nn patches in the dict
    #print('starting the quilting')
    quilted_img = generate_quilted_images(neighbors, patch_dict, img.shape[1],
                                          img.shape[2], patch_size, overlap,
                                          graphcut)
    #print('done with quilting')
    return quilted_img*255.0
