# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import tensorflow as tf
import defence_config as dc
import sys
import pdb
sys.path.insert(0,'tensorflow-classification')
from misc.utils import *
try:
    import cPickle as pickle
except ImportError:
    import pickle
import progressbar
import random

import faiss
#from lib.dataset import load_dataset, get_data_loader
#import lib.opts as opts



# function that indexes a large number of patches:
def simple_gather_patches(img_loader, imgs, num_patches, patch_size, patch_transform=None):
    
    # gt_labels = open(gt_labels).readlines()#[::10]
    num_images = len(imgs)
    patches, n = [], 0
    bar = progressbar.ProgressBar(num_patches)
    bar.start()
    
    for i in range(num_images):
        img = img_loader(imgs[i].strip(), False)
        mean=np.array([103.939, 116.779, 123.68])
        img = np.clip(img + mean,0,255)
        img = img.transpose(2,0,1)
        # print(type(img),img.shape)
        for _ in range(0, max(1, int(num_patches / num_images))):
            n += 1
            y = random.randint(0, img.shape[1] - patch_size)
            x = random.randint(0, img.shape[2] - patch_size)
            patch = img[:,y:y + patch_size, x:x + patch_size]
            if patch_transform is not None:
                patch = patch_transform(patch)
            patches.append(patch)
            if n % 100 == 0:
                bar.update(n)
            if n >= num_patches:
                break
        if n >= num_patches:
            break

    # copy all patches into single tensor:
    patches = np.stack(patches, axis=0)
    print(patches.shape)
    patches = patches.reshape((patches.shape[0], int(patches.size / patches.shape[0])))
    print(patches.shape)
    patches = patches/255.0
    return patches

# function that trains faiss index on patches and saves them:
def index_patches(patches, index_file, pca_dims=64):

    # settings for faiss:
    num_lists, M, num_bits = 200, 16, 8

    # assertions:
    assert type(pca_dims) == int and pca_dims > 0
    if pca_dims > patches.shape[1]:
        print('WARNING: Input dimension < %d. Using fewer PCA dimensions.' % pca_dims)
        pca_dims = patches.shape[1] - (patches.shape[1] % M)

    # construct faiss index:
    quantizer = faiss.IndexFlatL2(pca_dims)
    assert pca_dims % M == 0
    sub_index = faiss.IndexIVFPQ(quantizer, pca_dims, num_lists, M, num_bits)
    pca_matrix = faiss.PCAMatrix(patches.shape[1], pca_dims, 0, True)
    faiss_index = faiss.IndexPreTransform(pca_matrix, sub_index)

    # train faiss index:
    patches = patches#.numpy()
    faiss_index.train(patches)
    faiss_index.add(patches)

    # save faiss index:
    print('| writing faiss index to %s' % index_file)
    faiss.write_index(faiss_index, index_file)


# run all the things:
def create_faiss_patches(args):

    # load image dataset:
    
    isotropic, size = get_params(args['network'])
    config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))
    sess = tf.Session(config=config)
    sess.run(tf.global_variables_initializer())
    img_loader = loader_func(args['network'], sess, isotropic, size, None,'None') 
    imgs = open(args['img_list']).readlines()#[::10]

    print('| set up image loader...')
    # image_dataset = load_dataset(args, 'train', None, with_transformation=True)
    # image_dataset.imgs = image_dataset.imgs[:10000]  # we don't need all images

    # gather image patches:
    print('| gather image patches...')

    patches = simple_gather_patches(
        img_loader, imgs, args['num_patches'], args['quilting_patch_size'],
        patch_transform=None,
    )

    # save patches:
    patches = patches.astype('float32')
    with open(args['patches_file'], 'wb') as fwrite:
        print('| writing patches to %s' % args['patches_file'])
        pickle.dump(patches, fwrite, pickle.HIGHEST_PROTOCOL)
    # build faiss index:
    print('| training faiss index...')
    index_patches(patches, args['index_file'], pca_dims=args['pca_dims'])



# run:
if __name__ == '__main__':
    # parse input arguments:
    args = dc.quilt_params
    create_faiss_patches(args)
