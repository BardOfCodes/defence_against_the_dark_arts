import numpy as np
from skimage.io import imread
from skimage.transform import resize
import tensorflow as tf
import cv2
import pickle
import math

from PIL import Image
import defence_config as dc
from advanced_defences.transformations.tvm import reconstruct as tvm
from advanced_defences.transformations.quilting_fast import quilting
# util layers

# Obselete


def old_img_preprocess(img_path, size=224):
    mean = [103.939, 116.779, 123.68]
    img = imread(img_path)
    img = resize(img, (size, size))*255.0
    if len(img.shape) == 2:
        img = np.dstack([img, img, img])
    img[:, :, 0] -= mean[2]
    img[:, :, 1] -= mean[1]
    img[:, :, 2] -= mean[0]
    img[:, :, [0, 1, 2]] = img[:, :, [2, 1, 0]]
    img = np.reshape(img, [1, size, size, 3])
    return img
# Preprocessing for Inception V3


def v3_preprocess(img_path, pert_load, def_func, pert):
    img = imread(img_path)
    img = resize(img, (299, 299), preserve_range=True)
    if pert_load: img = np.clip(img+pert, 0,255)
    img = def_func(img)
    img = (img - 128) / 128
    if len(img.shape) == 2:
        img = np.dstack([img, img, img])
    img = np.reshape(img, [1, 299, 299, 3])
    return img

# Image preprocessing format
# Fog VGG models.


def vgg_preprocess(img_path,pert_loader,def_func, pert, size=224):
    mean = [103.939, 116.779, 123.68]
    img = imread(img_path)
    if len(img.shape) == 2:
        img = np.dstack([img, img, img])
    resFac = 256.0/min(img.shape[:2])
    newSize = list(map(int, (img.shape[0]*resFac, img.shape[1]*resFac)))
    img = resize(img, newSize, mode='constant', preserve_range=True)
    offset = [newSize[0]/2.0 -
              np.floor(size/2.0), newSize[1]/2.0-np.floor(size/2.0)]
    # print(offset,size)
    img = img[int(offset[0]):int(offset[0])+size,
              int(offset[1]):int(offset[1])+size, :]
    # this is where the reform should take place, and this is where the pert should be added.a
    img = def_func(img)
    if pert_load: img = np.clip(img+pert, 0,255)[0]
    img[:, :, 0] -= mean[2]
    img[:, :, 1] -= mean[1]
    img[:, :, 2] -= mean[0]
    img[:, :, [0, 1, 2]] = img[:, :, [2, 1, 0]]
    img = np.reshape(img, [1, size, size, 3])
    return img

# For Resnets,Caffenet and Googlenet
# From Caffe-tensorflow


def img_preprocess(img, scale=256, isotropic=False, crop=227, mean=np.array([103.939, 116.779, 123.68])):
    '''Crops, scales, and normalizes the given image.
    scale : The image wil be first scaled to this size.
            If isotropic is true, the smaller side is rescaled to this,
            preserving the aspect ratio.
    crop  : After scaling, a central crop of this size is taken.
    mean  : Subtracted from the image
    '''
    # Rescale
    if isotropic:
        img_shape = tf.to_float(tf.shape(img)[:2])
        min_length = tf.minimum(img_shape[0], img_shape[1])
        new_shape = tf.to_int32((scale / min_length) * img_shape)
    else:
        new_shape = tf.stack([scale, scale])
    img = tf.image.resize_images(img, new_shape)
    # Center crop
    # Use the slice workaround until crop_to_bounding_box supports deferred tensor shapes
    # See: https://github.com/tensorflow/tensorflow/issues/521
    offset = (new_shape - crop) / 2
    img = tf.slice(img, begin=tf.stack(
        [offset[0], offset[1], 0]), size=tf.stack([crop, crop, -1]))
    # Mean subtraction

    return tf.to_float(img)


def load_image():
    # Read the file
    image_path = tf.placeholder(tf.string, None)
    file_data = tf.read_file(image_path)
    # Decode the image data
    img = tf.image.decode_jpeg(file_data, channels=3)
    img = tf.reverse(img, [-1])
    return img, image_path

def defence_func(defence_name):
    quality = 50
    quant = 32
    name = 'temp.jpg'
    if defence_name == 'None':
        def func(img): return img
    elif defence_name == 'Gaussian':
        dicter = dc.defence_params['gaussian']
        def func(img, kernel_size = dicter['kernel_size'], sigma = dicter['sigma']):
            #print(type(img))
            return cv2.GaussianBlur(img, kernel_size, sigma)
    elif defence_name == 'Median':
        def func(img, kernel_size = dc.defence_params['median']['kernel_size']):
            return cv2.medianBlue(img, kernel_size)
    elif defence_name == 'Bilateral':
        dicter = dc.defence_params['bilateral']
        def func(img, diameter = dicter['diameter'], color_sigma = dicter['color_sigma'], 
                  space_sigma = dicter['space_sigma']):
            return cv2.bilateralFilter(img,diameter,color_sigma,space_sigma)
    elif defence_name == 'Bit_Compression':
        def func(img,quant= dc.defence_params['bit']['quant']):
            img = np.clip(np.round(img/quant)*quant,0,255)
            return img
    elif defence_name == 'JPEG':
        dicter = dc.defence_params['jpeg']
        def func(img, name = dicter['temp_name'], quality=dicter['quality']):
            img = np.uint8(img)
            img = Image.fromarray(img)
            img.save(name, quality = quality)
            load_img = np.array(Image.open(name))
            return load_img
    elif defence_name == 'tvm':
        dicter = dc.defence_params['tvm']
        def func(img, pix_drop_rate = dicter['pixel_drop_rate'], method = dicter['tvm_method'], weight = dicter['tvm_weight']):
            img = (img-128)/256.0*2
            img = tvm(img,pix_drop_rate, method, weight)
            img = (img+1)/2.0*255.0
            return img
    elif defence_name == 'quilting':
        import faiss
        dicter = dc.quilt_params
        patches_filename = dicter['patches_file'] 
        index_filename = dicter['index_file']
        with open(patches_filename, 'rb') as fread:
            patches = pickle.load(fread)
            patch_size = int(math.sqrt(patches.shape[1] / 3))
        faiss_index = faiss.read_index(index_filename)
        def func(img):
            im = quilting(
                img, faiss_index, patches,
                patch_size=patch_size,
                overlap=(patch_size // 2), 
                graphcut=True,
                k=dicter['quilting_neighbors'],
                random_stitch=dicter['quilting_random_stitch']
            )   
            # Clamping because some values are overflowing in quilting
            im = np.clip(im, 0,255)
            im = im.transpose(1,2,0)
            return im
    return func


def loader_func(network_name, sess, isotropic, size,pert,defence):
    def_func = defence_func(defence)
    if network_name == 'inceptionv3':
        def loader(image_name):
            im = v3_preprocess(image_name,pert_load, def_func=def_func, pert=pert)
            return im
    elif 'vgg' in network_name:
        def loader(image_name):
            im = vgg_preprocess(image_name, pert_load, def_func=def_func,pert=pert)
            return im
    else:
        img_tensor, image_path_tensor = load_image()
        processed_img = img_preprocess(
            img=img_tensor, isotropic=isotropic, crop=size)

        def loader(image_name, pert_load, pert=pert, processed_img=processed_img, image_path_tensor=image_path_tensor, sess=sess):
            im = sess.run([processed_img], feed_dict={
                          image_path_tensor: image_name})[0]
            if pert_load: im = np.clip(im+pert, 0,255)[0]
    # this is where the reform should take place, and this is where the pert should be added.         a
            #print(im.shape)
            im = def_func(im)
            cv2.imwrite('temp.png',im)
            mean=np.array([103.939, 116.779, 123.68])
            im = im - mean
            return im
    return loader


def get_params(net_name):
    isotropic = False
    if net_name == 'caffenet':
        size = 227
    elif net_name == 'inceptionv3':
        size = 299
    else:
        size = 224
        if not net_name == 'googlenet':
            isotropic = True
    return isotropic, size
