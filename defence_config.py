# in this file we will specify the hyperparams for the defences
import sys
faiss_loc = '/data1/aditya/faiss/python'
sys.path.insert(0,faiss_loc)

quilt_params = {'num_patches': 20000, 
                'quilting_patch_size': 5,
                'network':'googlenet',
                'patches_file':'temp.pkl',
                'img_list': 'utils/ilsvrc_test.txt',
                'index_file':'temp.faiss',
                'pca_dims':64,
                'quilting_neighbors': 1,
                'quilting_random_stitch':False}

defence_params = {}

## for gaussian
gaussian_params = {'kernel_size' : (5,5) , 'sigma' : 0}
defence_params['gaussian'] = gaussian_params

## for median
median_params = {'kernel_size': 5}
defence_params['median'] =  median_params

## for bilateral
bilateral_params = {'diameter': 9 , 'color_sigma': 75, 'space_sigma': 75}
defence_params['bilateral'] = bilateral_params

## for bit_compression
bit_com_params = {'quant': 32}
defence_params['bit'] = bit_com_params
## for jpeg:
jpeg_params = {'quality':50, 'temp_name':'temp.jpg'}
defence_params['jpeg'] = jpeg_params

## for tvm
tvm_params = {'pixel_drop_rate':0.5,'tvm_method':'bregman', 'tvm_weight':0.03}
defence_params['tvm'] = tvm_params
## for quilting
