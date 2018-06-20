# Defence Against The Dark Arts

A repository containing evaluation of various universal adversarial perturbations against various defense mechanisms.

UAPs are quite a big threat to anyone using deep learning, be it muggles or wizards. They are indeed the perfect curse that someone might use against your Deep model. In this repository we evaluated some of the recent defence techniques for various UAPs. 

## The Attacks
The following 3 Universal Adversarial attacks were evaluated:

1) Universal Adversarial Perturbations: [Paper](https://arxiv.org/abs/1610.08401).

2) Generalizable Objective for Universal Adversarial Perturbations: [Paper](https://arxiv.org/abs/1801.08092).

3) Network for Adversarial generation: [Paper](https://arxiv.org/abs/1712.03390).

I Look forward to anyone willing to contribute more perturbations to test ( UAPs specifically).

## The Defenses

The following defenses were evaluated:

1) Prediction using multiple crops of input (Not implemented in this repo.)

2) Gaussian smoothing, Median smoothing, and Bilateral Filtering.

3) JPEG Compression: [Paper](https://arxiv.org/abs/1608.00853)

4) BIT compression: [Paper](https://arxiv.org/abs/1704.01155)

5) TV-minimization: [Paper](https://arxiv.org/abs/1711.00117)

6) Image-quilting: [Paper](https://arxiv.org/abs/1711.00117)

7) Perturbation rectification network: [Paper](https://arxiv.org/abs/1711.05929)

Of course, you can contribute your defenses. 

**Note**: Quilting is still to be properly integrated. Till then, the code provide by the authors can be used [here](https://github.com/facebookresearch/adversarial_image_defenses). For Perturbation Rectification Network, code provided by the authors [here]() can be used.

**Note**: The numbers reported in [this](https://arxiv.org/abs/1801.08092) are from using the various author provided code only(For defence 5, 6, and 7).

# Instructions

1) Firstly, there is a long list of things to be installed, (specially for `tvm` and `quilting`). Instead of paraphrasing it here, I would recommend the user follow the instructions given by the authors [here](https://github.com/facebookresearch/adversarial_image_defenses).

2) After installation, download the weights for the networks,

```
# uncomment as required in the sourced file
cd weights
source download_weights.sh
``` 

3) Now, download the perturbations. 
  * GD-UAP: [Link](https://www.dropbox.com/s/ixjzg4itx10nhid/perturbations.tar.gz?dl=0)
  * NAG: UAP can be generated from [here](https://github.com/val-iisc/nag.git) (link to crafted perturbations will be added soon.)
  * UAP: [Link](https://github.com/LTS4/universal.git). **Note**: The UAP provided in this repository is for a different googlenet instance.

4) Evaluating the defence use:

```
python evaluate.py --network googlenet --adv_im perturbations/GD_UAP_perts/best_fool_rate_googlenet_with_data_sat_diff_reg_0.0.npy --img_list utils/ilsvrc_test.txt --gt_labels utils/ilsvrc_test_gt.txt --batch_size 10 --defence tvm
```

The various defences can be used by changing the arguement for `defence` to `Gaussian`, `Median`, `Bilateral`, `Bit_Compression`, `JPEG`, `tvm` and `quilting`. Each defence can be configured in the `defence_config.py` file. (Look at `tensorflow-classification/misc/utils.py` for closer look at the defence code.)

5) For quilting you need to create patches. This can be done using the following code:
```
# all the parameters are specified in defence_config.py file
python index_patches.py
```
# TODO

* Finish Quilting
* Add the perturbations from NAG and UAP
* Add results from paper.



