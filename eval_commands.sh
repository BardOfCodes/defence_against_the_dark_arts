#CUDA_VISIBLE_DEVICES=1 python evaluate.py --network googlenet --adv_im perturbations/GD_UAP_perts/best_fool_rate_googlenet_with_data_sat_diff_reg_0.0.npy --img_list utils/ilsvrc_test.txt --gt_labels utils/ilsvrc_test_gt.txt --batch_size 10 --defence quilting
CUDA_VISIBLE_DEVICES=2 python index_patches.py --network googlenet  --img_list utils/ilsvrc_test.txt 
