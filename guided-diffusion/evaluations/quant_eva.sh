# SAMPLE_FLAGS="..\\datasets\\samples_random_64x3.npz ..\\datasets\\samples_DNTC_64x3.npz"
SAMPLE_FLAGS="../../outputs/cifar10_reference.npz  ../../outputs/imagenet_quant_w8a8_all_5w_8192uniform_refineStructure_fakeLoop_1000iters.npz"
# raw_random samples_noise_backward_t_normal_64x3_128_16.npz  samples_raw_random_64x3_128_16.npz
# noise_backward_t_normal iddpm_imagenet64.npz samples_noise_backward_t_normal_64x3_1024_1.npz
# samples_noise_backward_t_normal_64x3_8192_1.npz samples_noise_backward_t_normal_64x3_1024_1.npz
# samples_random_random_64x3_1024_1.npz samples_noise_backward_t_normal_64x3_8192_1.npz
# samples_noise_backward_t_uniform_64x3_1024_1.npz

# srun -J eva -p normal -w irip-c3-compute-2 --gres=gpu -c 8 --mem 16G python evaluator.py $SAMPLE_FLAGS &
python evaluator.py $SAMPLE_FLAGS &