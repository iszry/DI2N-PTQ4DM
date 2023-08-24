#!/bin/bash     
export PYTHONPATH=".:guided-diffusion:improved-diffusion"

QUANT_FLAGS="--n_bits_w 8 --channel_wise --n_bits_a 8 --act_quant --order together --wwq --waq --awq --aaq \
--weight 0.01 --input_prob 0.5 --prob 0.5 --iters_w 1000 --calib_num_samples 128 --calib_block_size 128 \
--data_dir ./improved-diffusion/datasets/cifar_train --calib_im_mode noise_backward_t"
# --iters_w 100, default int 20000 --act_quant
MODEL_FLAGS=" --device cuda --image_size 32 --num_channels 128 --num_res_blocks 3 --learn_sigma True --dropout 0.3"
DIFFUSION_FLAGS="--diffusion_steps 4000 --timestep_respacing 250 --use_ddim True  \
--noise_schedule cosine "

BATCH_SIZE=100
NUM_SAMPLES=50000
# export CUDA_VISIBLE_DEVICES="7"

OUT_PATH="--out_path outputs/test.npz"
LOG_FILE="8192_64_uniform_log.txt"
CALIB_FLAGS="--calib_noise_mode fake_loop --calib_batch_size 128 --calib_t_mode uniform \
--calib_t_mode_normal_mean 0.4 --calib_t_mode_normal_std 0.4 --calib_sample_round 1 --test_mode only_calib "
# args.test_mode == "only_calib"
# srun -J AQU8192 -p normal -w irip-c3-compute-2 --gres=gpu -c 8 --mem 16G python improved-diffusion/scripts/quant_image_sample.py $OUT_PATH $CALIB_FLAGS $QUANT_FLAGS $MODEL_FLAGS --model_path improved-diffusion/download/cifar10_uncond_50M_500K.pt $DIFFUSION_FLAGS --num_samples $NUM_SAMPLES --batch_size $BATCH_SIZE   &
python improved-diffusion/scripts/quant_image_sample.py $OUT_PATH $CALIB_FLAGS $QUANT_FLAGS $MODEL_FLAGS --model_path improved-diffusion/download/cifar10_uncond_50M_500K.pt $DIFFUSION_FLAGS --num_samples $NUM_SAMPLES --batch_size $BATCH_SIZE   &
# python guided-diffusion/evaluations/evaluator.py outputs/cifar10_reference.npz $OUT_PATH &

# improved-diffusion/download/cifar10_uncond_50M_500K.pt
# improved-diffusion/download/imagenet64_uncond_100M_1500K.pt



# MODEL_FLAGS="--device cuda --image_size 32 --num_channels 128 --num_res_blocks 3 --learn_sigma True --dropout 0.3"
# DIFFUSION_FLAGS="--diffusion_steps 4000 --noise_schedule cosine"
#MODEL_FLAGS="--image_size 32 --num_channels 128 --num_res_blocks 3 --learn_sigma True --dropout 0.3"
#DIFFUSION_FLAGS="--diffusion_steps 4000 --timestep_respacing 250 --use_ddim False --noise_schedule cosine --device cuda"

#python improved-diffusion/scripts/quant_image_sample.py $CALIB_FLAGS $QUANT_FLAGS $MODEL_FLAGS --model_path improved-diffusion/download/cifar10_uncond_50M_500K.pt $DIFFUSION_FLAGS --num_samples $NUM_SAMPLES --batch_size $BATCH_SIZE &
#python improved-diffusion/scripts/quant_image_sample.py $CALIB_FLAGS $QUANT_FLAGS $MODEL_FLAGS --model_path /pretrained-model-path/imagenet64_uncond_100M_1500K.pt $DIFFUSION_FLAGS --num_samples $NUM_SAMPLES --batch_size $BATCH_SIZE &

