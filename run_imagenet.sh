#!/bin/bash
# DDiff on the ImageNet 256x256 test set (hyperparameters from the paper, Appendix H, Table 6).
# Uncomment the task(s) you want to run.

# Gaussian deblurring
python3 ddiff_sample.py \
    --dataset imagenet \
    --testdata_path "datasets/test-imagenet" \
    --batch_size 100 \
    --output_path "results/ddiff_deconv_imagenet/" \
    --task deconv \
    --forward_step 50 \
    --scale 1.8

# # Super resolution (4x)
# python3 ddiff_sample.py \
#     --dataset imagenet \
#     --testdata_path "datasets/test-imagenet" \
#     --batch_size 100 \
#     --output_path "results/ddiff_superres_imagenet/" \
#     --task downsample \
#     --forward_step 1 \
#     --scale 18

# # Inpainting (128x128 box)
# python3 ddiff_sample.py \
#     --dataset imagenet \
#     --testdata_path "datasets/test-imagenet" \
#     --batch_size 100 \
#     --output_path "results/ddiff_inpaint_box_imagenet/" \
#     --task inpaint \
#     --mask_type box \
#     --forward_step 1 \
#     --scale 50

# # Inpainting (70% random mask)
# python3 ddiff_sample.py \
#     --dataset imagenet \
#     --testdata_path "datasets/test-imagenet" \
#     --batch_size 100 \
#     --output_path "results/ddiff_inpaint_random_imagenet/" \
#     --task inpaint \
#     --mask_type random \
#     --forward_step 1 \
#     --scale 50

# # Motion deblurring
# python3 ddiff_sample.py \
#     --dataset imagenet \
#     --testdata_path "datasets/test-imagenet" \
#     --batch_size 100 \
#     --output_path "results/ddiff_motion_deblur_imagenet/" \
#     --task motion_deblur \
#     --forward_step 80 \
#     --scale 1.5

# # Phase retrieval (best of 5 runs)
# python3 ddiff_sample.py \
#     --dataset imagenet \
#     --testdata_path "datasets/test-imagenet" \
#     --batch_size 100 \
#     --output_path "results/ddiff_phase_retrieval_imagenet/" \
#     --task phase_retrieval \
#     --forward_step 1 \
#     --scale 38 \
#     --num_runs 5

# # Nonlinear deblurring (requires bkse, see README)
# python3 ddiff_sample.py \
#     --dataset imagenet \
#     --testdata_path "datasets/test-imagenet" \
#     --batch_size 100 \
#     --output_path "results/ddiff_nonlinear_blur_imagenet/" \
#     --task nonlinear_blur \
#     --forward_step 120 \
#     --scale 2.5

# # High dynamic range (2x)
# python3 ddiff_sample.py \
#     --dataset imagenet \
#     --testdata_path "datasets/test-imagenet" \
#     --batch_size 100 \
#     --output_path "results/ddiff_hdr_imagenet/" \
#     --task hdr \
#     --forward_step 100 \
#     --scale 3.8
