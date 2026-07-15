#!/bin/bash
# DDiff on the FFHQ 256x256 test set (hyperparameters from the paper, Appendix H, Table 5).
# Uncomment the task(s) you want to run.

# Gaussian deblurring
python3 ddiff_sample.py \
    --dataset ffhq \
    --testdata_path "datasets/test-ffhq" \
    --batch_size 100 \
    --output_path "results/ddiff_deconv_ffhq/" \
    --task deconv \
    --forward_step 50 \
    --scale 2.9

# # Super resolution (4x)
# python3 ddiff_sample.py \
#     --dataset ffhq \
#     --testdata_path "datasets/test-ffhq" \
#     --batch_size 100 \
#     --output_path "results/ddiff_superres_ffhq/" \
#     --task downsample \
#     --forward_step 1 \
#     --scale 18

# # Inpainting (128x128 box)
# python3 ddiff_sample.py \
#     --dataset ffhq \
#     --testdata_path "datasets/test-ffhq" \
#     --batch_size 100 \
#     --output_path "results/ddiff_inpaint_box_ffhq/" \
#     --task inpaint \
#     --mask_type box \
#     --forward_step 1 \
#     --scale 30

# # Inpainting (70% random mask)
# python3 ddiff_sample.py \
#     --dataset ffhq \
#     --testdata_path "datasets/test-ffhq" \
#     --batch_size 100 \
#     --output_path "results/ddiff_inpaint_random_ffhq/" \
#     --task inpaint \
#     --mask_type random \
#     --forward_step 1 \
#     --scale 50

# # Motion deblurring
# python3 ddiff_sample.py \
#     --dataset ffhq \
#     --testdata_path "datasets/test-ffhq" \
#     --batch_size 100 \
#     --output_path "results/ddiff_motion_deblur_ffhq/" \
#     --task motion_deblur \
#     --forward_step 80 \
#     --scale 2.9

# # Phase retrieval (best of 5 runs)
# python3 ddiff_sample.py \
#     --dataset ffhq \
#     --testdata_path "datasets/test-ffhq" \
#     --batch_size 100 \
#     --output_path "results/ddiff_phase_retrieval_ffhq/" \
#     --task phase_retrieval \
#     --forward_step 1 \
#     --scale 38 \
#     --num_runs 5

# # Nonlinear deblurring (requires bkse, see README)
# python3 ddiff_sample.py \
#     --dataset ffhq \
#     --testdata_path "datasets/test-ffhq" \
#     --batch_size 100 \
#     --output_path "results/ddiff_nonlinear_blur_ffhq/" \
#     --task nonlinear_blur \
#     --forward_step 120 \
#     --scale 2.5

# # High dynamic range (2x)
# python3 ddiff_sample.py \
#     --dataset ffhq \
#     --testdata_path "datasets/test-ffhq" \
#     --batch_size 100 \
#     --output_path "results/ddiff_hdr_ffhq/" \
#     --task hdr \
#     --forward_step 120 \
#     --scale 3.5
