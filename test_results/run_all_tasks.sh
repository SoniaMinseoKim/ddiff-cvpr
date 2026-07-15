#!/bin/bash
# Run DDiff on one FFHQ test image for all 8 tasks (paper Appendix H, Table 5
# hyperparameters) and build a measurement | reconstruction | ground-truth
# comparison figure for each task under test_results/.
set -u
cd "$(dirname "$0")/.."   # repo root

GPU=${GPU:-0}

run_task () {
    local name=$1; local label=$2; shift 2
    echo "=== [$(date)] Starting task: $name ==="
    python3 ddiff_sample.py \
        --dataset ffhq \
        --testdata_path "test_results/input" \
        --batch_size 1 \
        --gpu "$GPU" \
        --output_path "test_results/$name/" \
        "$@" \
    && python3 test_results/make_plot.py "test_results/$name" "$label" "test_results/${name}_comparison.png" \
    || echo "!!! Task $name FAILED"
    echo "=== [$(date)] Finished task: $name ==="
}

run_task deconv          "Gaussian Deblurring"        --task deconv --forward_step 50 --scale 2.9
run_task superres        "Super Resolution 4x"        --task downsample --forward_step 1 --scale 18
run_task inpaint_box     "Inpainting (Box)"           --task inpaint --mask_type box --forward_step 1 --scale 30
run_task inpaint_random  "Inpainting (Random)"        --task inpaint --mask_type random --forward_step 1 --scale 50
run_task motion_deblur   "Motion Deblurring"          --task motion_deblur --forward_step 80 --scale 2.9
run_task phase_retrieval "Phase Retrieval"            --task phase_retrieval --forward_step 1 --scale 38 --num_runs 5
run_task nonlinear_blur  "Nonlinear Deblurring"       --task nonlinear_blur --forward_step 120 --scale 2.5
run_task hdr             "High Dynamic Range"         --task hdr --forward_step 120 --scale 3.5

echo "=== [$(date)] ALL TASKS DONE ==="
