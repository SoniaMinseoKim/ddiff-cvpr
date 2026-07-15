"""Build a 3-column comparison figure (measurement | reconstruction | ground truth)
from a ddiff_sample.py output folder.

Usage: python3 test_results/make_plot.py <output_dir> <task_label> <figure_path>
"""

import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

out_dir, task_label, fig_path = sys.argv[1], sys.argv[2], sys.argv[3]

meas = plt.imread(f"{out_dir}/temp_meas/meas_0.png")
recon = plt.imread(f"{out_dir}/temp_recon/recon_0.png")
gt = plt.imread(f"{out_dir}/temp_gt/gt_0.png")

fig, axes = plt.subplots(1, 3, figsize=(12, 4.5))
for ax, img, title in zip(axes,
                          [meas, recon, gt],
                          ["Measurement", "Reconstruction", "Ground Truth"]):
    ax.imshow(img)
    ax.set_title(title, fontsize=14, loc='center')
    ax.axis('off')

fig.suptitle(task_label, fontsize=16)
fig.tight_layout()
fig.savefig(fig_path, dpi=150, bbox_inches='tight')
print(f"Saved {fig_path}")
