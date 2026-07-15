"""DDiff: Dual Ascent Diffusion for Inverse Problems (CVPR 2026).

Runs DDiff posterior optimization with a pretrained pixel-space diffusion
model on a folder of test images, and reports PSNR / SSIM / LPIPS.
"""

import argparse
import csv
import os

import torch
import yaml
import matplotlib.pyplot as plt
from functools import partial
from torch.nn import functional as F

from guided_diffusion.unet import create_model
from guided_diffusion.gaussian_diffusion import get_named_beta_schedule
from util.img_utils import clear_color
from util.logger import get_logger
from util.img_utils import Blurkernel
from util.fastmri_utils import fft2c_new
from util.resizer import Resizer

from PIL import Image
import numpy as np
from tqdm.auto import tqdm

from torchvision import transforms
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import lpips


parser = argparse.ArgumentParser(description='DDiff posterior sampling')
parser.add_argument('--testdata_path', type=str, default="datasets/test-ffhq")
parser.add_argument('--output_path', type=str, default="results/ddiff_deconv_ffhq/")
parser.add_argument('--dataset', type=str, default='ffhq', choices=['ffhq', 'imagenet'],
                    help='Which pretrained diffusion model / preprocessing to use')
parser.add_argument('--model_path', type=str, default=None,
                    help='Diffusion checkpoint. Defaults to models/ffhq_10m.pt or models/imagenet256.pt')
parser.add_argument('--batch_size', type=int, default=100,
                    help='Number of test images to process')
parser.add_argument('--task', type=str, default='deconv',
                    choices=['deconv', 'downsample', 'inpaint', 'motion_deblur',
                             'phase_retrieval', 'nonlinear_blur', 'hdr'])
parser.add_argument('--method', type=str, default='admm_ddim',
                    help='admm_ddim is DDiff; hqs_ddim is the DDiff-HQS ablation')
parser.add_argument('--mask_type', type=str, default='box', choices=['box', 'random'],
                    help='Mask type for the inpainting task')
parser.add_argument('--forward_step', type=int, default=1,
                    help='Noise threshold t0: timestep below which the reverse step is deterministic')
parser.add_argument('--scale', type=float, default=2.9,
                    help='Measurement step size gamma_0')
parser.add_argument('--num_runs', type=int, default=1,
                    help='Number of runs per image; the best-PSNR sample is kept (used for phase retrieval)')
parser.add_argument('--gpu', type=int, default=0)
args = parser.parse_args()


def fft2_m(x):
  """ FFT for multi-coil """
  if not torch.is_complex(x):
      x = x.type(torch.complex64)
  return torch.view_as_complex(fft2c_new(torch.view_as_real(x)))


logger = get_logger()
device_str = f"cuda:{args.gpu}" if torch.cuda.is_available() else 'cpu'
logger.info(f"Device set to {device_str}.")
device = torch.device(device_str)

if args.dataset == 'ffhq':
    model = create_model(
        image_size=256,
        num_channels=128,
        num_res_blocks=1,
        channel_mult="",
        learn_sigma=True,
        class_cond=False,
        use_checkpoint=False,
        attention_resolutions="16",
        num_heads=4,
        num_head_channels=64,
        num_heads_upsample=-1,
        use_scale_shift_norm=True,
        dropout=0,
        resblock_updown=True,
        use_fp16=False,
        use_new_attention_order=False,
        model_path=args.model_path or 'models/ffhq_10m.pt'
    )
else:  # imagenet
    model = create_model(
        image_size=256,
        num_channels=256,
        num_res_blocks=2,
        channel_mult="",
        learn_sigma=True,
        class_cond=False,
        use_checkpoint=False,
        attention_resolutions="32,16,8",
        num_heads=4,
        num_head_channels=64,
        num_heads_upsample=-1,
        use_scale_shift_norm=True,
        dropout=0,
        resblock_updown=True,
        use_fp16=False,
        use_new_attention_order=False,
        model_path=args.model_path or 'models/imagenet256.pt'
    )

model = model.to(device)
model.eval()

num_timesteps = 1000
betas = get_named_beta_schedule(schedule_name="linear", num_diffusion_timesteps=num_timesteps)

alphas = 1.0 - betas
alphas_cumprod = np.cumprod(alphas, axis=0)
alphas_cumprod_prev = np.append(1.0, alphas_cumprod[:-1])

sqrt_recip_alphas_cumprod = np.sqrt(1.0 / alphas_cumprod)
sqrt_recipm1_alphas_cumprod = (1.0 - alphas_cumprod) / np.sqrt(alphas_cumprod)

posterior_mean_coef1 = betas * np.sqrt(alphas_cumprod_prev) / (1.0-alphas_cumprod)
posterior_mean_coef2 = (1.0 - alphas_cumprod_prev) * np.sqrt(alphas) / (1.0 - alphas_cumprod)

posterior_variance = (
    betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
)
posterior_log_variance_clipped = np.log(
    np.append(posterior_variance[1], posterior_variance[1:])
)

def extract_and_expand(array, time, target):
    array = torch.from_numpy(array).to(target.device)[time].float()
    while array.ndim < target.ndim:
        array = array.unsqueeze(-1)
    return array.expand_as(target)

def get_variance(x, t):
    model_var_values = x
    min_log = posterior_log_variance_clipped
    max_log = np.log(betas)

    min_log = extract_and_expand(min_log, t, x)
    max_log = extract_and_expand(max_log, t, x)

    # The model_var_values is [-1, 1] for [min_var, max_var]
    frac = (model_var_values + 1.0) / 2.0
    model_log_variance = frac * max_log + (1-frac) * min_log
    return model_log_variance

def process_xstart(x):
    x = x.clamp(-1, 1)
    return x

def predict_xstart(x_t, t, eps):
    coef1 = extract_and_expand(sqrt_recip_alphas_cumprod, t, x_t)
    coef2 = extract_and_expand(sqrt_recipm1_alphas_cumprod, t, eps)
    score = - eps / np.sqrt(1.0 - alphas_cumprod[t])
    return coef1 * x_t + coef2 * score

def q_posterior_mean(x_start, x_t, t):
    """
    Compute the mean of the diffusion posterior:
        q(x_{t-1} | x_t, x_0)
    """
    assert x_start.shape == x_t.shape
    coef1 = extract_and_expand(posterior_mean_coef1, t, x_start)
    coef2 = extract_and_expand(posterior_mean_coef2, t, x_t)
    return coef1 * x_start + coef2 * x_t

def get_mean_and_xstart(x, t, model_output):
    pred_xstart = process_xstart(predict_xstart(x, t, model_output))
    mean = q_posterior_mean(pred_xstart, x, t)
    return mean, pred_xstart

def p_sample(model, x, t):
    model_output = model(x, t)

    model_output, model_var_values = torch.split(model_output, x.shape[1], dim=1)

    x_t, x_0_hat = get_mean_and_xstart(x, t, model_output)
    noise = torch.randn_like(x)
    model_log_variance = get_variance(model_var_values, t)

    if t != 0:  # no noise when t == 0
        x_t += torch.exp(0.5 * model_log_variance) * noise

    score = -model_output / np.sqrt(1.0 - alphas_cumprod[t])

    return {'sample': x_t, 'pred_xstart': x_0_hat, 'score': score}

sigma = 0.05  # measurement noise standard deviation
def noiser(data):
    return data + torch.randn_like(data, device=data.device) * sigma

def mask_gen(data, mask_type, image_size=256, margin=(16, 16)):
    mask = torch.ones_like(data)

    if mask_type == 'box':
        # DAPS-style random square box mask, with specified size range
        np.random.seed(0)
        mask_len_range = (128, 129)
        B, C, H, W = data.shape
        l, h = mask_len_range
        mask_h = np.random.randint(l, h)  # result: always 128
        mask_w = mask_h  # typically square
        margin_height, margin_width = margin
        maxt = image_size - margin_height - mask_h
        maxl = image_size - margin_width - mask_w
        t = np.random.randint(margin_height, maxt)
        lft = np.random.randint(margin_width, maxl)
        mask[..., t:t+mask_h, lft:lft+mask_w] = 0

    elif mask_type == 'random':
        # DAPS-style random pixel mask, with specified prob range
        np.random.seed(0)
        mask_prob_range = (0.70, 0.71)
        total = image_size * image_size
        prob = np.random.uniform(*mask_prob_range)   # e.g. ~0.705
        num_pixels_to_mask = int(prob * total)
        for i in range(data.shape[0]):
            mask_indices = np.random.choice(total, num_pixels_to_mask, replace=False)
            for c in range(data.shape[1]):
                mask_1d = mask[i, c].view(-1)
                mask_1d[mask_indices] = 0
                mask[i, c] = mask_1d.view(image_size, image_size)

    return mask

def forward(data, task, mask_type):
    if task=='inpaint':
      mask = mask_gen(data, mask_type)
      return data * mask.to(device), mask

    elif task == 'downsample':
        # Get input shape (B, C, H, W)
        in_shape = list(data.shape)
        # Use scale_factor 4 as specified
        scale_factor = 4
        resizer = Resizer(in_shape, 1 / scale_factor).to(device)
        downsampled = resizer(data)
        return downsampled, resizer

    elif task=='deconv':
      conv = Blurkernel(blur_type='gaussian',
                        kernel_size=61,
                        std=3.0,
                        device=device).to(device)
      return conv(data), conv

    elif task=='motion_deblur':
        np.random.seed(0)
        conv = Blurkernel(blur_type='motion',
                            kernel_size=61,
                            std=0.5,
                            device=device).to(device)
        return conv(data), conv

    elif task=='phase_retrieval':
      oversample = 2.0
      pad = int((oversample / 8.0) * 256)
      padded = F.pad(data, (pad, pad, pad, pad))
      amplitude = fft2_m(padded).abs()
      return amplitude, padded # padded is just placeholder for mask

    elif task=='hdr':
        scale = 2
        return torch.clip((data * scale), -1, 1), scale # placeholder for mask

    elif task=='nonlinear_blur':
        '''
        Nonlinear deblur requires external codes (bkse).
        '''
        from bkse.models.kernel_encoding.kernel_wizard import KernelWizard

        opt_yml_path = 'bkse/options/generate_blur/default.yml'
        with open(opt_yml_path, "r") as f:
            opt = yaml.safe_load(f)["KernelWizard"]
            model_path = opt["pretrained"]
        blur_model = KernelWizard(opt)
        blur_model.eval()
        blur_model.load_state_dict(torch.load('bkse/' + model_path))
        blur_model = blur_model.to(device)

        # Create a specific generator with its own seed
        generator = torch.Generator()
        generator.manual_seed(0)

        random_kernel = torch.randn(1, 512, 2, 2, generator=generator).to(device) * 1.2
        data = (data + 1.0) / 2.0  #[-1, 1] -> [0, 1]
        blurred = blur_model.adaptKernel(data, kernel=random_kernel)
        blurred = (blurred * 2.0 - 1.0).clamp(-1, 1) #[0, 1] -> [-1, 1]

        return blurred, {'blur_model': blur_model, 'random_kernel': random_kernel} # blur_model is just placeholder for mask

def forward_grad_like(data, task, mask):
    if task=='inpaint':
      return data * mask.to(device)

    elif task == 'downsample' or task=='deconv' or task=='motion_deblur':
      return mask(data)

    elif task=='phase_retrieval':
      oversample = 2.0
      pad = int((oversample / 8.0) * 256)
      padded = F.pad(data, (pad, pad, pad, pad))
      amplitude = fft2_m(padded).abs()
      return amplitude

    elif task=='hdr':
        return torch.clip((data * mask), -1, 1)

    elif task=='nonlinear_blur':
        data = (data + 1.0) / 2.0  #[-1, 1] -> [0, 1]
        blurred = mask['blur_model'].adaptKernel(data, kernel=mask['random_kernel'])
        blurred = (blurred * 2.0 - 1.0).clamp(-1, 1) #[0, 1] -> [-1, 1]

        return blurred

def grad_likelihood(x_prev, x_0_hat, measurement, task, mask):
    difference = measurement - forward_grad_like(x_0_hat, task, mask)
    norm = torch.linalg.norm(difference)
    neg_log_likelihood = norm**2 / (2*sigma**2)
    grad = torch.autograd.grad(outputs=neg_log_likelihood, inputs=x_prev)[0]
    return grad

def conditioning(x_prev, x_t, x_0_hat, measurement, task, method, mask, scale, anneal_factor=1):
    grad = grad_likelihood(x_prev=x_prev, x_0_hat=x_0_hat, measurement=measurement, task=task, mask=mask)
    if method=='dps':
      x_t -= grad * (scale / torch.linalg.norm(grad))
    elif method=='admm_ddim':
      x_t -= grad * (scale / torch.linalg.norm(grad)) * anneal_factor
    elif method=='hqs_ddim':
      scale = 90
      x_t -= grad * (scale / torch.linalg.norm(grad))
    return x_t


def p_sample_loop(model, x_start, measurement, measurement_cond_fn, task, method, mask, scale, forward_step, record, save_root):

    img = x_start

    if method=='admm_ddim' or method=='hqs_ddim':
      seq = list(range(num_timesteps))[::-1]
      pbar = tqdm(seq)

      # Create a mapping from timesteps to their previous timestep in the sequence
      prev_t = {}
      for i in range(len(seq) - 1):
          prev_t[seq[i]] = seq[i + 1]
      prev_t[seq[-1]] = -1  # Handle the last timestep

      u_t = torch.zeros_like(img)  # Initialize Lagrangian multiplier
      threshold = 1e-4  # Early stopping threshold
    else:
      pbar = tqdm(list(range(num_timesteps))[::-1])

    device = x_start.device

    for idx in pbar:
        time = torch.tensor([idx] * img.shape[0], device=device)

        if method=='uncond':
          img = img.requires_grad_()
          out = p_sample(x=img, t=time, model=model)
          img = out['sample']

        elif method=='dps':
          img = img.requires_grad_()
          out = p_sample(x=img, t=time, model=model)
          img = measurement_cond_fn(x_t=out['sample'],
                          measurement=measurement,
                          x_prev=img,
                          x_0_hat=out['pred_xstart'],
                          task=task,
                          method=method,
                          mask=mask,
                          scale=scale)

        elif method=='admm_ddim':
            img = img.requires_grad_()
            # Step 1: Compute x_0^(t) using the score network (z-update, Eq. 11)
            alpha_bar_t = extract_and_expand(alphas_cumprod, time, img)
            out = p_sample(x=img, t=time, model=model)
            x_0_pred = out['pred_xstart']  # This is x_0^(t) in the algorithm

            # Step-down policy on the measurement step size (Appendix H: a=3.3, b=0.1, t_gamma=90)
            anneal_factor = 3.3 if idx > 90 else 0.1

            # Step 2: Measurement step on x_0^(t) - u (x-update, Eq. 10)
            x_u_input = x_0_pred-u_t
            x_0_hat = measurement_cond_fn(x_t=x_u_input,
                measurement=measurement,
                x_prev=x_u_input,
                x_0_hat=x_u_input,
                task=task,
                method=method,
                mask=mask,
                scale=scale,
                anneal_factor=anneal_factor)

            with torch.no_grad():
                # Step 3: Compute implied noise
                eps_hat = (img - torch.sqrt(alpha_bar_t) * x_0_hat) / torch.sqrt(1 - alpha_bar_t)

                # Step 4: Check early stopping condition
                relative_error = torch.norm(x_0_hat - x_0_pred) / torch.norm(x_0_pred)
                pbar.set_description(f"Relative error: {relative_error.item():.6f}")

                prev_time = prev_t[idx]

                if relative_error < threshold and idx < prev_time:
                    print(f"Early stopping at iteration {idx} with relative error {relative_error.item():.6f}")
                    break

                # Step 5: Update for next iteration (reverse diffusion, Eq. 12)
                if idx > 0:  # Avoid going out of bounds
                    alpha_bar_prev = extract_and_expand(alphas_cumprod, prev_time, img)
                    sigma_t = torch.sqrt((1 - alpha_bar_prev) / (1 - alpha_bar_t) * (1 - alpha_bar_t/alpha_bar_prev))
                    noise = torch.randn_like(img) if idx > forward_step else 0  # Lower noise at end

                    if task == "phase_retrieval" or task=="inpaint" or task=="downsample":
                        # DDPM update (sigma_t = sqrt(1 - alpha_bar_prev), see Appendix H)
                        img = torch.sqrt(alpha_bar_prev) * (x_0_hat + u_t) + torch.sqrt(1 - alpha_bar_prev) * noise
                    else:
                        # DDIM update
                        img = torch.sqrt(alpha_bar_prev) * (x_0_hat + u_t) + \
                            torch.sqrt(1 - alpha_bar_prev - sigma_t**2) * eps_hat + \
                            sigma_t * noise

                    # Update Lagrangian multiplier (dual update, Eq. 6)
                    u_t = u_t + x_0_hat - x_0_pred

                # Clear cache after each step
                torch.cuda.empty_cache()

        elif method=='hqs_ddim':
            img = img.requires_grad_()
            # Step 1: Compute x_0^(t) using the score network
            alpha_bar_t = extract_and_expand(alphas_cumprod, time, img)
            out = p_sample(x=img, t=time, model=model)
            x_0_pred = out['pred_xstart']  # This is x_0^(t) in the algorithm

            x_u_input = x_0_pred
            x_0_hat = measurement_cond_fn(x_t=x_u_input,
                measurement=measurement,
                x_prev=x_u_input,
                x_0_hat=x_u_input,
                task=task,
                method=method,
                mask=mask,
                anneal_factor=alpha_bar_t)

            with torch.no_grad():
                # Step 3: Compute implied noise
                eps_hat = (img - torch.sqrt(alpha_bar_t) * x_0_hat) / torch.sqrt(1 - alpha_bar_t)

                # Step 4: Check early stopping condition
                relative_error = torch.norm(x_0_hat - x_0_pred) / torch.norm(x_0_pred)
                pbar.set_description(f"Relative error: {relative_error.item():.6f}")

                prev_time = prev_t[idx]
                if relative_error < threshold and idx < prev_time:
                    print(f"Early stopping at iteration {idx} with relative error {relative_error.item():.6f}")
                    break

                # Step 5: Update for next iteration
                if idx > 0:  # Avoid going out of bounds
                    alpha_bar_prev = extract_and_expand(alphas_cumprod, prev_time, img)
                    sigma_t = torch.sqrt((1 - alpha_bar_prev) / (1 - alpha_bar_t) * (1 - alpha_bar_t/alpha_bar_prev))

                    # DDIM update
                    noise = torch.randn_like(img) if idx > 1 else 0  # Lower noise at end
                    img = torch.sqrt(alpha_bar_prev) * (x_0_hat) + \
                        torch.sqrt(1 - alpha_bar_prev - sigma_t**2) * eps_hat + \
                        sigma_t * noise

                # Clear cache after each step
                torch.cuda.empty_cache()

        img = img.detach_()

        if record:
            if idx % 10 == 0:
                file_path = os.path.join(save_root, f"progress/x_{str(idx).zfill(4)}.png")
                plt.imsave(file_path, clear_color(img))

    return img

################ evaluation over the test set ####################

# Load Data
if args.dataset == 'ffhq':
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.CenterCrop((256, 256)),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
else:  # imagenet
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(256),
        transforms.CenterCrop(256),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

# Directory settings
input_dir = args.testdata_path
out_path = args.output_path
os.makedirs(out_path, exist_ok=True)

# Initialize metrics arrays
psnr_values = []
lpips_values = []
ssim_values = []
best_psnr = 0
best_sample = None
best_fname = None

# Initialize LPIPS loss function
loss_fn = lpips.LPIPS(net='alex').to(device)

# Get all image files
image_files = sorted([f for f in os.listdir(input_dir)
                      if f.lower().endswith(('.png', '.jpg', '.jpeg'))])

# Ensure we have images to process
if not image_files:
    print(f"No image files found in {input_dir}")
    exit(1)

# Process only the first batch_size images if there are more
image_files = image_files[:args.batch_size]
print(f"Found {len(image_files)} images to process")

# Create directories for ground truth / reconstructions / measurements
gt_dir = os.path.join(out_path, "temp_gt")
recon_dir = os.path.join(out_path, "temp_recon")
meas_dir = os.path.join(out_path, "temp_meas")
os.makedirs(gt_dir, exist_ok=True)
os.makedirs(recon_dir, exist_ok=True)
os.makedirs(meas_dir, exist_ok=True)

# Choose task and method
task = args.task
method = args.method
mask_type = args.mask_type

for idx, fname in enumerate(image_files):
    print(f"Processing image {idx+1}/{len(image_files)}: {fname}")

    # Load image
    img_path = os.path.join(input_dir, fname)
    img = Image.open(img_path).convert('RGB')
    img = transform(img)

    # Inference
    ref_img = img.to(device).unsqueeze(dim=0)

    sample_fn = partial(p_sample_loop, model=model, measurement_cond_fn=conditioning,
                      task=task, method=method, scale=args.scale, forward_step=args.forward_step)

    # Forward measurement model (done once per image)
    y, mask = forward(ref_img, task, mask_type)
    y_n = noiser(y)

    # Save the noisy measurement for visualization (log scale for Fourier amplitudes)
    meas_vis = torch.log(1 + y_n.abs()) if task == 'phase_retrieval' else y_n
    plt.imsave(os.path.join(meas_dir, f"meas_{idx}.png"), clear_color(meas_vis))

    # --- Run sampling num_runs times and keep the best PSNR ---
    best_psnr_image = -float('inf')
    best_sample_image = None
    best_sample_np = None

    for run_idx in range(args.num_runs):
        # Sampling (rewind randomness by making new x_start)
        x_start = torch.randn(ref_img.shape, device=device).requires_grad_()
        sample = sample_fn(x_start=x_start, measurement=y_n, mask=mask, record=False, save_root=out_path)

        # Convert tensors to numpy for metric calculation
        ref_img_np = ref_img.squeeze(0).permute(1, 2, 0).cpu().numpy()
        sample_np = sample.squeeze(0).permute(1, 2, 0).cpu().numpy()

        # Calculate PSNR
        psnr_value = psnr(ref_img_np, sample_np, data_range=ref_img_np.max() - ref_img_np.min())

        # Track best for this image
        if psnr_value > best_psnr_image:
            best_psnr_image = psnr_value
            best_sample_image = sample
            best_sample_np = sample_np

    # Compute additional metrics on the best sample
    lpips_value = loss_fn.forward(ref_img, best_sample_image).item() if best_sample_image is not None else 0
    ssim_value = ssim(ref_img_np, best_sample_np,
                      channel_axis=2,
                      data_range=ref_img_np.max() - ref_img_np.min(),
                      win_size=5)

    # Save ground truth and reconstruction
    plt.imsave(os.path.join(gt_dir, f"gt_{idx}.png"), clear_color(ref_img))
    plt.imsave(os.path.join(recon_dir, f"recon_{idx}.png"), clear_color(best_sample_image))

    # Store metrics
    psnr_values.append(best_psnr_image)
    lpips_values.append(lpips_value)
    ssim_values.append(ssim_value)

    # Track overall best PSNR (across images)
    if best_psnr_image > best_psnr:
        best_psnr = best_psnr_image
        best_sample = best_sample_image
        best_fname = fname

# Calculate average metrics
avg_psnr = np.mean(psnr_values)
avg_lpips = np.mean(lpips_values)
avg_ssim = np.mean(ssim_values)

# Save metrics to csv
metrics_csv_path = os.path.join(os.path.dirname(recon_dir), 'image_quality_metrics.csv')

try:
    with open(metrics_csv_path, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)

        # Write header row with all metrics
        csv_writer.writerow(['Image', 'PSNR', 'SSIM', 'LPIPS'])

        # Write individual image metrics
        for i, fname in enumerate(image_files):
            base_fname = os.path.basename(fname)
            csv_writer.writerow([
                base_fname,
                f"{psnr_values[i]:.4f}",
                f"{ssim_values[i]:.4f}",
                f"{lpips_values[i]:.6f}"
            ])

        # Add empty row for readability
        csv_writer.writerow([])

        # Add summary statistics
        csv_writer.writerow(['Average',
                            f"{np.mean(psnr_values):.4f}",
                            f"{np.mean(ssim_values):.4f}",
                            f"{np.mean(lpips_values):.6f}"])

        csv_writer.writerow(['Std Dev',
                            f"{np.std(psnr_values):.4f}",
                            f"{np.std(ssim_values):.4f}",
                            f"{np.std(lpips_values):.6f}"])

        csv_writer.writerow(['Min',
                            f"{np.min(psnr_values):.4f}",
                            f"{np.min(ssim_values):.4f}",
                            f"{np.min(lpips_values):.6f}"])

        csv_writer.writerow(['Max',
                            f"{np.max(psnr_values):.4f}",
                            f"{np.max(ssim_values):.4f}",
                            f"{np.max(lpips_values):.6f}"])


    print(f"All metrics successfully saved to {metrics_csv_path}")

except Exception as e:
    print(f"Error saving metrics to CSV: {e}")


# Print average metrics
print(f"Average PSNR: {avg_psnr:.2f}")
print(f"Average LPIPS: {avg_lpips:.4f}")
print(f"Average SSIM: {avg_ssim:.4f}")

# Save only the best reconstruction
print(f"Best PSNR: {best_psnr:.2f} (Image: {best_fname})")
plt.imsave(os.path.join(out_path, f'best_recon_{task}_{method}.png'), clear_color(best_sample))
