<!-- ===== ORIGINAL =====
# ddiff-cvpr
[CVPR 2026] Dual Ascent Diffusion for Inverse Problems
===== END ORIGINAL ===== -->

# Dual Ascent Diffusion for Inverse Problems

<p align="center">
  <b>CVPR 2026</b>
</p>

<p align="center">
  <a href="https://soniaminseokim.github.io/">Minseo (Sonia) Kim</a><sup>1</sup>,
  <a href="https://axlevy.com/">Axel Levy</a><sup>1</sup>,
  <a href="https://stanford.edu/~gordonwz/">Gordon Wetzstein</a><sup>1</sup>
  <br>
  <sup>1</sup>Stanford University
</p>

<p align="center">
  <a href="https://soniaminseokim.github.io/ddiff/"><img src="https://img.shields.io/badge/Project-Page-blue" alt="Project Page"></a>
  <a href="https://www.arxiv.org/abs/2505.17353"><img src="https://img.shields.io/badge/arXiv-2505.17353-b31b1b.svg" alt="arXiv"></a>
  <a href="https://www.arxiv.org/pdf/2505.17353"><img src="https://img.shields.io/badge/Paper-PDF-green" alt="PDF"></a>
</p>

> **🚧 Code coming soon — the full implementation will be released shortly. Stay tuned!**

> **TL;DR:** We solve maximum-a-posteriori (MAP) inverse problems with diffusion-model priors using a **dual ascent optimization** framework, yielding higher-quality, faster, and more noise-robust reconstructions than the state of the art.

## Abstract

Ill-posed inverse problems are fundamental in many domains, ranging from astrophysics to medical imaging. Emerging diffusion models provide a powerful prior for solving these problems. Existing maximum-a-posteriori (MAP) or posterior sampling approaches, however, rely on different computational approximations, leading to inaccurate or suboptimal samples. To address this issue, we introduce a new approach to solving MAP problems with diffusion model priors using a dual ascent optimization framework. Our framework achieves better image quality as measured by various metrics for image restoration problems, it is more robust to high levels of measurement noise, it is faster, and it estimates solutions that represent the observations more faithfully than the state of the art.

## Links

- 📄 **Paper (arXiv):** https://www.arxiv.org/abs/2505.17353
- 📑 **PDF:** https://www.arxiv.org/pdf/2505.17353
- 🌐 **Project Page:** https://soniaminseokim.github.io/ddiff/

## Citation

If you find this work useful, please consider citing:

```bibtex
@inproceedings{kim2026dualascentdiffusion,
  title={Dual Ascent Diffusion for Inverse Problems},
  author={Minseo Kim and Axel Levy and Gordon Wetzstein},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2026},
  url={https://arxiv.org/abs/2505.17353},
}
```
