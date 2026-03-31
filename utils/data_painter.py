

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import skimage

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torchvision.models._utils")


# ---- polar plot visualization ----
title_size = 30
metric_size = 18
font_prop_title = fm.FontProperties(size=title_size)
font_prop_metric = fm.FontProperties(size=metric_size)
cmap_temt = "jet"
shading_mode = "gouraud"


# plot GT vs predicted spectrum in polar coordinates
def paint_spectrum(gt_spectrum, pred_spectrum, save_path=None):
    H, W = gt_spectrum.shape

    psnr_val = skimage.metrics.peak_signal_noise_ratio(gt_spectrum, pred_spectrum, data_range=1.0)
    ssim_val = skimage.metrics.structural_similarity(gt_spectrum, pred_spectrum, data_range=1.0)

    if shading_mode == "gouraud":
        r = np.linspace(0, 1, H)
        theta = np.linspace(0, 2.0 * np.pi, W)
    else:
        r = np.linspace(0, 1, H + 1)
        theta = np.linspace(0, 2.0 * np.pi, W + 1)

    r, theta = np.meshgrid(r, theta)

    title_color = 'black'

    fig, axs = plt.subplots(1, 2, subplot_kw={'projection': 'polar'}, figsize=(6.4 * 2, 5.6))

    gt_data = np.flipud(gt_spectrum).T
    pred_data = np.flipud(pred_spectrum).T

    axs[0].grid(False)
    axs[0].pcolormesh(theta, r, gt_data, cmap=cmap_temt, shading=shading_mode)
    axs[0].axis('off')

    axs[1].grid(False)
    axs[1].pcolormesh(theta, r, pred_data, cmap=cmap_temt, shading=shading_mode)
    axs[1].axis('off')

    axs[0].text(0.5, -0.10, "GT", transform=axs[0].transAxes, ha='center', va='top',
                fontproperties=font_prop_title, color=title_color)

    axs[1].text(0.5, -0.10, "Synthesized", transform=axs[1].transAxes, ha='center', va='top',
                fontproperties=font_prop_title, color=title_color)

    axs[1].text(0.5, -0.20, f"PSNR: {psnr_val:.2f} dB  |  SSIM: {ssim_val:.4f}",
                transform=axs[1].transAxes, ha='center', va='top',
                fontproperties=font_prop_metric, color='dimgray')

    plt.tight_layout()

    plt.savefig(save_path, dpi=300, bbox_inches='tight', transparent=False)

    plt.close()


# ---- metric plots (bar, CDF, histogram) ----

def plot_metric_bar(values, metric_name, save_path):
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.bar(range(len(values)), values, color='steelblue', alpha=0.8, width=1.0)
    mean_val = np.mean(values)
    ax.axhline(y=mean_val, color='red', linestyle='--', linewidth=1.5, label=f'Mean: {mean_val:.4f}')
    ax.set_xlabel('Test Sample Index', fontsize=12)
    ax.set_ylabel(metric_name, fontsize=12)
    ax.set_title(f'{metric_name} per Test Sample (N={len(values)})', fontsize=14)
    ax.legend(fontsize=11)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()


def plot_metric_cdf(values, metric_name, save_path):
    sorted_vals = np.sort(values)
    cdf = np.arange(1, len(sorted_vals) + 1) / len(sorted_vals)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(sorted_vals, cdf, color='steelblue', linewidth=2)
    median_val = np.median(values)
    ax.axvline(x=median_val, color='red', linestyle='--', linewidth=1.5, label=f'Median: {median_val:.4f}')
    ax.set_xlabel(metric_name, fontsize=12)
    ax.set_ylabel('CDF', fontsize=12)
    ax.set_title(f'CDF of {metric_name} (N={len(values)})', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()


def plot_metric_histogram(values, metric_name, save_path, bins=50):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(values, bins=bins, color='steelblue', alpha=0.8)
    ax.axvline(np.mean(values), color='red', linestyle='--',
               label=f'Mean: {np.mean(values):.2f}')
    ax.set_xlabel(metric_name); ax.set_ylabel('Count')
    ax.set_title(f'{metric_name} Distribution (N={len(values)})'); ax.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=200); plt.close()
