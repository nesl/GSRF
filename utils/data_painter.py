# -*- coding: utf-8 -*-

"""
painter for data
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torchvision.models._utils")


title_size = 36
font_prop_title = fm.FontProperties(size=title_size)
cmap_temt = "viridis"  # viridis, jet
sharding_temt = "flat"

def paint_spectrum(gt_spectrum, pred_spectrum_nerf, save_path=None):
    # create a polar grid
    r = np.linspace(0, 1, 91) # change this depending on your radial distance
    theta = np.linspace(0, 2.0 * np.pi, 361)

    r, theta = np.meshgrid(r, theta)

    title_color = 'black'

    fig, axs = plt.subplots(1, 2, subplot_kw={'projection': 'polar'}, figsize=(6.4 * 2, 4.8))

    axs[0].grid(False) 
    cax1 = axs[0].pcolormesh(theta, r, np.flipud(gt_spectrum).T, cmap=cmap_temt, shading=sharding_temt)
    axs[0].axis('off')

    axs[1].grid(False)
    cax2 = axs[1].pcolormesh(theta, r, np.flipud(pred_spectrum_nerf).T, cmap=cmap_temt, shading=sharding_temt)
    axs[1].axis('off')

    titles = ["GT", "Test"]

    for ax, title in zip(axs[1:], titles[1:]):
        ax.text(0.5, -0.15, title, transform=ax.transAxes, ha='center', va='top', fontproperties=font_prop_title, color=title_color)
        
    axs[0].text(0.5, -0.15, titles[0], transform=axs[0].transAxes, ha='center', va='top', fontproperties=font_prop_title, color=title_color)

    plt.tight_layout()

    plt.savefig(save_path, dpi=300, bbox_inches='tight', transparent=False)

    plt.close()




