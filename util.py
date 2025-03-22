import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

def calculate_metrics(pred, target):
    """
    Calculate PSNR and SSIM between prediction and target
    """
    # Convert tensors to numpy arrays
    if torch.is_tensor(pred):
        pred = pred.detach().cpu().numpy()
    if torch.is_tensor(target):
        target = target.detach().cpu().numpy()
    
    # Calculate metrics for batch
    psnr_vals = []
    ssim_vals = []
    
    for i in range(pred.shape[0]):
        # Get single image from batch
        p = pred[i, 0]  # Assuming single channel
        t = target[i, 0]
        
        # Normalize data to 0-1 range for metrics if needed
        p_min, p_max = p.min(), p.max()
        t_min, t_max = t.min(), t.max()
        
        if p_max > p_min:
            p = (p - p_min) / (p_max - p_min)
        if t_max > t_min:
            t = (t - t_min) / (t_max - t_min)
        
        # Calculate metrics
        psnr_val = psnr(t, p, data_range=1.0)
        ssim_val = ssim(t, p, data_range=1.0)
        
        psnr_vals.append(psnr_val)
        ssim_vals.append(ssim_val)
    
    return np.mean(psnr_vals), np.mean(ssim_vals)

def save_visualizations(incomplete, outputs, complete, filepath, title="Visualization"):
    """Helper function to save visualizations"""
    # Create a figure to visualize results
    fig, axes = plt.subplots(min(4, len(incomplete)), 3, figsize=(16, 15), gridspec_kw={"width_ratios": [1, 1, 1.0675]})
    
    # Handle the case where there's only one sample
    if min(4, len(incomplete)) == 1:
        axes = axes.reshape(1, 3)
    
    for i in range(min(4, len(incomplete))):
        # Get the images
        input_img = incomplete[i, 0].cpu().numpy()
        output_img = outputs[i, 0].cpu().numpy()
        target_img = complete[i, 0].cpu().numpy()
        
        # Determine global min and max for consistent colormap scaling
        vmin = min(input_img.min(), output_img.min(), target_img.min())
        vmax = max(input_img.max(), output_img.max(), target_img.max())
        
        # Plot input
        im = axes[i, 0].imshow(input_img, cmap='viridis', vmin=vmin, vmax=vmax)
        axes[i, 0].set_title('Input (Incomplete)')
        axes[i, 0].axis('off')
        
        # Plot output
        im = axes[i, 1].imshow(output_img, cmap='viridis', vmin=vmin, vmax=vmax)
        axes[i, 1].set_title('Output (Predicted)')
        axes[i, 1].axis('off')
        
        # Plot ground truth
        im = axes[i, 2].imshow(target_img, cmap='viridis', vmin=vmin, vmax=vmax)
        axes[i, 2].set_title('Ground Truth (Complete)')
        axes[i, 2].axis('off')
        
        # Add a colorbar to the last image in each row
        plt.colorbar(im, ax=axes[i, 2], fraction=0.04, pad=0.02)
    
    # Add title information
    plt.suptitle(title, fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    # Save the figure
    plt.savefig(filepath, dpi=600)
    plt.close(fig)

def setup_directories(args):
    """Create necessary directories for saving results"""
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    snapshot_path = os.path.join(args.output_dir, "snapshots")
    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)
    
    visual_path = os.path.join(args.output_dir, "visualizations")
    if not os.path.exists(visual_path):
        os.makedirs(visual_path)
    
    return snapshot_path, visual_path