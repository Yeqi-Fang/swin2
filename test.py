import os
import torch
import torch.nn as nn
import numpy as np
import logging
from tqdm import tqdm
import time
from util import calculate_metrics, save_visualizations, setup_directories

def test(args, model, test_loader, device):
    """Testing function"""
    # Setup directories and logging
    snapshot_path, visual_path = setup_directories(args)
    
    # Setup logging
    logging.basicConfig(filename=os.path.join(snapshot_path, 'testing.log'), 
                        level=logging.INFO,
                        format='%(asctime)s - %(message)s')
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    logging.getLogger('').addHandler(console)
    
    model.eval()
    criterion = nn.MSELoss()
    test_loss = 0
    
    # Metrics storage
    psnr_vals = []
    ssim_vals = []
    
    # Create folder for test visualizations
    test_vis_path = os.path.join(visual_path, 'test_results')
    if not os.path.exists(test_vis_path):
        os.makedirs(test_vis_path)
    
    logging.info(f"Starting testing on {len(test_loader)} batches")
    start_time = time.time()
    
    with torch.no_grad():
        for step, batch in enumerate(tqdm(test_loader, desc='Testing')):
            incomplete, complete = batch
            incomplete, complete = incomplete.to(device), complete.to(device)
            
            # Forward pass
            outputs = model(incomplete)
            loss = criterion(outputs, complete)
            test_loss += loss.item()
            
            # Calculate metrics
            batch_psnr, batch_ssim = calculate_metrics(outputs, complete)
            psnr_vals.append(batch_psnr)
            ssim_vals.append(batch_ssim)
            
            # Visualize every N batches or for specific batches of interest
            if step % args.vis_frequency == 0 or step < 5:
                save_visualizations(
                    incomplete, outputs, complete,
                    os.path.join(test_vis_path, f'batch_{step}.png'),
                    title=f'Test Results - Batch {step}'
                )
    
    # Calculate average metrics
    avg_loss = test_loss / len(test_loader)
    avg_psnr = np.mean(psnr_vals)
    avg_ssim = np.mean(ssim_vals)
    
    test_time = time.time() - start_time
    
    # Log results
    logging.info(f'Testing completed in {test_time:.2f} seconds')
    logging.info(f'Average Test Loss: {avg_loss:.6f}')
    logging.info(f'Average PSNR: {avg_psnr:.4f}')
    logging.info(f'Average SSIM: {avg_ssim:.4f}')
    
    # Save results to file
    results_file = os.path.join(snapshot_path, 'test_results.txt')
    with open(results_file, 'w') as f:
        f.write(f'Average Test Loss: {avg_loss:.6f}\n')
        f.write(f'Average PSNR: {avg_psnr:.4f}\n')
        f.write(f'Average SSIM: {avg_ssim:.4f}\n')
    
    return avg_loss, avg_psnr, avg_ssim