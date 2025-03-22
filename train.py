import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import logging
from tqdm import tqdm
import time
from util import calculate_metrics, save_visualizations, setup_directories

def train(args, model, train_loader, test_loader, device):
    """Training function"""
    # Setup directories and logging
    snapshot_path, visual_path = setup_directories(args)
    
    # Setup logging
    logging.basicConfig(filename=os.path.join(snapshot_path, 'training.log'), 
                        level=logging.INFO,
                        format='%(asctime)s - %(message)s')
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    logging.getLogger('').addHandler(console)
    
    # Setup TensorBoard
    writer = SummaryWriter(os.path.join(snapshot_path, 'log'))
    
    # Setup optimizer
    if args.batch_size != 24 and args.batch_size % 6 == 0:
        args.base_lr *= args.batch_size / 24
    optimizer = optim.SGD(model.parameters(), lr=args.base_lr, 
                          momentum=0.9, weight_decay=0.0001)
    
    # Setup loss functions
    criterion = nn.MSELoss()
    
    # Initialize best loss for model saving
    best_loss = float('inf')
    best_psnr = 0
    
    # Training loop
    for epoch in range(args.max_epochs):
        model.train()
        epoch_loss = 0
        
        # Use tqdm for progress bar
        train_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{args.max_epochs}')
        for step, batch in enumerate(train_bar):
            incomplete, complete = batch
            incomplete, complete = incomplete.to(device), complete.to(device)
            
            # Forward pass
            outputs = model(incomplete)
            loss = criterion(outputs, complete)
            
            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Update progress bar
            epoch_loss += loss.item()
            train_bar.set_postfix(loss=epoch_loss/(step+1))
            
            # Log to TensorBoard periodically
            if (step + 1) % 20 == 0:
                writer.add_scalar('train/loss', loss.item(), epoch * len(train_loader) + step)
        
        # Log training metrics
        avg_loss = epoch_loss / len(train_loader)
        logging.info(f'Epoch {epoch+1} - Training Loss: {avg_loss:.6f}')
        writer.add_scalar('train/epoch_loss', avg_loss, epoch)
        
        # Test after each epoch
        test_loss, test_psnr, test_ssim = test_epoch(args, model, test_loader, device, 
                                                    epoch, visual_path)
        
        # Log test metrics
        logging.info(f'Epoch {epoch+1} - Test Loss: {test_loss:.6f}, PSNR: {test_psnr:.4f}, SSIM: {test_ssim:.4f}')
        writer.add_scalar('test/loss', test_loss, epoch)
        writer.add_scalar('test/psnr', test_psnr, epoch)
        writer.add_scalar('test/ssim', test_ssim, epoch)
        
        # Save models
        if test_loss < best_loss:
            save_mode_path = os.path.join(snapshot_path, 'best_loss_model.pth')
            torch.save(model.state_dict(), save_mode_path)
            logging.info(f"Saved best loss model to {save_mode_path}")
            best_loss = test_loss
        
        if test_psnr > best_psnr:
            save_mode_path = os.path.join(snapshot_path, 'best_psnr_model.pth')
            torch.save(model.state_dict(), save_mode_path)
            logging.info(f"Saved best PSNR model to {save_mode_path}")
            best_psnr = test_psnr
        
        # Save latest model
        save_mode_path = os.path.join(snapshot_path, 'latest_model.pth')
        torch.save(model.state_dict(), save_mode_path)
        
        # Save checkpoint every N epochs
        if (epoch + 1) % args.save_interval == 0:
            save_mode_path = os.path.join(snapshot_path, f'epoch_{epoch+1}.pth')
            torch.save(model.state_dict(), save_mode_path)
            logging.info(f"Saved checkpoint to {save_mode_path}")
    
    writer.close()
    return model

def test_epoch(args, model, test_loader, device, epoch, visual_path):
    """Function to test model during training"""
    model.eval()
    test_loss = 0
    criterion = nn.MSELoss()
    
    psnr_vals = []
    ssim_vals = []
    
    # Visualize a batch of results
    visualized = False
    
    with torch.no_grad():
        for step, batch in enumerate(test_loader):
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
            
            # Visualize first batch results
            if not visualized and step == 0:
                save_visualizations(
                    incomplete, outputs, complete,
                    os.path.join(visual_path, f'epoch_{epoch+1}.png'),
                    title=f'Epoch {epoch+1} Results'
                )
                visualized = True
    
    # Calculate average metrics
    avg_loss = test_loss / len(test_loader)
    avg_psnr = np.mean(psnr_vals)
    avg_ssim = np.mean(ssim_vals)
    
    return avg_loss, avg_psnr, avg_ssim