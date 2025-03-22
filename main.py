import os
import argparse
import torch
import logging
import random
import numpy as np
from torch.utils.data import DataLoader

# Import custom modules
from swin_transformer_unet_skip_expand_decoder_sys import SwinTransformerSys
from vit import SwinUnet
from dateset import create_dataloaders
from train import train
from test import test

def main():
    parser = argparse.ArgumentParser(description='Incomplete Ring PET Reconstruction')
    
    # Data parameters
    parser.add_argument('--data_dir', type=str, default='./data', help='Path to the dataset')
    parser.add_argument('--output_dir', type=str, default='./output', help='Output directory')
    
    # Model parameters
    parser.add_argument('--img_size', type=int, default=224, help='Input image size')
    parser.add_argument('--patch_size', type=int, default=4, help='Patch size')
    parser.add_argument('--in_chans', type=int, default=1, help='Input channels')
    parser.add_argument('--num_classes', type=int, default=1, help='Output channels')
    parser.add_argument('--embed_dim', type=int, default=96, help='Embedding dimension')
    parser.add_argument('--depths', type=int, nargs='+', default=[2, 2, 2, 2], help='Encoder depths')
    parser.add_argument('--decoder_depths', type=int, nargs='+', default=[2, 2, 2, 1], help='Decoder depths')
    parser.add_argument('--num_heads', type=int, nargs='+', default=[3, 6, 12, 24], help='Number of attention heads')
    parser.add_argument('--window_size', type=int, default=7, help='Window size for attention')
    parser.add_argument('--drop_path_rate', type=float, default=0.2, help='Drop path rate')
    parser.add_argument('--pretrain_ckpt', type=str, default="./pretrained_ckpt/swin_tiny_patch4_window7_224.pth", 
                        help='Path to pretrained checkpoint')
    
    # Training parameters
    parser.add_argument('--batch_size', type=int, default=24, help='Batch size')
    parser.add_argument('--base_lr', type=float, default=0.01, help='Base learning rate')
    parser.add_argument('--max_epochs', type=int, default=150, help='Maximum number of epochs')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of data loading workers')
    parser.add_argument('--save_interval', type=int, default=10, help='Interval for saving models')
    parser.add_argument('--vis_frequency', type=int, default=5, help='Frequency of visualization during testing')
    parser.add_argument('--seed', type=int, default=1234, help='Random seed')
    parser.add_argument('--test', type=bool, default=False, help='Random seed')
    # Mode
    parser.add_argument('--mode', type=str, choices=['train', 'test'], default='train', 
                        help='Train or test mode')
    
    args = parser.parse_args()
    
    # Set random seeds for reproducibility
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create config object for SwinUnet
    class Config:
        pass
    
    config = Config()
    config.DATA = Config()
    config.MODEL = Config()
    config.MODEL.SWIN = Config()
    config.TRAIN = Config()
    
    # Set configuration parameters according to the provided YAML structure
    # MODEL section
    config.MODEL.TYPE = "swin"
    config.MODEL.NAME = "swin_tiny_patch4_window7_224"
    config.MODEL.DROP_PATH_RATE = args.drop_path_rate
    config.MODEL.PRETRAIN_CKPT = args.pretrain_ckpt
    
    # MODEL.SWIN section
    config.MODEL.SWIN.FINAL_UPSAMPLE = "expand_first"
    config.MODEL.SWIN.EMBED_DIM = args.embed_dim
    config.MODEL.SWIN.DEPTHS = args.depths
    config.MODEL.SWIN.DECODER_DEPTHS = args.decoder_depths
    config.MODEL.SWIN.NUM_HEADS = args.num_heads
    config.MODEL.SWIN.WINDOW_SIZE = args.window_size
    
    # Additional required parameters
    config.DATA.IMG_SIZE = args.img_size
    config.MODEL.SWIN.PATCH_SIZE = args.patch_size
    config.MODEL.SWIN.IN_CHANS = args.in_chans
    config.MODEL.SWIN.QKV_BIAS = True
    config.MODEL.SWIN.QK_SCALE = None
    config.MODEL.DROP_RATE = 0.0
    config.MODEL.SWIN.APE = False
    config.MODEL.SWIN.PATCH_NORM = True
    config.TRAIN.USE_CHECKPOINT = False
    
    # Initialize model
    model = SwinUnet(config, img_size=args.img_size, num_classes=args.num_classes).to(device)
    
    # Load pretrained weights using the load_from method
    model.load_from(config)
    
    # Create dataloaders
    train_loader, test_loader = create_dataloaders(
        args.data_dir, 
        batch_size=args.batch_size, 
        num_workers=args.num_workers,
        test=args.test
    )
    
    # Run in the specified mode
    if args.mode == 'train':
        model = train(args, model, train_loader, test_loader, device)
    elif args.mode == 'test':
        test(args, model, test_loader, device)

if __name__ == '__main__':
    main()