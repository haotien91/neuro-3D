import torch
import torch.nn as nn
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader
from eeg_data_process.EEGdataset import AllDataFeatureTwoEEG
from PointGeneration.ds_shape_color_generation import EEGTo3DDiffusionModel
import argparse

def analyze_attention(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Load Model
    # We need to instantiate the full model to load the weights
    model = EEGTo3DDiffusionModel(
        beta_start=1e-5, beta_end=8e-3, beta_schedule='linear', 
        sub=args.sub, generate_type='shape',
        point_cloud_model_embed_dim=64, in_channels=1027, out_channels=3,
        retri_pretrain_model=f"{args.data_path}/model/retraival/{args.sub}/{args.pretrain_model}/"
    )
    
    # Load checkpoint
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    if 'model' in checkpoint:
        state_dict = checkpoint['model']
    else:
        state_dict = checkpoint
    
    # Remove 'module.' prefix if present
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('module.'):
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v
            
    model.load_state_dict(new_state_dict, strict=False)
    model.to(device)
    model.eval()
    print("Model loaded.")

    # 2. Load Data
    test_set = AllDataFeatureTwoEEG(
        args.data_path, sub_list=[args.sub], train=False, 
        num_classes=72
    )
    dataloader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False)
    
    # Get one batch
    batch = next(iter(dataloader))
    eeg_data = batch['eeg_data'].float().to(device)
    eeg_data2 = batch['eeg_data2'].float().to(device)
    
    # Apply Condition Mode
    if args.condition_mode == 'random':
        print("Using RANDOM condition (torch.randn_like)")
        eeg_data = torch.randn_like(eeg_data)
        eeg_data2 = torch.randn_like(eeg_data2)
    elif args.condition_mode == 'zero':
        print("Using ZERO condition (torch.zeros_like)")
        eeg_data = torch.zeros_like(eeg_data)
        eeg_data2 = torch.zeros_like(eeg_data2)
    else:
        print("Using NORMAL condition")
    
    print(f"EEG Data Shape: {eeg_data.shape}")
    print(f"EEG Data 2 Shape: {eeg_data2.shape}")

    # 3. Gradient Saliency Analysis
    print("Computing Gradient Saliency...")
    eeg_data.requires_grad = True
    eeg_data2.requires_grad = True
    
    # Forward pass
    # We only need the EEG encoder part: model.meta_eeg_video
    # The modified forward returns: clip_out, cls_result, clip_out2, cls_result2, dyn_weights, stc_weights
    eeg_features1, shape_result, eeg_features2, color_result, dyn_weights, stc_weights = model.meta_eeg_video(eeg_data, eeg_data2)
    
    # Compute Loss (Sum of features to maximize activation)
    loss = eeg_features1.sum() + eeg_features2.sum()
    loss.backward()
    
    # Get Gradients
    grad_dyn = eeg_data.grad.abs().cpu().numpy() # [B, 64, 600]
    grad_stc = eeg_data2.grad.abs().cpu().numpy() # [B, 64, 250]
    
    # Compute Importance
    # Channel Importance: Mean over Time and Batch
    channel_imp_dyn = grad_dyn.mean(axis=(0, 2))
    channel_imp_stc = grad_stc.mean(axis=(0, 2))
    
    # Time Importance (Gradient): Mean over Channel and Batch
    time_imp_grad_dyn = grad_dyn.mean(axis=(0, 1))
    time_imp_grad_stc = grad_stc.mean(axis=(0, 1))
    
    # 4. Attention Weight Analysis
    print("Analyzing Attention Weights...")
    # dyn_weights: [B, 600, 600] (Batch, Query_Time, Key_Time)
    # stc_weights: [B, 250, 250]
    
    # Time Importance (Attention): Mean over Batch and Query_Time
    # This tells us: "On average, how much attention does each Key_Time receive?"
    time_imp_attn_dyn = dyn_weights.mean(dim=(0, 1)).detach().cpu().numpy()
    time_imp_attn_stc = stc_weights.mean(dim=(0, 1)).detach().cpu().numpy()
    
    # 5. Visualization
    os.makedirs(args.output_dir, exist_ok=True)
    suffix = f"_{args.condition_mode}"
    
    # Plot 1: Channel Importance
    plt.figure(figsize=(12, 6))
    x = np.arange(len(channel_imp_dyn))
    plt.bar(x - 0.2, channel_imp_dyn, width=0.4, label='Dynamic EEG', alpha=0.7)
    plt.bar(x + 0.2, channel_imp_stc, width=0.4, label='Static EEG', alpha=0.7)
    plt.xlabel('Channel Index')
    plt.ylabel('Gradient Saliency (Importance)')
    plt.title(f'Channel Importance ({args.sub}) - {args.condition_mode}')
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    plt.savefig(f"{args.output_dir}/channel_importance{suffix}.png")
    print(f"Saved {args.output_dir}/channel_importance{suffix}.png")
    
    # Plot 2: Time Importance (Dynamic)
    plt.figure(figsize=(12, 6))
    ax1 = plt.gca()
    ax2 = ax1.twinx()
    
    x_dyn = np.arange(len(time_imp_grad_dyn))
    l1 = ax1.plot(x_dyn, time_imp_grad_dyn, 'b-', label='Gradient Saliency', alpha=0.7)
    l2 = ax2.plot(x_dyn, time_imp_attn_dyn, 'r--', label='Attention Weight', alpha=0.7)
    
    ax1.set_xlabel('Time Step (Dynamic, 100Hz)')
    ax1.set_ylabel('Gradient Saliency', color='b')
    ax2.set_ylabel('Attention Weight', color='r')
    plt.title(f'Temporal Importance - Dynamic EEG ({args.sub}) - {args.condition_mode}')
    
    # Legend
    lns = l1 + l2
    labs = [l.get_label() for l in lns]
    ax1.legend(lns, labs, loc=0)
    
    plt.savefig(f"{args.output_dir}/time_importance_dynamic{suffix}.png")
    print(f"Saved {args.output_dir}/time_importance_dynamic{suffix}.png")

    # Plot 3: Time Importance (Static)
    plt.figure(figsize=(12, 6))
    ax1 = plt.gca()
    ax2 = ax1.twinx()
    
    x_stc = np.arange(len(time_imp_grad_stc))
    l1 = ax1.plot(x_stc, time_imp_grad_stc, 'b-', label='Gradient Saliency', alpha=0.7)
    l2 = ax2.plot(x_stc, time_imp_attn_stc, 'r--', label='Attention Weight', alpha=0.7)
    
    ax1.set_xlabel('Time Step (Static, 250Hz)')
    ax1.set_ylabel('Gradient Saliency', color='b')
    ax2.set_ylabel('Attention Weight', color='r')
    plt.title(f'Temporal Importance - Static EEG ({args.sub}) - {args.condition_mode}')
    
    # Legend
    lns = l1 + l2
    labs = [l.get_label() for l in lns]
    ax1.legend(lns, labs, loc=0)
    
    plt.savefig(f"{args.output_dir}/time_importance_static{suffix}.png")
    print(f"Saved {args.output_dir}/time_importance_static{suffix}.png")

    # Plot 4: 2D Attention Map (Dynamic)
    plt.figure(figsize=(10, 8))
    # dyn_weights: [B, 600, 600]. Mean over batch -> [600, 600]
    avg_dyn_attn = dyn_weights.mean(dim=0).detach().cpu().numpy()
    plt.imshow(avg_dyn_attn, aspect='auto', origin='upper', cmap='viridis')
    plt.colorbar(label='Attention Weight')
    plt.xlabel('Key Time Step (100Hz)')
    plt.ylabel('Query Time Step (100Hz)')
    plt.title(f'Average Attention Map - Dynamic ({args.sub}) - {args.condition_mode}')
    plt.savefig(f"{args.output_dir}/attention_map_dynamic{suffix}.png")
    print(f"Saved {args.output_dir}/attention_map_dynamic{suffix}.png")

    # Plot 5: 2D Attention Map (Static)
    plt.figure(figsize=(10, 8))
    # stc_weights: [B, 250, 250]. Mean over batch -> [250, 250]
    avg_stc_attn = stc_weights.mean(dim=0).detach().cpu().numpy()
    plt.imshow(avg_stc_attn, aspect='auto', origin='upper', cmap='viridis')
    plt.colorbar(label='Attention Weight')
    plt.xlabel('Key Time Step (250Hz)')
    plt.ylabel('Query Time Step (250Hz)')
    plt.title(f'Average Attention Map - Static ({args.sub}) - {args.condition_mode}')
    plt.savefig(f"{args.output_dir}/attention_map_static{suffix}.png")
    print(f"Saved {args.output_dir}/attention_map_static{suffix}.png")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--sub', type=str, default='sub03')
    parser.add_argument('--data_path', type=str, default='./')
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--pretrain_model', type=str, default='')
    parser.add_argument('--output_dir', type=str, default='analysis_sub03')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--eeg_channels', type=int, default=64)
    parser.add_argument('--condition_mode', type=str, default='normal', choices=['normal', 'zero', 'random'])
    
    args = parser.parse_args()
    analyze_attention(args)
