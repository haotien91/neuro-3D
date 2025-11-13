import os

import torch
import torch.optim as optim
from torch.nn import CrossEntropyLoss
from torch.nn import functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
from einops.layers.torch import Rearrange
import random
import wandb
import sys
import argparse
sys.path.append('.')
from eeg_data_process.extract_eeg_feature import VideoImageEEGClassifyColor3
from eeg_data_process.EEGdataset import AllDataFeatureTwoEEG
from eeg_data_process.clip_loss import ClipLoss
from eeg_data_process.channel_selection import CHANNEL_INDICES_64_TO_32, CHANNEL_INDICES_64_TO_22


###python classification/retri_shape_color.py --model 'VideoImageEEGClassifyColor3'

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

class_weights = torch.tensor([1.0, 1.0, 0.1, 0.1, 1.0, 1.0])


def get_result(labels, cls_result, K=3):
    cls_result_soft = cls_result.softmax(1)
    _, predicted = torch.max(cls_result_soft, 1)
    correct = (predicted == labels).sum().item()
    
    topk_pred = cls_result_soft.topk(K, dim=1)[1]
    top_5_correct = topk_pred.eq(labels.view(-1, 1).expand_as(topk_pred)).sum().item()
    coarse_num = top_5_correct
    return correct, top_5_correct, coarse_num

feature_cls = 'color_video_fea' # color_video_fea, color_point_fea, gray_video_fea, gray_point_fea, txt_fea

def train_model(config, eeg_model, dataloader, optimizer, device, text_features_all, img_features_all, lr_scheduler):
    eeg_model.train()
    text_features_all = text_features_all[:, 0].to(device).float()
    img_features_all = img_features_all[:, 0].to(device).float()
    
    total_loss = 0
    
    shape_correct, shape_top5_correct = 0, 0
    color_correct, color_top2_correct = 0, 0
    retri_correct, retri_top5_correct = 0, 0
    total = 0

    alpha=0.99
    mse_loss_fn = nn.MSELoss()
    criterion1 = nn.CrossEntropyLoss()
    criterion2 = nn.CrossEntropyLoss(class_weights.to(device))
    for batch_idx, batch_data in enumerate(dataloader):
        if batch_idx % 50 == 0:
            print(batch_idx)
        
        eeg_data = batch_data['eeg_data'].to(device).float()
        eeg_data2 = batch_data['eeg_data2'].to(device).float()
        # img_features = batch_data['color_video_fea'].to(device).float()
        # eeg_data = batch_data['color_video_fea'].to(device).float()
        img_features = batch_data[feature_cls].to(device).float()
        # img_features = batch_data['gray_point_fea'].to(device).float()
        # txt_features = batch_data['txt_fea'].to(device).float()
        labels_color = batch_data['color_label'].to(device).long()
        labels_shape = batch_data['cls_label'].to(device)
        
        optimizer.zero_grad()
        eeg_features1, shape_result, eeg_features2, color_result = eeg_model(eeg_data, eeg_data2)
                
        # features_list.append(eeg_features)
        #eeg_model.loss_func(eeg_features2, img_features, logit_scale) + 
        logit_scale = eeg_model.logit_scale
        img_loss = eeg_model.loss_func(eeg_features1, img_features, logit_scale) + eeg_model.loss_func(eeg_features2, img_features, logit_scale)
        contrastive_loss = img_loss# + text_loss
        regress_loss =  mse_loss_fn(eeg_features2, img_features) + mse_loss_fn(eeg_features1, img_features)
        # import pdb;pdb.set_trace()
        loss_cls = criterion2(color_result, labels_color) + criterion1(shape_result, labels_shape)
        loss = alpha * regress_loss * 10 + (1 - alpha) * contrastive_loss * 10 + loss_cls * 0.1
        # loss = loss_cls * 0.1
        loss.backward()
        optimizer.step()
        total += labels_shape.shape[0]
        
        logits_img = logit_scale * eeg_features1 @ img_features_all.T
        logits_single = logits_img
        predicted = torch.argmax(logits_single, dim=1) # (n_batch, ) \in {0, 1, ..., n_cls-1}
        retri_correct += (predicted == labels_shape).sum().item()

        total_loss += loss.item()
        co1, co2, co3 = get_result(labels_shape, shape_result, 5)
        shape_correct += co1
        shape_top5_correct += co2

        co1, co2, co3 = get_result(labels_color, color_result, 2)
        color_correct += co1
        color_top2_correct += co2
        
        lr_scheduler.step()

    average_loss = total_loss / (batch_idx+1)
    shape_acc, shape_top5_acc = shape_correct / total, shape_top5_correct / total
    color_acc, color_top2_acc = color_correct / total, color_top2_correct / total
    retri_acc, retri_top5_acc = retri_correct / total, retri_top5_correct / total
    return average_loss, retri_acc, retri_top5_acc, shape_acc, shape_top5_acc, color_acc, color_top2_acc

def evaluate_model(config, eeg_model, dataloader, device, text_features_all, img_features_all):
    eeg_model.eval()
    
    if len(img_features_all.shape) == 3:
        img_features_all = img_features_all[:, 0].to(device).float()
        text_features_all = text_features_all[:, 0].to(device).float()
    else:
        img_features_all = img_features_all.to(device).float()
        text_features_all = text_features_all.to(device).float()
    
    total_loss = 0
    total = 0
    shape_correct, shape_top5_correct = 0, 0
    color_correct, color_top2_correct = 0, 0
    retri_correct, retri_top5_correct = 0, 0
    batch_idx = -1  # Initialize to handle empty dataloader
    
    criterion1 = nn.CrossEntropyLoss()
    criterion2 = nn.CrossEntropyLoss(class_weights.to(device))
    with torch.no_grad():
        for batch_idx, batch_data in enumerate(dataloader):
            eeg_data = batch_data['eeg_data'].to(device).float()
            eeg_data2 = batch_data['eeg_data2'].to(device).float()
            # img_features = batch_data['color_video_fea'].to(device).float()
            # eeg_data = batch_data['color_video_fea'].to(device).float()
            img_features = batch_data[feature_cls].to(device).float()
            # img_features = batch_data['gray_point_fea'].to(device).float()
            labels_color = batch_data['color_label'].to(device).long()
            labels_shape = batch_data['cls_label'].to(device)
            
            # import pdb;pdb.set_trace()
            eeg_features1, shape_result, eeg_features2, color_result = eeg_model(eeg_data, eeg_data2)
            # import pdb;pdb.set_trace()
            loss_cls = criterion2(color_result, labels_color) + criterion1(shape_result, labels_shape)
            logit_scale = eeg_model.logit_scale 
            # import pdb;pdb.set_trace()
            loss = loss_cls * 0.1
            total_loss += loss.item()
            total += labels_shape.shape[0]
            
            co1, co2, co3 = get_result(labels_shape, shape_result, 5)
            shape_correct += co1
            shape_top5_correct += co2
            co1, co2, co3 = get_result(labels_color, color_result, 2)
            color_correct += co1
            color_top2_correct += co2
            
            for idx, label in enumerate(labels_shape):
                logits_text = logit_scale * eeg_features1[idx] @ img_features_all.T  
                logits_single = logits_text
                predicted_label = torch.argmax(logits_single).item() # (n_batch, ) \in {0, 1, ..., n_cls-1}
                if predicted_label == label.item():
                    retri_correct += 1
                _, top5_indices = torch.topk(logits_single, 5, largest =True)                             
                if label.item() in [i for i in top5_indices.tolist()]:                
                    retri_top5_correct+=1    
            
    average_loss = total_loss / (batch_idx+1)
    shape_acc, shape_top5_acc = shape_correct / total, shape_top5_correct / total
    color_acc, color_top2_acc = color_correct / total, color_top2_correct / total
    retri_acc, retri_top5_acc = retri_correct / total, retri_top5_correct / total
    return average_loss, retri_acc, retri_top5_acc, shape_acc, shape_top5_acc, color_acc, color_top2_acc

def adjust_lr(optimizer, epoch, lr, config):
    lr_c = lr * ((1 - epoch/(config['epochs'] + 10)) ** 0.9)
    print(lr_c)
    for p in optimizer.param_groups:
        p['lr'] = lr_c

def main_train_loop(sub, current_time, eeg_model, train_dataloader, test_dataloader, optimizer, device, 
                    text_features_train_all, text_features_test_all, img_features_train_all, img_features_test_all, config):
    # Use run_name directly for cleaner paths that match recon_main.py expectations
    save_model_path = f"{config['save_path']}/{sub}/{current_time}"
    os.makedirs(save_model_path, exist_ok=True)

    # Initialize W&B if enabled
    use_wandb = config.get('use_wandb', False)
    if use_wandb:
        try:
            import wandb
            wandb_run_name = config.get('wandb_run_name', current_time)
            wandb_config = {
                'sub': sub,
                'model': config['model'],
                'lr': config['lr'],
                'epochs': config['epochs'],
                'batch_size': config['batch_size'],
                'num_classes': config['num_classes'],
                'num_channels': config['num_channels'],
                'time_len1': config['time_len1'],
                'time_len2': config['time_len2'],
            }
            wandb.init(
                project=config.get('wandb_project', 'neuro-3d'),
                entity=config.get('wandb_entity', None),
                name=wandb_run_name,
                config=wandb_config,
                dir=save_model_path
            )
            print(f"[W&B] Initialized run: {wandb_run_name}")
        except ImportError:
            print("[W&B] wandb not installed. Install with: pip install wandb")
            use_wandb = False
        except Exception as e:
            print(f"[W&B] Failed to initialize: {e}")
            print("[W&B] Tip: Run 'wandb login' first if not logged in")
            use_wandb = False

    # Initialize CSV logging
    import csv
    csv_path = f"{save_model_path}/metrics.csv"
    csv_file = open(csv_path, 'w', newline='')
    csv_writer = csv.DictWriter(csv_file, fieldnames=[
        'epoch', 'train_loss', 'test_loss',
        'train_retri_acc', 'train_retri_top5_acc',
        'train_shape_acc', 'train_shape_top5_acc',
        'train_color_acc', 'train_color_top2_acc',
        'test_retri_acc', 'test_retri_top5_acc',
        'test_shape_acc', 'test_shape_top5_acc',
        'test_color_acc', 'test_color_top2_acc',
        'best_retri_acc', 'best_shape_acc', 'best_color_acc'
    ])
    csv_writer.writeheader()
    csv_file.flush()
    print(f"[CSV] Logging to: {csv_path}")

    record_log = open(config['save_path'] + '/result_color_shape_all.txt', 'a')

    log_f = open(save_model_path + '/log.txt', 'w')
    for key in config.keys():
        log_f.write(f"{key}:{config[key]}\n")
    log_f.flush()
    
    train_losses, test_losses = [], []
    train_retri_accuracies, train_shape_accuracies, train_color_accuracies = [], [], []
    test_retri_accuracies, test_shape_accuracies, test_color_accuracies = [], [], []
    
    best_shape_acc, best_shape_top5_acc, best_shape_top5_acc_all = 0.0, 0.0, 0.0
    best_color_acc, best_color_top2_acc, best_color_top2_acc_all = 0.0, 0.0, 0.0
    best_retri_acc, best_retri_top5_acc, best_retri_top5_acc_all = 0.0, 0.0, 0.0

    results = []  
    total_steps = int((config['epochs']+5) * len(train_dataloader))
    # import pdb;pdb.set_trace()
    lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=config['lr'],
        total_steps=total_steps,
        final_div_factor=10000,
        last_epoch=-1, pct_start=2 / config['epochs']
    )
    
    import time
    training_start_time = time.time()
    
    for epoch in range(config['epochs']):
        if epoch == 0:
            test_loss, test_retri_acc, test_retri_top5_acc, test_shape_acc, test_shape_top5_acc, test_color_acc, test_color_top2_acc = evaluate_model(config, eeg_model, test_dataloader, device, text_features_test_all, img_features_test_all)
            print(test_loss, test_retri_acc, test_shape_acc, test_color_acc)
            trian_loss, train_retri_acc, train_retri_top5_acc, train_shape_acc, train_shape_top5_acc, train_color_acc, train_color_top2_acc = evaluate_model(config, eeg_model, train_dataloader, device, text_features_train_all, img_features_train_all)
            print(trian_loss, train_retri_acc, train_shape_acc, train_color_acc)
        train_loss, train_retri_acc, train_retri_top5_acc, train_shape_acc, train_shape_top5_acc, train_color_acc, train_color_top2_acc = train_model(config, eeg_model, train_dataloader, optimizer, device, text_features_train_all, img_features_train_all, lr_scheduler)
        if (epoch +1) % 20 == 0:                               
            file_path = f"{save_model_path}/{epoch+1}.pth"
            torch.save(eeg_model.state_dict(), file_path)            
            print(f"model saved in {file_path}!")
        test_loss, test_retri_acc, test_retri_top5_acc, test_shape_acc, test_shape_top5_acc, test_color_acc, test_color_top2_acc = evaluate_model(config, eeg_model, test_dataloader, device, text_features_test_all, img_features_test_all)
        # adjust_lr(optimizer, epoch, config['lr'], config)
        
        train_losses.append(train_loss)
        train_retri_accuracies.append(train_retri_acc)
        train_shape_accuracies.append(train_shape_acc)
        train_color_accuracies.append(train_color_acc)
        test_losses.append(test_loss)
        test_retri_accuracies.append(test_retri_acc)
        test_shape_accuracies.append(test_shape_acc)
        test_color_accuracies.append(test_color_acc)
        
        # Append results for this epoch
        epoch_results = {
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_retri_acc": train_retri_acc,
            "train_shape_acc": train_shape_acc,
            "train_color_acc": train_color_acc,
            "test_loss": test_loss,
            "test_retri_acc": test_retri_acc,
            "test_shape_acc": test_shape_acc,
            "test_color_acc": test_color_acc,
        }

        results.append(epoch_results)
        
        if test_retri_acc > best_retri_acc or (test_retri_acc == best_retri_acc and test_retri_top5_acc > best_retri_top5_acc):
            best_retri_top5_acc = test_retri_top5_acc
            best_retri_acc = test_retri_acc
            file_path = f"{save_model_path}/best-retri.pth"
            torch.save(eeg_model.state_dict(), file_path)            
            print(f"model saved in {file_path}!")
        
        if test_shape_acc > best_shape_acc or (test_shape_acc == best_shape_acc and test_shape_top5_acc > best_shape_top5_acc):
            best_shape_acc = test_shape_acc
            best_shape_top5_acc = test_shape_top5_acc
            file_path = f"{save_model_path}/best-shape.pth"
            torch.save(eeg_model.state_dict(), file_path)            
            print(f"model saved in {file_path}!")
        
        if test_color_acc > best_color_acc or (test_color_acc == best_color_acc and test_color_top2_acc > best_color_top2_acc):
            best_color_acc = test_color_acc
            best_color_top2_acc = test_color_top2_acc
            file_path = f"{save_model_path}/best-color.pth"
            torch.save(eeg_model.state_dict(), file_path)            
            print(f"model saved in {file_path}!")
        
        
        best_epoch_info = {
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "train_retri_acc": train_retri_acc,
                "train_shape_acc": train_shape_acc,
                "train_color_acc": train_color_acc,
                "test_loss": test_loss,
                "test_retri_acc": best_retri_acc,
                "test_shape_acc": best_shape_acc,
                "test_color_acc": best_color_acc,
            }
        best_retri_top5_acc_all = max([best_retri_top5_acc_all, test_retri_top5_acc])
        best_shape_top5_acc_all = max([best_shape_top5_acc_all, test_shape_top5_acc])
        best_color_top2_acc_all = max([best_color_top2_acc_all, test_color_top2_acc])

        ss = f"Epoch {epoch + 1}/{config['epochs']} - Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}\n"
        ss += f"train_retri_acc: {train_retri_acc:.4f}, train_shape_acc: {train_shape_acc:.4f}, train_color_acc: {train_color_acc:.4f}\n"
        ss += f"test_retri_acc: {test_retri_acc:.4f}. test_shape_acc: {test_shape_acc:.4f}, test_color_acc: {test_color_acc:.4f}\n"
        ss += f"best_retri_acc: {best_retri_acc:.4f}, best_shape_acc: {best_shape_acc:.4f}, best_color_acc: {best_color_acc:.4f}\n"
        ss += f"test_retri_top5_acc: {test_retri_top5_acc:.4f}, test_shape_top5_acc: {test_shape_top5_acc:.4f}, test_color_top2_acc: {test_color_top2_acc:.4f}\n"
        ss += f"best_retri_top5_acc_all: {best_retri_top5_acc_all:.4f}, best_shape_top5_acc_all: {best_shape_top5_acc_all:.4f}, best_color_top2_acc_all: {best_color_top2_acc_all:.4f}\n"
        print(ss)
        log_f.write(ss)
        log_f.flush()

        # Log to W&B
        if use_wandb:
            elapsed_time = time.time() - training_start_time
            wandb.log({
                'epoch': epoch + 1,
                'train/loss': train_loss,
                'train/retri_acc': train_retri_acc,
                'train/retri_top5_acc': train_retri_top5_acc,
                'train/shape_acc': train_shape_acc,
                'train/shape_top5_acc': train_shape_top5_acc,
                'train/color_acc': train_color_acc,
                'train/color_top2_acc': train_color_top2_acc,
                'test/loss': test_loss,
                'test/retri_acc': test_retri_acc,
                'test/retri_top5_acc': test_retri_top5_acc,
                'test/shape_acc': test_shape_acc,
                'test/shape_top5_acc': test_shape_top5_acc,
                'test/color_acc': test_color_acc,
                'test/color_top2_acc': test_color_top2_acc,
                'best/retri_acc': best_retri_acc,
                'best/shape_acc': best_shape_acc,
                'best/color_acc': best_color_acc,
                'time/elapsed_hours': elapsed_time / 3600,
                'time/elapsed_minutes': elapsed_time / 60,
            }, step=epoch + 1)

        # Log to CSV
        csv_writer.writerow({
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'test_loss': test_loss,
            'train_retri_acc': train_retri_acc,
            'train_retri_top5_acc': train_retri_top5_acc,
            'train_shape_acc': train_shape_acc,
            'train_shape_top5_acc': train_shape_top5_acc,
            'train_color_acc': train_color_acc,
            'train_color_top2_acc': train_color_top2_acc,
            'test_retri_acc': test_retri_acc,
            'test_retri_top5_acc': test_retri_top5_acc,
            'test_shape_acc': test_shape_acc,
            'test_shape_top5_acc': test_shape_top5_acc,
            'test_color_acc': test_color_acc,
            'test_color_top2_acc': test_color_top2_acc,
            'best_retri_acc': best_retri_acc,
            'best_shape_acc': best_shape_acc,
            'best_color_acc': best_color_acc,
        })
        csv_file.flush()
    log_f.close()
    csv_file.close()
    
    total_training_time = time.time() - training_start_time
    print(f"\n[TIME] Total training time: {total_training_time/3600:.2f} hours ({total_training_time/60:.2f} minutes)")
    
    if use_wandb:
        wandb.log({'time/total_hours': total_training_time / 3600})
        wandb.finish()
        print("[W&B] Run finished")
    print(f"[CSV] Metrics saved to: {csv_path}")

    ss = f"{sub}, {current_time}:\n"
    ss += f"best_retri_acc: {best_retri_acc:.4f}, best_shape_acc: {best_shape_acc:.4f}, best_color_acc: {best_color_acc:.4f}\n"
    ss += f"test_retri_top5_acc: {test_retri_top5_acc:.4f}, best_shape_top5_acc: {best_shape_top5_acc:.4f}, test_color_top2_acc: {test_color_top2_acc:.4f}\n"
    ss += f"best_retri_top5_acc_all: {best_retri_top5_acc_all:.4f}, best_shape_top5_acc_all: {best_shape_top5_acc_all:.4f}, best_color_top2_acc_all: {best_color_top2_acc_all:.4f}\n"
    record_log.write(ss + '\n')
    record_log.flush()
    record_log.close()
    return results

def get_parameter_number(model):
    total_num = sum(p.numel() for p in model.parameters())
    trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {'Total': total_num, 'Trainable': trainable_num}

import datetime

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_path', type=str, default="")
    parser.add_argument('--model', type=str, default="VideoImageEEGClassifyColor3")
    parser.add_argument('--time_cls', type=int, default=1)
    parser.add_argument('--max_len', type=int, default=600)
    parser.add_argument('--sub', type=str, default='sub13')
    parser.add_argument('--num_classes', type=int, default=72)
    parser.add_argument('--use_32_channels', action='store_true')  # legacy
    parser.add_argument('--eeg_channels', type=int, default=0, choices=[0,22,32,64])
    parser.add_argument('--run_name', type=str, default='')
    # Logging
    parser.add_argument('--use_wandb', action='store_true', help='Enable Weights & Biases logging')
    parser.add_argument('--wandb_project', type=str, default='neuro-3d', help='W&B project name')
    parser.add_argument('--wandb_entity', type=str, default='', help='W&B entity/username')
    parser.add_argument('--wandb_run_name', type=str, default='', help='W&B run name')
    opt = parser.parse_args()
    return opt


def main(args, sub='sub03'):
    set_seed(0)
    
    # 新增：根據參數決定 channel 數
    if args.eeg_channels in [22,32,64]:
        target = args.eeg_channels
    elif args.use_32_channels:
        target = 32
    else:
        target = 64
    if target == 64:
        num_channels = 64
    elif target == 32:
        num_channels = len(CHANNEL_INDICES_64_TO_32)
    else:
        num_channels = len(CHANNEL_INDICES_64_TO_22)

    print(f"Using {num_channels} channels.")
    
    config = {
        "save_path": args.root_path + "model/retraival/",
        "data_path": args.root_path,
        "lr": 1e-3,
        "epochs": 200,
        "batch_size": 128,
        "time_len1": 600,
        "time_len2": 250,
        "model": args.model,
        "time": args.time_cls,
        "num_classes": args.num_classes,
        "num_channels": num_channels,
        # W&B config
        "use_wandb": args.use_wandb,
        "wandb_project": args.wandb_project,
        "wandb_entity": args.wandb_entity if args.wandb_entity else None,
        "wandb_run_name": args.wandb_run_name if args.wandb_run_name else None,
    }

    device = "cuda" if torch.cuda.is_available() else "cpu"
    num_latents = 1024
    
    if config["model"] == "VideoImageEEGClassifyColor3":
        eeg_model = VideoImageEEGClassifyColor3(
            num_channels=num_channels,  # 改這裡
            sequence_length=config['time_len1'], 
            sequence_length2=config['time_len2'], 
            num_latents=num_latents, 
            cls_num=config['num_classes']
        )
    eeg_model.to(device)
    
    print(get_parameter_number(eeg_model))
    optimizer = torch.optim.AdamW(eeg_model.parameters(), lr=config['lr'], weight_decay=5e-4)#, weight_decay=5e-4
    # optimizer = torch.optim.Adam(eeg_model.parameters(), lr=config['lr'])#, weight_decay=5e-4

    # Charless Yu: 0927 改，測試所有 subject
    if sub == "all":
        # -------------------------
        all_subjects = ['sub01', 'sub02', 'sub03', 'sub04', 'sub05', 'sub06', 'sub07', 'sub08', 'sub09', 'sub10', 'sub11', 'sub12']

        # train_dataset = AllDataFeatureTwoEEG(config['data_path'], sub_list=all_subjects, train=True, aug_data=True)
        # test_dataset = AllDataFeatureTwoEEG(config['data_path'], sub_list=all_subjects, train=False)
        # -------------------------

        train_dataset = AllDataFeatureTwoEEG(
            config['data_path'], 
            sub_list=all_subjects, 
            train=True, 
            aug_data=True,
            num_classes=config['num_classes'],
            use_32_channels=args.use_32_channels,
            eeg_channels=args.eeg_channels
        )
        test_dataset = AllDataFeatureTwoEEG(
            config['data_path'], 
            sub_list=all_subjects, 
            train=False,
            num_classes=config['num_classes'],
            use_32_channels=args.use_32_channels,
            eeg_channels=args.eeg_channels
        )
    

    else:
        # train_dataset = AllDataFeatureTwoEEG(config['data_path'], sub_list=[sub], train=True, aug_data=True)
        # # train_dataset = AllDataFeatureTwoEEG(config['data_path'], sub_list=['sub09', 'sub10', 'sub11', 'sub12', 'sub13', 'sub14'], train=True, aug_data=True)
        # test_dataset = AllDataFeatureTwoEEG(config['data_path'], sub_list=[sub], train=False)

        train_dataset = AllDataFeatureTwoEEG(
            config['data_path'], 
            sub_list=[sub], 
            train=True, 
            aug_data=True,
            num_classes=config['num_classes'],
            use_32_channels=args.use_32_channels,
            eeg_channels=args.eeg_channels
        )
        test_dataset = AllDataFeatureTwoEEG(
            config['data_path'], 
            sub_list=[sub], 
            train=False,
            num_classes=config['num_classes'],
            use_32_channels=args.use_32_channels,
            eeg_channels=args.eeg_channels
        )
        

    
    print(f"train len:{train_dataset.__len__()}, test len:{test_dataset.__len__()}")
    
    txt_features_train_all = train_dataset.txt_features
    txt_features_test_all = test_dataset.txt_features
    if feature_cls == 'color_video_fea':
        img_features_train_all = train_dataset.color_video_features
        img_features_test_all = test_dataset.color_video_features
    elif feature_cls == 'color_point_fea':
        img_features_train_all = train_dataset.color_point_features
        img_features_test_all = test_dataset.color_point_features
    elif feature_cls == 'gray_video_fea':
        img_features_train_all = train_dataset.gray_video_features
        img_features_test_all = test_dataset.gray_video_features
    elif feature_cls == 'gray_point_fea':
        img_features_train_all = train_dataset.gray_point_features
        img_features_test_all = test_dataset.gray_point_features
    elif feature_cls == 'txt_fea':
        img_features_train_all = train_dataset.txt_features
        img_features_test_all = test_dataset.txt_features
    # import pdb;pdb.set_trace()
    
    print(img_features_train_all.shape, img_features_test_all.shape, txt_features_train_all.shape, txt_features_test_all.shape)
    
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, num_workers=0, drop_last=True)
    # Use smaller batch size for test to avoid dropping all data when num_classes is small
    test_batch_size = min(72, len(test_dataset))
    test_loader = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False, num_workers=0, drop_last=False)
    current_time = datetime.datetime.now().strftime("%m-%d_%H-%M")
    run_name = args.run_name if args.run_name else current_time
    results = main_train_loop(sub, run_name, eeg_model, train_loader, test_loader, optimizer, device, 
                                  txt_features_train_all, txt_features_test_all, img_features_train_all, img_features_test_all, config)

if __name__ == '__main__':
    args = parse_args()
    main(args, args.sub)
