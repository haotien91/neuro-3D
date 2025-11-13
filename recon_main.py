import torch
import torch.nn as nn
import numpy as np
import os, sys
from pathlib import Path

from PointGeneration.ds_shape_color_generation import EEGTo3DDiffusionModel

from contextlib import nullcontext
from accelerate import Accelerator
from train_utils.parse import parse_args
from train_utils import training_utils
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
import datetime
from eeg_data_process.EEGdataset import AllDataFeatureTwoEEG
from eeg_data_process.clip_loss import ClipLoss
import open3d as o3d
from accelerate import DistributedDataParallelKwargs
import time
from eeg_data_process.channel_selection import CHANNEL_INDICES_64_TO_32, CHANNEL_INDICES_64_TO_22

def get_parameter_number(model):
    total_num = sum(p.numel() for p in model.parameters())
    trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {'Total': total_num, 'Trainable': trainable_num}

def main():
    args = parse_args()
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(mixed_precision=args.mixed_precision, cpu=False, gradient_accumulation_steps=args.gradient_accumulation_steps, kwargs_handlers=[ddp_kwargs])
    training_utils.setup_distributed_print(accelerator.is_main_process)
    print(f'Current working directory: {os.getcwd()}')
    print(args)
    training_utils.set_seed(args.seed)
    
    # Calculate EEG channels based on args
    if args.eeg_channels in [22, 32, 64]:
        target = args.eeg_channels
    elif args.use_32_channels:
        target = 32
    else:
        target = 64
    if target == 64:
        eeg_num_channels = 64
    elif target == 32:
        eeg_num_channels = len(CHANNEL_INDICES_64_TO_32)
    else:
        eeg_num_channels = len(CHANNEL_INDICES_64_TO_22)
    print(f'Using {eeg_num_channels} EEG channels and {args.num_classes} object classes')
    if args.generation_type == 'shape':
        model = EEGTo3DDiffusionModel(beta_start=args.beta_start, beta_end=args.beta_end, beta_schedule=args.beta_schedule, sub=args.sub, generate_type=args.generation_type,
            retri_pretrain_model=f"{args.data_path}/model/retraival/{args.sub}/{args.pretrain_model}/",point_cloud_model_embed_dim=args.point_cloud_model_embed_dim, in_channels=args.in_channels,out_channels=args.out_channels,
            eeg_num_channels=eeg_num_channels, eeg_cls_num=args.num_classes)
    else:
        model = EEGTo3DDiffusionModel(beta_start=args.beta_start, beta_end=args.beta_end, beta_schedule=args.beta_schedule, sub=args.sub, generate_type=args.generation_type,
            point_cloud_model_embed_dim=args.point_cloud_model_embed_dim, in_channels=args.in_channels, out_channels=args.out_channels+3,
            eeg_num_channels=eeg_num_channels, eeg_cls_num=args.num_classes)
    print(f'Parameters (total): {sum(p.numel() for p in model.parameters()):_d}')
    print(f'Parameters (train): {sum(p.numel() for p in model.parameters() if p.requires_grad):_d}')
    optimizer = training_utils.get_optimizer(args, model, accelerator)
    scheduler = training_utils.get_scheduler(args, optimizer)
    
    train_state: training_utils.TrainState = training_utils.resume_from_checkpoint(args, model, optimizer, scheduler)

    train_set = AllDataFeatureTwoEEG(args.data_path, sub_list=[args.sub], train=True, aug_data=True,
                                     num_classes=args.num_classes, use_32_channels=args.use_32_channels, eeg_channels=args.eeg_channels)
    test_set = AllDataFeatureTwoEEG(args.data_path, sub_list=[args.sub], train=False, point_path=args.ply_point_path,
                                    num_classes=args.num_classes, use_32_channels=args.use_32_channels, eeg_channels=args.eeg_channels)
    point_features_train_all = train_set.color_point_features[:, 0].float()
    video_features_train_all = train_set.color_video_features[:, 0].float()
    point_features_test_all = test_set.color_point_features[:, 0].float()
    video_features_test_all = test_set.color_video_features[:, 0].float()
    
    dataloader_test = DataLoader(dataset=test_set, batch_size=48, num_workers=args.num_workers, shuffle=False)
    dataloader_train = DataLoader(dataset=train_set, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True)
    total_batch_size = args.batch_size * accelerator.num_processes * accelerator.gradient_accumulation_steps
    model, optimizer, scheduler, dataloader_train, dataloader_test = accelerator.prepare(model, optimizer, scheduler, dataloader_train, dataloader_test)
    
    if args.task == 'sample':
        if args.checkpoint_resume == '':
            print('please input checkpoint path!')
            return
        model_save_path = args.checkpoint_resume[:-4] + '/'
        print(model_save_path)
        os.makedirs(model_save_path, exist_ok=True)
        video_acc = retrieval_test(args, model, dataloader_test, point_features_test_all, video_features_test_all)
        print(video_acc)
        # import pdb;pdb.set_trace()
        visualize(args, model, dataloader_test, accelerator, train_state.step, point_features_test_all, video_features_test_all, model_save_path, infer_steps=1000, num=5)
        return 
    video_acc = retrieval_test(args, model, dataloader_test, point_features_test_all, video_features_test_all)
    print(video_acc)
    video_acc = retrieval_test(args, model, dataloader_train, point_features_train_all, video_features_train_all)
    print(video_acc)
    ss = f'\n***** Starting training at {datetime.datetime.now()} *****\n'
    ss += f'    Dataset train size: {len(dataloader_train.dataset):_}\n'
    ss += f'    Dataset val size: {len(dataloader_train.dataset):_}\n'
    ss += f'    Dataloader train size: {len(dataloader_train):_}\n'
    ss += f'    Batch size per device = {args.batch_size}\n'
    ss += f'    Total train batch size (w. parallel, dist & accum) = {total_batch_size}\n'
    ss += f'    Gradient Accumulation steps = {args.gradient_accumulation_steps}\n'
    ss += f'    Max training steps = {args.max_steps}\n'
    ss += f'    Training state = {train_state}\n'
    # if accelerator.is_main_process:
    _ts = datetime.datetime.now().strftime("%Y-%m-%d--%H-%M-%S")
    run_name = args.run_name if getattr(args, 'run_name', '') else _ts
    model_save_path = args.data_path + args.model_save_path + args.generation_type + '/' + args.sub + '/' + run_name
    os.makedirs(model_save_path, exist_ok=True)
    print(model_save_path)
    
    # Check if resuming from checkpoint
    is_resuming = args.checkpoint_resume != '' and train_state.step > 0
    log_mode = 'a' if is_resuming else 'w'
    
    log_info_txt = open(model_save_path + '/log.txt', log_mode)
    if is_resuming:
        log_info_txt.write(f'\n\n=== Resuming training from step {train_state.step} at {datetime.datetime.now()} ===\n')
    log_info_txt.write(ss)
    log_info_txt.flush()
    print(ss)
    print(get_parameter_number(model))

    # Initialize W&B if enabled
    use_wandb = args.use_wandb if hasattr(args, 'use_wandb') else False
    if use_wandb and accelerator.is_main_process:
        try:
            import wandb
            wandb_run_name = args.wandb_run_name if (hasattr(args, 'wandb_run_name') and args.wandb_run_name) else run_name
            wandb_config = {
                'sub': args.sub,
                'generation_type': args.generation_type,
                'lr': args.lr,
                'max_steps': args.max_steps,
                'batch_size': args.batch_size,
                'num_classes': args.num_classes,
                'eeg_channels': eeg_num_channels,
                'in_channels': args.in_channels,
                'out_channels': args.out_channels,
            }
            
            # Check if we should resume W&B run
            wandb_resume = 'allow' if is_resuming else None
            wandb_id = None
            if is_resuming:
                # Try to find existing W&B run ID
                wandb_id_file = f"{model_save_path}/.wandb_run_id"
                if os.path.exists(wandb_id_file):
                    with open(wandb_id_file, 'r') as f:
                        wandb_id = f.read().strip()
                    print(f"[W&B] Resuming run with ID: {wandb_id}")
                else:
                    print(f"[W&B] Warning: No W&B run ID found, creating new run")
            
            wandb.init(
                project=args.wandb_project if hasattr(args, 'wandb_project') else 'neuro-3d',
                entity=args.wandb_entity if (hasattr(args, 'wandb_entity') and args.wandb_entity) else None,
                name=wandb_run_name,
                id=wandb_id,
                resume=wandb_resume,
                config=wandb_config,
                dir=model_save_path
            )
            
            # Save W&B run ID for future resume
            if not is_resuming:
                wandb_id_file = f"{model_save_path}/.wandb_run_id"
                with open(wandb_id_file, 'w') as f:
                    f.write(wandb.run.id)
                print(f"[W&B] Saved run ID: {wandb.run.id}")
            
            print(f"[W&B] {'Resumed' if is_resuming else 'Initialized'} run: {wandb_run_name}")
        except ImportError:
            print("[W&B] wandb not installed. Install with: pip install wandb")
            use_wandb = False
        except Exception as e:
            print(f"[W&B] Failed to initialize: {e}")
            print("[W&B] Tip: Run 'wandb login' first if not logged in")
            use_wandb = False

    # Initialize CSV logging
    if accelerator.is_main_process:
        import csv
        csv_path = f"{model_save_path}/train_log.csv"
        csv_mode = 'a' if is_resuming else 'w'
        csv_file = open(csv_path, csv_mode, newline='')
        csv_writer = csv.DictWriter(csv_file, fieldnames=[
            'step', 'lr', 'train_loss', 'loss_dm', 'grad_norm',
            'test_video_acc', 'best_video_acc'
        ])
        if not is_resuming:
            csv_writer.writeheader()
        csv_file.flush()
        print(f"[CSV] {'Appending to' if is_resuming else 'Logging to'}: {csv_path}")
    else:
        csv_writer = None
        csv_file = None
    
    best_video_acc = 0
    training_start_time = time.time()
    
    while True:
        video_acc_count_all, total_all = 0, 0
        for i, batch in enumerate(dataloader_train):
            model.train()
            if args.generation_type == 'shape':
                point_c = None
                pc = batch['point_cloud'].float()[:, :, :3]
                eeg_data = batch['eeg_data'].float()
                eeg_data2 = batch['eeg_data2'].float()
                point_features, video_features = batch['color_point_fea'].float(), batch['color_video_fea'].float()
                labels = batch['cls_label']
            elif args.generation_type == 'color':
                point_c = batch['point_cloud'].float()[:, :, :3]
                pc = batch['point_cloud'].float()[:, :, 3:]
                eeg_data = batch['eeg_data'].float()
                eeg_data2 = batch['eeg_data2'].float()
                point_features, video_features = batch['color_point_fea'].float(), batch['color_video_fea'].float()
                labels = batch['color_label']
            else:
                return
            fea_list = {'point_features': point_features, 'video_features': video_features,
            'point_features_all':point_features_train_all, 'video_features_all':video_features_train_all}
            with accelerator.accumulate(model):
                loss_dm = model(pc, eeg_data, eeg_data2, mode='train', shape_c=point_c, fea_list=fea_list, labels=labels)
                loss = loss_dm

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    if args.clip_grad_norm is not None:
                        accelerator.clip_grad_norm_(model.parameters(), args.clip_grad_norm)
                    grad_norm_clipped = training_utils.compute_grad_norm(model.parameters())
                optimizer.step()
                optimizer.zero_grad()
                if accelerator.sync_gradients:
                    scheduler.step()
                    train_state.step += 1
                loss_value = loss.item()
            if accelerator.sync_gradients:
                if accelerator.is_main_process and train_state.step % args.log_step_freq == 0:
                    video_acc = retrieval_test(args, model, dataloader_test, point_features_test_all, video_features_test_all)
                    if video_acc > best_video_acc:
                        best_video_acc = video_acc
                    step_s = str(train_state.step).zfill(7)
                    ss = '\nstep: ' + step_s + ',  '
                    ss += f'lr: {optimizer.param_groups[0]["lr"]}, train_loss:{loss_value} DM:{float(loss_dm)}\n'
                    ss += f'test video_acc:{video_acc:.4f}\n'
                    ss += f'test best video_acc:{best_video_acc:.4f}\n'
                    log_info_txt.write(ss)
                    log_info_txt.flush()
                    print(ss)

                    # Log to W&B
                    if use_wandb:
                        elapsed_time = time.time() - training_start_time
                        wandb.log({
                            'step': train_state.step,
                            'train/lr': optimizer.param_groups[0]["lr"],
                            'train/loss': loss_value,
                            'train/loss_dm': float(loss_dm),
                            'train/grad_norm': grad_norm_clipped,
                            'test/video_acc': video_acc,
                            'test/best_video_acc': best_video_acc,
                            'time/elapsed_hours': elapsed_time / 3600,
                            'time/elapsed_minutes': elapsed_time / 60,
                        }, step=train_state.step)

                    # Log to CSV
                    if csv_writer is not None:
                        csv_writer.writerow({
                            'step': train_state.step,
                            'lr': optimizer.param_groups[0]["lr"],
                            'train_loss': loss_value,
                            'loss_dm': float(loss_dm),
                            'grad_norm': grad_norm_clipped,
                            'test_video_acc': video_acc,
                            'best_video_acc': best_video_acc,
                        })
                        csv_file.flush()

                if accelerator.is_main_process and (train_state.step % args.checkpoint_freq == 0 or train_state.step == 1):
                    
                    checkpoint_dict = {
                        'model': accelerator.unwrap_model(model).state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'scheduler': scheduler.state_dict(),
                        'epoch': train_state.epoch,
                        'step': train_state.step,
                        'best_val': train_state.best_val
                    }
                    checkpoint_path = f'{model_save_path}/checkpoint-{train_state.step}.pth'
                    accelerator.save(checkpoint_dict, checkpoint_path)
                    print(f'Saved checkpoint to {Path(checkpoint_path).resolve()}')
                if train_state.step >= args.max_steps:
                    total_training_time = time.time() - training_start_time
                    print(f'Ending training at: {datetime.datetime.now()}')
                    print(f'Final train state: {train_state}')
                    print(f'[TIME] Total training time: {total_training_time/3600:.2f} hours ({total_training_time/60:.2f} minutes)')
                    ss = f'Ending training at: {datetime.datetime.now()}\n'
                    ss += f'Final train state: {train_state}\n'
                    ss += f'Total training time: {total_training_time/3600:.2f} hours\n'
                    log_info_txt.write(ss)
                    log_info_txt.flush()
                    log_info_txt.close()
                    if accelerator.is_main_process:
                        if csv_file is not None:
                            csv_file.close()
                            print(f"[CSV] Metrics saved to: {csv_path}")
                        if use_wandb:
                            wandb.log({'time/total_hours': total_training_time / 3600})
                            wandb.finish()
                            print("[W&B] Run finished")
                    return
                if train_state.step >= 100500:
                    log_info_txt.flush()
                    log_info_txt.close()
                    return

def visualize(args, model, dataloader_test, accelerator, epoch_step, point_features_test_all, video_features_test_all, model_save_path, infer_steps=200, num=1):
    model.eval()
    for num_index in range(0, num):
        time_b = time.time()
        video_acc_count_all, total_all = 0, 0
        for batch_idx, batch in enumerate(dataloader_test):
            if args.generation_type == 'shape':
                point_c = None
                pc = batch['point_cloud'].float()[:, :, :3]
                eeg_data = batch['eeg_data'].float()
                eeg_data2 = batch['eeg_data2'].float()
                point_features, video_features = batch['color_point_fea'].float(), batch['color_video_fea'].float()
                labels = batch['cls_label']
            elif args.generation_type == 'color':
                point_c = batch['point_cloud'].float()[:, :, :3]
                pc = batch['point_cloud'].float()[:, :, 3:]
                # c1, c2 = batch['color_video_fea'].float(), batch['color_point_fea'].float()
                eeg_data = batch['eeg_data'].float()
                eeg_data2 = batch['eeg_data2'].float()
                point_features, video_features = batch['color_point_fea'].float(), batch['color_video_fea'].float()
                labels = batch['color_label']
            else:
                return
            fea_list = {'point_features': point_features, 'video_features': video_features,
            'point_features_all':point_features_test_all, 'video_features_all':video_features_test_all}
            output, (video_acc_count, total, acc_list) = model(pc, eeg_data, eeg_data2, mode='sample', shape_c=point_c, fea_list=fea_list, labels=labels,
                                                                    return_sample_every_n_steps=-1, num_inference_steps=infer_steps, disable_tqdm=(not accelerator.is_main_process))
            video_acc_count_all += video_acc_count
            total_all += total
            for ii in range(0, output.shape[0]):
                point_pred, point_lbl = output[ii].detach().cpu().numpy(), pc[ii].detach().cpu().numpy()
                point_pred, point_lbl = output[ii].detach().cpu().numpy(), pc[ii].detach().cpu().numpy()
                pcd = o3d.geometry.PointCloud()
                if args.generation_type == 'shape':
                    pcd.points = o3d.utility.Vector3dVector(point_pred)
                elif args.generation_type == 'color':
                    pcd.points = o3d.utility.Vector3dVector(point_c[ii].detach().cpu().numpy())
                    point_pred = (point_pred + 1.0) / 2.0
                    # print(point_pred.min(), point_pred.max(), point_lbl.min(), point_lbl.max())
                    point_pred[point_pred <= 0] = 0
                    point_pred[point_pred >= 1] = 1.0
                    pcd.colors = o3d.utility.Vector3dVector(point_pred)
                name = batch['name'][ii]
                if acc_list[ii]:
                    print(name * 10)
                o3d.io.write_point_cloud(f'{model_save_path}/{epoch_step}-{name}-{num_index}.ply', pcd)
        time_e = time.time()
        print(time_e - time_b)
        print(f'step:{epoch_step}, testset, video acc:{(video_acc_count_all / total_all):.4f}')
    model.train()

def retrieval_test(args, model, dataloader_test, point_features_test_all, video_features_test_all):
    video_acc_count_all, total_all = 0, 0
    model.eval()
    for batch_idx, batch in enumerate(dataloader_test):
        if args.generation_type == 'shape':
            point_c = None
            pc = batch['point_cloud'].float()[:, :, :3]
            eeg_data = batch['eeg_data'].float()
            eeg_data2 = batch['eeg_data2'].float()
            point_features, video_features = batch['color_point_fea'].float(), batch['color_video_fea'].float()
            labels = batch['cls_label']
        elif args.generation_type == 'color':
            point_c = batch['point_cloud'].float()[:, :, :3]
            pc = batch['point_cloud'].float()[:, :, 3:]
            eeg_data = batch['eeg_data'].float()
            eeg_data2 = batch['eeg_data2'].float()
            point_features, video_features = batch['color_point_fea'].float(), batch['color_video_fea'].float()
            labels = batch['color_label']
        else:
            return
        fea_list = {'point_features': point_features, 'video_features': video_features,
        'point_features_all':point_features_test_all, 'video_features_all':video_features_test_all}
        (video_acc_count, total, acc_list) = model(pc, eeg_data, eeg_data2, mode='test_retrieval', shape_c=point_c, fea_list=fea_list, labels=labels)
        video_acc_count_all += video_acc_count
        total_all += total
    model.train()
    return video_acc_count_all / total_all

if __name__ == '__main__':
    main()

