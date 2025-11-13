import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='')
    parser.add_argument('--model_save_path', type=str, default='model/point_generate/')
    parser.add_argument('--sub', type=str, default='sub10')
    parser.add_argument('--task', type=str, default='train')
    parser.add_argument('--pretrain_model', type=str, default='')

    parser.add_argument('--max_steps', type=int, default=100000)
    parser.add_argument('--log_step_freq', type=int, default=10)
    parser.add_argument('--checkpoint_freq', type=int, default=20000)
    parser.add_argument('--test_freq', type=int, default=5000)


    parser.add_argument('--mixed_precision', type=str, default='fp16')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1)
    parser.add_argument('--seed', type=int, default=1234)

    parser.add_argument('--beta_start', type=float, default=1e-5)
    parser.add_argument('--beta_end', type=float, default=8e-3)
    parser.add_argument('--beta_schedule', type=str, default='linear')
    parser.add_argument('--point_cloud_model_embed_dim', type=int, default=64)
    parser.add_argument('--in_channels', type=int, default=128+3)
    parser.add_argument('--out_channels', type=int, default=3)
    parser.add_argument('--generation_type', type=str, default='shape', choices=['shape', 'color'])
    parser.add_argument('--model_type', type=str, default='', choices=['', 'dynamic', 'static', 'concat', 'NoDe'])

    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=1e-6)
    parser.add_argument('--scheduler_type', type=str, default='cosine')
    parser.add_argument('--batch_size', type=int, default=48)
    parser.add_argument('--num_workers', type=int, default=4)
    
    parser.add_argument('--checkpoint_resume', type=str, default='')
    parser.add_argument('--clip_grad_norm', type=float, default=50.0)
    parser.add_argument('--ply_point_path', type=str, default='')
    # PLY selection: all (default) will output k=0..4; k selects a specific index via --ply_k; best aliases k=0
    parser.add_argument('--ply_mode', type=str, default='all', choices=['all', 'k', 'best'])
    parser.add_argument('--ply_k', type=int, default=0)
    
    # EEG data configuration
    parser.add_argument('--num_classes', type=int, default=72, help='Number of object classes to use (default: 72)')
    parser.add_argument('--use_32_channels', action='store_true', help='Use 32-channel EEG layout instead of 64')
    parser.add_argument('--eeg_channels', type=int, default=0, choices=[0, 22, 32, 64], help='Explicit EEG channel count override (0=auto/legacy flag)')

    # Run naming
    parser.add_argument('--run_name', type=str, default='', help='Custom run name for output directories; if empty, uses timestamp')

    # Logging and tracking
    parser.add_argument('--use_wandb', action='store_true', help='Enable Weights & Biases logging')
    parser.add_argument('--wandb_project', type=str, default='neuro-3d', help='W&B project name')
    parser.add_argument('--wandb_entity', type=str, default='', help='W&B entity/username (optional)')
    parser.add_argument('--wandb_run_name', type=str, default='', help='W&B run name (defaults to run_name)')

    opt = parser.parse_args()
    return opt