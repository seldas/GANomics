import os
import time
import argparse
import yaml
import torch
from torch.utils.data import DataLoader
from src.datasets.genomics_dataset import GenomicsDataset
from src.models.ganomics_model import GANomicsModel

def parse_args():
    parser = argparse.ArgumentParser(description="Train GANomics Model")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML configuration file")
    parser.add_argument("--max_samples", type=int, help="Override max_samples in config")
    parser.add_argument("--name", type=str, help="Override output name in config")
    parser.add_argument("--n_epochs", type=int, help="Override n_epochs in config")
    parser.add_argument("--n_epochs_decay", type=int, help="Override n_epochs_decay in config")
    return parser.parse_args()

def train():
    args = parse_args()
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # CLI Overrides
    if args.max_samples:
        config['dataset']['max_samples'] = args.max_samples
    if args.name:
        config['output']['name'] = args.name
    if args.n_epochs:
        config['train']['n_epochs'] = args.n_epochs
    if args.n_epochs_decay:
        config['train']['n_epochs_decay'] = args.n_epochs_decay

    # Set device
    device = torch.device(config['train'].get('device', 'cpu') if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Create Dataset and DataLoader
    dataset = GenomicsDataset(
        config['dataset']['path_A'], 
        config['dataset']['path_B'], 
        is_train=True,
        max_samples=config['dataset'].get('max_samples')
    )
    dataloader = DataLoader(dataset, batch_size=config['train']['batch_size'], shuffle=True)
    
    print(f"Number of training samples: {len(dataset)}")

    # Initialize Model
    model = GANomicsModel(
        input_nc=config['model']['input_nc'],
        output_nc=config['model']['output_nc'],
        lr=config['optimizer']['lr'],
        betas=(config['optimizer']['beta1'], config['optimizer']['beta2']),
        lambda_A=config['model']['lambda_A'],
        lambda_B=config['model']['lambda_B'],
        lambda_feedback=config['model']['lambda_feedback'],
        lambda_idt=config['model']['lambda_idt'],
        gan_mode=config['model']['gan_mode'],
        device=device
    )

    # Create Output Directories
    os.makedirs(config['output']['checkpoints_dir'], exist_ok=True)
    save_dir = os.path.join(config['output']['checkpoints_dir'], config['output']['name'])
    os.makedirs(save_dir, exist_ok=True)
    
    log_dir = config['output'].get('logs_dir', 'results/logs')
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"{config['output']['name']}_log.txt")
    
    with open(log_file, 'a') as f:
        f.write(f"================ Training Loss ({time.ctime()}) ================\n")

    # Training Loop
    n_epochs = config['train']['n_epochs']
    n_epochs_decay = config['train']['n_epochs_decay']
    total_epochs = n_epochs + n_epochs_decay
    explosion_factor = config['train'].get('explosion_factor', 3.0)
    
    loss_history = []
    epoch = 1
    while epoch <= total_epochs:
        epoch_start_time = time.time()
        iter_data_time = time.time()
        
        for i, data in enumerate(dataloader):
            iter_start_time = time.time()
            t_data = iter_start_time - iter_data_time
            
            model.set_input(data)
            model.optimize_parameters()
            
            t_comp = (time.time() - iter_start_time) / config['train']['batch_size']
            
            losses = model.get_current_losses()
            curr_total = losses.get('G_total', 0)
            loss_history.append(curr_total)
            
            # explosion check (vs rolling mean of last 50 iters)
            if len(loss_history) > 50:
                avg_recent = sum(loss_history[-50:]) / 50
                if curr_total > explosion_factor * avg_recent:
                    print(f"!!! Loss explosion detected (G_total={curr_total:.2f}, avg={avg_recent:.2f})")
                    print(f"!!! Rolling back to net_latest.pth and resuming...")
                    latest_path = os.path.join(save_dir, "net_latest.pth")
                    if os.path.exists(latest_path):
                        model.load_networks(latest_path)
                        for optimizer in [model.optimizer_G, model.optimizer_D]:
                            for param_group in optimizer.param_groups:
                                param_group['lr'] *= 0.5
                        loss_history = []
                        continue 

            if i % config['train']['print_freq'] == 0:
                # Format log like example.txt
                # (epoch: 2, iters: 50, time: 0.089, data: 0.182) D_A: 22.749 ...
                log_msg = f"(epoch: {epoch}, iters: {i}, time: {t_comp:.3f}, data: {t_data:.3f}) "
                for name in model.loss_names:
                    val = losses.get(name, 0)
                    log_msg += f"{name}: {val:.3f} "
                
                print(log_msg)
                with open(log_file, 'a') as f:
                    f.write(log_msg + "\n")
            
            iter_data_time = time.time()
                
        print(f"End of epoch {epoch} / {total_epochs} \t Time Taken: {time.time() - epoch_start_time:.2f} sec")

        if epoch % config['train']['save_epoch_freq'] == 0:
            save_path = os.path.join(save_dir, f"net_epoch_{epoch}.pth")
            model.save_networks(save_path)
            model.save_networks(os.path.join(save_dir, "net_latest.pth"))
            print(f"Saved stable checkpoint to {save_path}")
        
        epoch += 1 # Only increment if successful

if __name__ == "__main__":
    train()
