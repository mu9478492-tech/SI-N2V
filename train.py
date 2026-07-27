# train.py
import os
import argparse
import yaml
import random
import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset import get_datasets
from model import build_model  # Ensure your model.py is included in the directory
from trainer import SIN2VTrainer

def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

def main():
    seed_everything(42)
    parser = argparse.ArgumentParser(description="SIN2V Training")
    parser.add_argument('--exp_name', type=str, required=True, help="Experiment name")
    args = parser.parse_args()

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    with open('config.yaml.yaml', 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)

    run_root = os.path.join(cfg.get('save_path') or "./experiments", args.exp_name)
    ckpt_dir, vis_root = os.path.join(run_root, "checkpoints"), os.path.join(run_root, "visualizations")
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(vis_root, exist_ok=True)

    train_ds, test_ds = get_datasets(cfg)
    train_loader = DataLoader(train_ds, batch_size=cfg.get('batch_size') or 12, shuffle=True, num_workers=cfg.get('num_workers') or 4, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False) if test_ds else None

    model = build_model(cfg).to(device)
    trainer = SIN2VTrainer(model=model, train_loader=train_loader, test_loader=test_loader, cfg=cfg, device=device, save_dirs=(ckpt_dir, vis_root))

    total_epochs = cfg.get('epochs') or 100
    for epoch in range(total_epochs):
        loss = trainer.train_epoch(epoch)
        print(f"Epoch {epoch}/{total_epochs} | Loss: {loss:.5f}")
        torch.save(model.state_dict(), os.path.join(ckpt_dir, 'latest_model.pth'))
        if (epoch % 5 == 0) or (epoch == total_epochs - 1):
            torch.save(model.state_dict(), os.path.join(ckpt_dir, f'ep_{epoch:03d}.pth'))
            if test_loader: trainer.test_inference(epoch)

if __name__ == "__main__":
    main()