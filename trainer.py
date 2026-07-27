# trainer.py
import os
import torch
import torch.optim as optim
from PIL import Image
import numpy as np
from tqdm import tqdm

from utils import percentile_normalize, get_luminance_weight, apply_probabilistic_mask


class SIN2VTrainer:
    def __init__(self, model, train_loader, test_loader, cfg, device, save_dirs=None):
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.device = device
        self.save_dirs = save_dirs

        self.accum_iter = cfg.get('accum_iter') or 1
        self.optimizer = optim.AdamW(model.parameters(), lr=float(cfg.get('lr') or 0.0001),
                                     weight_decay=float(cfg.get('weight_decay') or 0.00005))
        self.scaler = torch.cuda.amp.GradScaler()

        print("Trainer initialized: [SIN2V] - Structure & Intensity Priors Active")

    def train_epoch(self, epoch):
        self.model.train()
        epoch_loss = 0
        pbar = tqdm(self.train_loader, desc=f"Ep {epoch}", dynamic_ncols=True)
        self.optimizer.zero_grad()

        for i, batch in enumerate(pbar):
            raw = batch[0].to(self.device, non_blocking=True)
            if torch.isnan(raw).any(): continue

            raw = percentile_normalize(raw)

            with torch.cuda.amp.autocast():
                v1_masked, mask1, v1_target = apply_probabilistic_mask(raw)
                pred1 = self.model(v1_masked)
                mse_recon = (pred1 - v1_target) ** 2
                w = get_luminance_weight(v1_target)
                loss = (mse_recon * w * mask1).sum() / (mask1.sum() + 1e-8)
                loss = loss / self.accum_iter

            self.scaler.scale(loss).backward()
            if (i + 1) % self.accum_iter == 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()

            epoch_loss += loss.item() * self.accum_iter

        return epoch_loss / len(self.train_loader)

    def test_inference(self, epoch):
        if self.test_loader is None: return
        self.model.eval()
        save_dir = os.path.join(self.save_dirs[1], f'epoch_{epoch:03d}')
        os.makedirs(save_dir, exist_ok=True)

        with torch.no_grad():
            for batch in self.test_loader:
                raw, filename = batch[0].to(self.device), batch[1][0]
                input_raw = percentile_normalize(raw)
                pred = torch.clamp(self.model(input_raw), 0, 1)

                vis = torch.cat([input_raw, pred], dim=3)
                vis_np = (vis[0, 0].cpu().numpy() * 255).astype(np.uint8)
                name, _ = os.path.splitext(filename)
                Image.fromarray(vis_np).save(os.path.join(save_dir, f"{name}_pred.png"))