import copy
import torch
from torch import optim
from att_network_tnb101_macro import NASLoss, DynamicNASFramework
from torch_geometric.data import Batch


class OnlineTrainer:
    def __init__(self, pretrained_path, num_metrics, max_cycles, loss_threshold, diff_threshold, total_budget=25, device='cuda'):

        self.num_metrics = num_metrics
        self.pretrained_path = pretrained_path
        self.max_cycles = max_cycles
        self.loss_threshold = loss_threshold
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.model = self._load_pretrained(self.pretrained_path, False, self.num_metrics)
        self.model.to(self.device)
        self.total_budget = total_budget
        self.collected_data = []
        self.criterion = NASLoss(diff_threshold, alpha=0.5, beta=0.3)
        self.reload_count = 0
        self.error_count = 0
        self.best_direction_loss = float('inf')
        self.best_total_loss = float('inf')
        self.best_corr_loss = float('inf')
        self.best_align_loss = float('inf')
        self.no_improve_steps = 0
        self.patience = 100
        self.cycle_count = 0
        self.max_cycles = max_cycles
        self.best_weights = None
        self.current_lr = 3e-4

    def _load_pretrained(self, path, metrics_pred, num_metrics):
        checkpoint = torch.load(path, map_location='cpu')

        model = DynamicNASFramework(
            num_metrics=num_metrics,
            pretrain_mode=metrics_pred
        )

        required_keys = ['arch_encoder', 'metric_head']
        for key in required_keys:
            if key not in checkpoint:
                raise ValueError(f"The checkpoint is missing the necessary keys: {key}")

        model.arch_encoder.load_state_dict(checkpoint['arch_encoder'])
        if metrics_pred:
            model.metric_head.load_state_dict(checkpoint['metric_head'])

        best_test_loss = checkpoint.get('test_loss', float('inf'))
        epoch = checkpoint.get('epoch', 0)

        print(f"Load the pretrain model, Epoch: {epoch}, Best test loss: {best_test_loss:.4e}")
        return model

    def select_and_train(self, step, new_arch_data=None):

        if new_arch_data is not None:
            for data in new_arch_data:
                self.collected_data.append({
                    'arch': data['arch'].to(self.device),
                    'metric_ids': data['metric_ids'].to(self.device),
                    'metrics': data['metrics'].to(self.device),
                    'true_score': data['true_score'].to(self.device)
                })

        self.best_total_loss = 10000
        self.cycle_count = 0
        self.no_improve_steps = 0
        self.reload_count = 0
        self.best_weights = None
        self.model = self._load_pretrained(self.pretrained_path, False, self.num_metrics)
        self.model.to(self.device)
        step_0 = step
        while self.reload_count < self.max_cycles:
            if self.best_weights is not None:
                self.model.load_state_dict(self.best_weights)
            step = int(step + self.cycle_count * step_0 / 10)
            if self.cycle_count == 100:
                break

            optimizer = optim.AdamW(
                self.model.parameters(),
                lr=self.current_lr,
                weight_decay=0.01,
                betas=(0.9, 0.999)
            )
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=step)

            early_stop_flag = self._train_with_early_stop(optimizer, scheduler, step)

            if self.best_total_loss <= self.loss_threshold:
                break

            if early_stop_flag:
                self._trigger_restart_policy()

        return {'total_loss': self.best_total_loss,
                'corr_loss': self.best_corr_loss,
                'align_loss': self.best_align_loss,
                'direction_loss': self.best_direction_loss,
                }

    def _train_with_early_stop(self, optimizer, scheduler, max_steps):

        self.model.train()
        for current_step in range(max_steps):
            batch = self._prepare_batch()

            optimizer.zero_grad()

            outputs = self.model(
                arch=batch['arch'],
                metric_ids=batch['metric_ids'],
                metrics=batch['metrics']
            )

            loss_dict = self._compute_loss(outputs, batch)

            current_total_loss = loss_dict['loss'].item()
            current_corr_loss = loss_dict['corr']
            current_align_loss = loss_dict['align']
            current_direction_loss = loss_dict['direction']
            if current_total_loss < self.best_total_loss:
                self.best_total_loss = current_total_loss
                self.best_corr_loss = current_corr_loss
                self.best_align_loss = current_align_loss
                self.best_direction_loss = current_direction_loss
                self.no_improve_steps = 0
                self.best_weights = copy.deepcopy(self.model.state_dict())
            else:
                self.no_improve_steps += 1
            if self.no_improve_steps >= self._dynamic_patience():
                return True

            if current_total_loss <= self.loss_threshold and self.error_count == 0:
                self.model.eval()
                return False

            loss_dict['loss'].backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

        return False


    def _dynamic_patience(self):
        base = self.patience
        return int(base + self.cycle_count * self.patience / 10)

    def _trigger_restart_policy(self):
        self.cycle_count += 1

    def _prepare_batch(self):
        batch = {
            'arch': Batch.from_data_list([d['arch'] for d in self.collected_data]),
            'metric_ids': torch.stack([d['metric_ids'] for d in self.collected_data]),
            'metrics': torch.stack([d['metrics'] for d in self.collected_data]),
            'true_score': torch.cat([d['true_score'] for d in self.collected_data])
        }
        return batch

    def _compute_loss(self, outputs, batch):
        targets = {
            'true_scores': batch['true_score']
        }

        loss_dict = self.criterion(
            outputs=outputs,
            targets=targets,
            mode='online'
        )
        return loss_dict

    def save_checkpoint(self, path):
        torch.save({
            'model_state': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'collected_data': self.collected_data,
            'budget_used': len(self.collected_data)
        }, path)

