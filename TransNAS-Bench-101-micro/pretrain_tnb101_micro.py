import torch
import torch.optim as optim
from collections import defaultdict
from att_network_tnb101_micro import NASLoss
import time
import os
import math


def evaluate(model, dataloader, loss_fn, device, metric_names):
    model.eval()
    total_samples = 0
    test_metrics = defaultdict(float)

    with torch.no_grad():
        for batch in dataloader:
            arch_batch = batch['arch'].to(device)
            metrics = batch['metrics'].to(device)

            metrics = torch.nan_to_num(metrics, nan=1.0)
            metrics[metrics == 0] = 1e-6

            outputs = model(arch_batch)

            loss_dict = loss_fn(
                outputs={
                    'metric_pred': outputs['metric_pred'],
                    'recon_arch': None,
                },
                targets={
                    'metrics': metrics,
                    'arch_feat': None,
                },
                mode='pretrain_test'
            )

            batch_size = metrics.size(0)
            total_samples += batch_size
            test_metrics['mse'] += loss_dict['mse'] * batch_size

            per_metric_mse = torch.mean(
                (outputs['metric_pred'] - metrics) ** 2,
                dim=0
            ).cpu().tolist()
            for idx, name in enumerate(metric_names):
                test_metrics[name] += per_metric_mse[idx] * batch_size

    avg_mse = test_metrics['mse'] / total_samples
    metric_avg = {name: test_metrics[name] / total_samples for name in metric_names}
    return avg_mse, metric_avg


def run_pretrain(model, train_loader, test_loader, device, metric_names, save_dir, diff_threshold, epochs=100, lr=5e-3):

    model.train()
    optimizer = optim.AdamW(model.parameters(), lr=lr)

    total_steps = epochs * len(train_loader)
    warmup_steps = int(0.1 * total_steps)

    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: (
            step / warmup_steps if step < warmup_steps
            else 0.5 * (1 + math.cos(math.pi * (step - warmup_steps) / (total_steps - warmup_steps)))
        )
    )

    loss_fn = NASLoss(diff_threshold, alpha=0.5, beta=0.3)

    best_test_loss = float('inf')
    os.makedirs(save_dir, exist_ok=True)

    test_metrics = {}
    for epoch in range(1, epochs + 1):
        epoch_start = time.time()
        train_metrics = defaultdict(float)
        train_total_mse = 0.0
        train_total_recon = 0.0
        train_total_samples = 0

        model.train()
        for batch in train_loader:
            arch_batch = batch['arch'].to(device)
            metrics = batch['metrics'].to(device)

            metrics = torch.nan_to_num(metrics, nan=1.0)
            metrics[metrics == 0] = 1e-6
            outputs = model(arch_batch)

            loss_dict = loss_fn(
                outputs={
                    'metric_pred': outputs['metric_pred'],
                    'recon_arch': outputs['recon_arch']
                },
                targets={
                    'metrics': metrics,
                    'arch_feat': arch_batch.x,
                },
                mode='pretrain_train'
            )

            optimizer.zero_grad()
            loss_dict['loss'].backward()
            optimizer.step()
            scheduler.step()

            batch_size = metrics.size(0)
            train_total_samples += batch_size
            train_total_mse += loss_dict['mse'] * batch_size
            train_total_recon += loss_dict['recon'] * batch_size

            with torch.no_grad():
                per_metric_mse = torch.mean(
                    (outputs['metric_pred'] - metrics) ** 2,
                    dim=0
                ).cpu().tolist()
                for idx, name in enumerate(metric_names):
                    train_metrics[name] += per_metric_mse[idx] * batch_size

        test_avg_mse, test_metric_avg = evaluate(
            model, test_loader, loss_fn, device, metric_names
        )

        train_avg_mse = train_total_mse / train_total_samples
        train_avg_recon = train_total_recon / train_total_samples
        train_metric_avg = {name: train_metrics[name] / train_total_samples for name in metric_names}

        train_str = " | ".join([f"{k}:{v:.3e}" for k, v in train_metric_avg.items()])
        test_str = " | ".join([f"{k}:{v:.3e}" for k, v in test_metric_avg.items()])
        print(
            f"Epoch {epoch:03d} | Time: {time.time() - epoch_start:.1f}s\n"
            f"  [Train] MSE: {train_avg_mse:.3e} | Recon: {train_avg_recon:.3e} | {train_str}\n"
            f"  [Test]  MSE: {test_avg_mse:.3e} | {test_str}"
        )

        current_test_loss = test_avg_mse

        for k, v in test_metric_avg.items():
            if k not in test_metrics:
                test_metrics[k] = []
            test_metrics[k].append(v)

        if current_test_loss < best_test_loss:
            best_test_loss = current_test_loss
            torch.save({
                'arch_encoder': model.arch_encoder.state_dict(),
                'metric_head': model.metric_head.state_dict(),
                'test_loss': best_test_loss,
                'epoch': epoch
            }, f"{save_dir}/best_model.pth")
            print(f"Epoch {epoch}: Save the best model, test loss: {best_test_loss:.4e}")


