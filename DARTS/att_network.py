import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool


class EnhancedGATEncoder(nn.Module):
    def __init__(self, input_dim=14, hidden_dim=128, num_heads=4, edge_dim=7):
        super().__init__()
        self.conv1 = GATConv(input_dim, hidden_dim // num_heads, heads=num_heads, edge_dim=edge_dim)
        self.conv2 = GATConv(hidden_dim, hidden_dim // num_heads, heads=num_heads, edge_dim=edge_dim)
        self.mask_token = nn.Parameter(torch.randn(1, input_dim))
        self.recon_head = nn.Linear(hidden_dim, input_dim)

    def forward(self, data, mask_ratio=0.0):
        if mask_ratio > 0:
            data = self._random_mask(data, mask_ratio)

        x = F.elu(self.conv1(data.x, data.edge_index, edge_attr=data.edge_attr))
        x = F.elu(self.conv2(x, data.edge_index, edge_attr=data.edge_attr))
        graph_emb = global_mean_pool(x, data.batch)

        recon_pred = self.recon_head(x) if self.training else None
        return graph_emb, recon_pred

    def _random_mask(self, data, ratio=0.2):
        mask_nodes = torch.rand(data.x.size(0)) < ratio
        data.x[mask_nodes] = self.mask_token
        return data


class DynamicNASFramework(nn.Module):
    def __init__(self, num_metrics, embed_dim=256, hidden_dim=128, pretrain_mode=True):
        super().__init__()
        self.norm_encoder = EnhancedGATEncoder()
        self.reduce_encoder = EnhancedGATEncoder()
        self.pretrain_mode = pretrain_mode
        if pretrain_mode:
            self.metric_head = nn.Sequential(
                nn.Linear(256, 128),
                nn.GELU(),
                nn.Linear(128, num_metrics)
            )
        else:
            for param in self.norm_encoder.parameters():
                param.requires_grad = False
            for param in self.reduce_encoder.parameters():
                param.requires_grad = False

                self.arch_proj = nn.Sequential(
                    nn.Linear(embed_dim, hidden_dim * 2),
                    nn.LayerNorm(hidden_dim * 2),
                    nn.GELU(),
                    nn.Linear(hidden_dim * 2, hidden_dim)
                )

                self.metric_embed = nn.Embedding(num_metrics, hidden_dim)
                self.metric_proj = nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim * 2),
                    nn.GELU(),
                    nn.Linear(hidden_dim * 2, hidden_dim)
                )

                self.cross_attn = nn.MultiheadAttention(
                    embed_dim=hidden_dim,
                    num_heads=8,
                    batch_first=True
                )
                self.gate = nn.Sequential(
                    nn.Linear(2 * hidden_dim, hidden_dim),
                    nn.Sigmoid(),
                )

                self.weight_mlp = nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim * 2),
                    nn.LayerNorm(hidden_dim * 2),
                    nn.GELU(),
                    nn.Linear(hidden_dim * 2, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.GELU(),
                    nn.Linear(hidden_dim, 1)
                )

    def forward(self, norm_cell, reduce_cell, metric_ids=None, metrics=None):
        z_norm, recon_norm = self.norm_encoder(norm_cell, mask_ratio=0.2 if self.training else 0)
        z_red, recon_red = self.reduce_encoder(reduce_cell, mask_ratio=0.2 if self.training else 0)
        combined = torch.cat([z_norm, z_red], dim=1)

        if self.pretrain_mode:
            metric_pred = self.metric_head(combined)
            return {
                'metric_pred': metric_pred,
                'recon_norm': recon_norm,
                'recon_red': recon_red
            }
        else:
            metric_emb = self.metric_embed(metric_ids)
            metric_emb = self.metric_proj(metric_emb) + metric_emb

            arch_feat = self.arch_proj(combined)

            attn_out, _ = self.cross_attn(
                query=metric_emb,
                key=arch_feat.unsqueeze(1),
                value=arch_feat.unsqueeze(1),
                need_weights=False
            )

            gate_input = torch.einsum('bmh,bh->bmh', attn_out, arch_feat)
            gate = self.gate(torch.cat([gate_input, metric_emb.expand_as(attn_out)], dim=-1))
            fused_feat = attn_out * gate + attn_out

            weights = self.weight_mlp(fused_feat).squeeze(-1)
            weights_abs_sum = weights.abs().sum(dim=1, keepdim=True) + 1e-8
            normalized_weights = weights / weights_abs_sum
            positive_weights = torch.relu(normalized_weights)
            negative_weights = torch.relu(-normalized_weights)
            positive_contribution = positive_weights * metrics
            negative_contribution = negative_weights * (1.0 - metrics)
            metric_score = (positive_contribution + negative_contribution).sum(dim=1, keepdim=True)

            return {
                'weights': normalized_weights,
                'score': metric_score,
            }

    def switch_mode(self, pretrain_mode):
        self.pretrain_mode = pretrain_mode
        if not pretrain_mode:
            for param in self.norm_encoder.parameters():
                param.requires_grad = False
            for param in self.reduce_encoder.parameters():
                param.requires_grad = False


class NASLoss(nn.Module):
    def __init__(self, diff_threshold, alpha=0.5, beta=0.3):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.diff_threshold = diff_threshold

    def forward(self, outputs, targets, mode):

        if mode.startswith('pretrain'):
            mse_loss = F.mse_loss(outputs['metric_pred'], targets['metrics'])
            if mode == 'pretrain_train':
                recon_norm = F.mse_loss(outputs['recon_norm'], targets['norm_feat'])
                recon_red = F.mse_loss(outputs['recon_red'], targets['red_feat'])
                recon_loss = 0.5 * (recon_norm + recon_red) * self.beta
                total_loss = mse_loss * self.alpha + recon_loss
                return {
                    'loss': total_loss,
                    'mse': mse_loss.item(),
                    'recon': recon_loss.item()
                }
            else:
                return {
                    'loss': mse_loss * self.alpha,
                    'mse': mse_loss.item(),
                    'recon': 0.0
                }

        elif mode == 'online':
            pred_scores = outputs['score'].squeeze(-1)
            true_score = targets['true_scores']

            corr_matrix = torch.corrcoef(torch.stack([pred_scores, true_score]))
            corr_loss = 1 - corr_matrix[0, 1]

            i, j = torch.combinations(torch.arange(len(pred_scores)), 2).unbind(1)
            delta_s = pred_scores[i] - pred_scores[j]
            delta_y = true_score[i] - true_score[j]

            diff_threshold = 0.1

            valid_mask = torch.where(torch.abs(delta_y) < diff_threshold, torch.ones_like(delta_y, dtype=torch.float32), (delta_s * delta_y > 0).float())

            delta_s_abs = (delta_s.abs() * valid_mask - delta_s.abs().mean()) / (delta_s.abs().std() + 1e-8)
            delta_y_abs = (delta_y.abs() * valid_mask - delta_y.abs().mean()) / (delta_y.abs().std() + 1e-8)
            align_loss = torch.mean(torch.abs(delta_s_abs - delta_y_abs))

            direction_loss = torch.mean(torch.relu(-delta_s * delta_y)) * 100

            total_loss = corr_loss + 0.5 * align_loss + direction_loss

            return {
                'loss': total_loss,
                'corr': corr_loss.item(),
                'align': align_loss.item(),
                'direction': direction_loss.item(),
            }

        else:
            raise ValueError(f"Invalid mode: {mode}")



