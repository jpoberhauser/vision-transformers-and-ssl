"""
Vanilla DETR — end-to-end implementation from scratch.

Pieces, in dependency order:
  - ToyDetectionDataset       toy synthetic detection dataset (colored shapes)
  - Backbone                  truncated ResNet + 1x1 projection
  - PositionalEncoding2D      2D sinusoidal PE (split y / x halves)
  - TransformerEncoderLayer   self-attn + FFN with residual + LN
  - TransformerEncoder        stack of N encoder layers
  - TransformerDecoderLayer   self-attn + cross-attn + FFN with residual + LN
  - TransformerDecoder        stack of N decoder layers, returns per-layer outputs
  - DetrPredictionHeads       class head (Linear) + bbox head (3-layer MLP + sigmoid)
  - DETR                      composes everything; output dict with aux_outputs
  - HungarianMatcher          bipartite matching via scipy.linear_sum_assignment
  - DetrLoss                  class CE + L1 + GIoU + aux supervision

Plus utilities:
  - detr_collate              DataLoader collate for variable-length targets
  - train_detr                training loop with split LR (backbone vs transformer)
  - predict_and_visualize     inference + cross-attention overlay
"""

from typing import Dict, List, Tuple

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from torchvision import models
from torchvision.ops import box_convert, generalized_box_iou

from scipy.optimize import linear_sum_assignment

import matplotlib.pyplot as plt
import matplotlib.patches as patches


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

def make_synthetic_sample(H: int = 224, W: int = 224, K: int = 5,
                          min_objects: int = 1, max_objects: int = 8):
    """One synthetic detection sample with colored rectangles on a black background.

    Returns:
        image:   [3, H, W] float tensor in [0, 1]
        targets: {'boxes':  [M, 4] cxcywh normalized to [0, 1],
                  'labels': [M] long tensor with values in [1, K]}
    """
    palette = torch.tensor([
        [1.0, 0.0, 0.0],   # red
        [0.0, 1.0, 0.0],   # green
        [0.0, 0.0, 1.0],   # blue
        [1.0, 1.0, 0.0],   # yellow
        [1.0, 0.0, 1.0],   # magenta
        [0.0, 1.0, 1.0],   # cyan
        [1.0, 0.5, 0.0],   # orange
        [0.5, 0.0, 1.0],   # purple
    ])[:K]
    assert K <= len(palette), "extend the palette for more classes"

    image = torch.zeros(3, H, W)
    M = np.random.randint(min_objects, max_objects + 1)

    boxes, labels = [], []
    for _ in range(M):
        cls = np.random.randint(1, K + 1)
        w = np.random.randint(W // 10, W // 3)
        h = np.random.randint(H // 10, H // 3)
        x1 = np.random.randint(0, W - w)
        y1 = np.random.randint(0, H - h)
        x2, y2 = x1 + w, y1 + h

        image[:, y1:y2, x1:x2] = palette[cls - 1][:, None, None]

        cx = (x1 + x2) / 2 / W
        cy = (y1 + y2) / 2 / H
        boxes.append([cx, cy, w / W, h / H])
        labels.append(cls)

    targets = {
        'boxes':  torch.tensor(boxes,  dtype=torch.float32),
        'labels': torch.tensor(labels, dtype=torch.long),
    }
    return image, targets


def visualize(image: Tensor, targets: Dict[str, Tensor], ax=None):
    """Plot an image with its ground-truth bounding boxes."""
    if ax is None:
        _, ax = plt.subplots(figsize=(5, 5))
    H, W = image.shape[1:]
    ax.imshow(image.permute(1, 2, 0).cpu().numpy().clip(0, 1))
    for box, label in zip(targets['boxes'], targets['labels']):
        cx, cy, bw, bh = box.tolist()
        x1, y1 = (cx - bw / 2) * W, (cy - bh / 2) * H
        ax.add_patch(patches.Rectangle(
            (x1, y1), bw * W, bh * H,
            linewidth=2, edgecolor='white', facecolor='none'))
        ax.text(x1, y1 - 4, f'cls {label.item()}',
                color='white', fontsize=9,
                bbox=dict(facecolor='black', alpha=0.6, pad=1))
    ax.axis('off')
    return ax


class ToyDetectionDataset(Dataset):
    """Pre-generated synthetic dataset of colored shapes for detection training."""

    def __init__(self, num_samples: int = 1000, image_size: int = 128,
                 min_objects: int = 1, max_objects: int = 5, num_classes: int = 3):
        self.num_samples = num_samples
        self.image_size = image_size
        self.max_objects = max_objects
        self.num_classes = num_classes
        self.images, self.targets = [], []
        for _ in range(num_samples):
            img, tgt = make_synthetic_sample(
                H=image_size, W=image_size,
                K=num_classes, min_objects=min_objects, max_objects=max_objects,
            )
            self.images.append(img)
            self.targets.append(tgt)
        self.images = torch.stack(self.images)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx: int):
        return self.images[idx], self.targets[idx]


def detr_collate(batch):
    """Stack images, keep targets as a list (variable M per image)."""
    images = torch.stack([b[0] for b in batch], dim=0)
    targets = [b[1] for b in batch]
    return images, targets


# ---------------------------------------------------------------------------
# Backbone
# ---------------------------------------------------------------------------

class Backbone(nn.Module):
    """Truncated torchvision ResNet + 1x1 conv projection to d_model channels."""

    def __init__(self, d_model: int = 256, name: str = 'resnet18'):
        super().__init__()
        backbone_fn = getattr(models, name)
        resnet = backbone_fn(weights='DEFAULT')
        c5_channels = resnet.fc.in_features
        self.body = nn.Sequential(*list(resnet.children())[:-2])
        self.projection = nn.Conv2d(c5_channels, d_model, kernel_size=1)

    def forward(self, x: Tensor) -> Tensor:
        x = self.body(x)
        x = self.projection(x)
        return x


# ---------------------------------------------------------------------------
# Positional encoding
# ---------------------------------------------------------------------------

class PositionalEncoding2D(nn.Module):
    """2D sinusoidal positional encoding: half channels encode y, half encode x."""

    def __init__(self, d_model: int = 256, temperature: float = 10000.0):
        super().__init__()
        assert d_model % 4 == 0, 'd_model must be divisible by 4'
        self.d_model = d_model
        self.temperature = temperature

    def forward(self, x: Tensor) -> Tensor:
        B, _, H, W = x.shape
        device = x.device
        num_freqs = self.d_model // 4
        freqs = self.temperature ** (
            2 * torch.arange(num_freqs, device=device).float() / (self.d_model // 2)
        )
        y_pos = torch.arange(H, device=device).float()
        x_pos = torch.arange(W, device=device).float()

        y_div = y_pos[:, None] / freqs[None, :]   # [H, d_model/4]
        x_div = x_pos[:, None] / freqs[None, :]   # [W, d_model/4]

        y_emb = torch.stack([y_div.sin(), y_div.cos()], dim=-1).flatten(-2)  # [H, d/2]
        x_emb = torch.stack([x_div.sin(), x_div.cos()], dim=-1).flatten(-2)  # [W, d/2]

        y_grid = y_emb[:, None, :].expand(H, W, -1)
        x_grid = x_emb[None, :, :].expand(H, W, -1)
        pe = torch.cat([y_grid, x_grid], dim=-1)             # [H, W, d_model]
        pe = pe.permute(2, 0, 1).unsqueeze(0).expand(B, -1, -1, -1)
        return pe


# ---------------------------------------------------------------------------
# Transformer encoder
# ---------------------------------------------------------------------------

class TransformerEncoderLayer(nn.Module):
    """Self-attn with PE on Q/K only, V is features. Residual + LN around each sublayer."""

    def __init__(self, d_model: int = 256, n_heads: int = 8,
                 dim_ff: int = 2048, dropout: float = 0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True,
        )
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_ff, d_model),
            nn.Dropout(dropout),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x: Tensor, pos: Tensor) -> Tensor:
        # x, pos: [B, L, d_model]
        q = k = x + pos
        sa_out, _ = self.self_attn(query=q, key=k, value=x)
        x = self.norm1(x + sa_out)
        ff_out = self.ffn(x)
        x = self.norm2(x + ff_out)
        return x


class TransformerEncoder(nn.Module):
    """Stack of N encoder layers."""

    def __init__(self, n_layers: int = 6, **layer_kwargs):
        super().__init__()
        self.layers = nn.ModuleList(
            [TransformerEncoderLayer(**layer_kwargs) for _ in range(n_layers)]
        )

    def forward(self, x: Tensor, pos: Tensor) -> Tensor:
        for layer in self.layers:
            x = layer(x, pos)
        return x


# ---------------------------------------------------------------------------
# Transformer decoder
# ---------------------------------------------------------------------------

class TransformerDecoderLayer(nn.Module):
    """Self-attn over queries + cross-attn to encoder memory + FFN, each with residual + LN."""

    def __init__(self, d_model: int = 256, n_heads: int = 8,
                 dim_ff: int = 2048, dropout: float = 0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True,
        )
        self.cross_attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True,
        )
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_ff, d_model),
            nn.Dropout(dropout),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

    def forward(self, tgt: Tensor, memory: Tensor,
                pos: Tensor, query_pos: Tensor) -> Tensor:
        # Self-attn over queries
        q = k = tgt + query_pos
        sa_out, _ = self.self_attn(query=q, key=k, value=tgt)
        tgt = self.norm1(tgt + sa_out)

        # Cross-attn: queries attend to encoder memory
        Q = tgt + query_pos
        K = memory + pos
        ca_out, _ = self.cross_attn(query=Q, key=K, value=memory)
        tgt = self.norm2(tgt + ca_out)

        # FFN
        ff_out = self.ffn(tgt)
        tgt = self.norm3(tgt + ff_out)
        return tgt


class TransformerDecoder(nn.Module):
    """Stack of N decoder layers. Returns output AFTER EACH LAYER (for auxiliary losses)."""

    def __init__(self, n_layers: int = 6, **layer_kwargs):
        super().__init__()
        self.layers = nn.ModuleList(
            [TransformerDecoderLayer(**layer_kwargs) for _ in range(n_layers)]
        )

    def forward(self, tgt: Tensor, memory: Tensor,
                pos: Tensor, query_pos: Tensor) -> List[Tensor]:
        outs = []
        for layer in self.layers:
            tgt = layer(tgt, memory, pos, query_pos)
            outs.append(tgt)
        return outs


# ---------------------------------------------------------------------------
# Prediction heads
# ---------------------------------------------------------------------------

class DetrPredictionHeads(nn.Module):
    """Class head (Linear -> K+1 logits) and bbox head (3-layer MLP + sigmoid)."""

    def __init__(self, d_model: int = 256, num_classes: int = 3):
        super().__init__()
        self.class_head = nn.Linear(d_model, num_classes + 1)
        self.bbox_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, 4),
        )

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        # x: [B, N, d_model]
        # returns: (logits [B, N, K+1], boxes [B, N, 4] in cxcywh normalized to [0,1])
        class_out = self.class_head(x)
        bbox_out = self.bbox_head(x).sigmoid()
        return class_out, bbox_out


# ---------------------------------------------------------------------------
# Full DETR model
# ---------------------------------------------------------------------------

class DETR(nn.Module):
    """Backbone -> 1x1 conv -> + 2D PE -> Encoder -> Decoder w/ N learned object queries -> heads."""

    def __init__(self,
                 num_classes: int = 3,
                 num_queries: int = 100,
                 d_model: int = 256,
                 n_heads: int = 8,
                 n_enc_layers: int = 6,
                 n_dec_layers: int = 6,
                 dim_ff: int = 2048,
                 backbone_name: str = 'resnet18'):
        super().__init__()
        self.num_queries = num_queries
        self.d_model = d_model

        self.backbone = Backbone(d_model=d_model, name=backbone_name)
        self.pos_encoding = PositionalEncoding2D(d_model=d_model)
        self.encoder = TransformerEncoder(
            n_layers=n_enc_layers, d_model=d_model, n_heads=n_heads, dim_ff=dim_ff,
        )
        self.decoder = TransformerDecoder(
            n_layers=n_dec_layers, d_model=d_model, n_heads=n_heads, dim_ff=dim_ff,
        )
        self.query_pos_emb = nn.Embedding(num_queries, d_model)
        self.heads = DetrPredictionHeads(d_model=d_model, num_classes=num_classes)

    def forward(self, images: Tensor) -> Dict[str, Tensor]:
        # Backbone + projection
        feat = self.backbone(images)               # [B, d_model, H', W']
        pe = self.pos_encoding(feat)               # [B, d_model, H', W']
        B = feat.shape[0]

        # Flatten spatial -> sequence
        feat_flat = feat.flatten(2).transpose(1, 2)   # [B, L, d_model]
        pos_flat  = pe.flatten(2).transpose(1, 2)     # [B, L, d_model]

        # Encoder
        memory = self.encoder(feat_flat, pos_flat)    # [B, L, d_model]

        # Decoder inputs
        tgt = memory.new_zeros(B, self.num_queries, self.d_model)
        query_pos = self.query_pos_emb.weight.unsqueeze(0).expand(B, -1, -1)

        # Decoder — returns per-layer outputs for aux losses
        dec_outs = self.decoder(tgt, memory, pos_flat, query_pos)

        # Apply prediction heads to every decoder-layer output
        head_outs = [self.heads(x) for x in dec_outs]
        all_logits = torch.stack([h[0] for h in head_outs])   # [n_layers, B, N, K+1]
        all_boxes  = torch.stack([h[1] for h in head_outs])   # [n_layers, B, N, 4]

        return {
            'pred_logits': all_logits[-1],
            'pred_boxes':  all_boxes[-1],
            'aux_outputs': [
                {'pred_logits': all_logits[i], 'pred_boxes': all_boxes[i]}
                for i in range(len(dec_outs) - 1)
            ],
        }


# ---------------------------------------------------------------------------
# Hungarian matcher
# ---------------------------------------------------------------------------

class HungarianMatcher(nn.Module):
    """Bipartite matching between predictions and ground truths via scipy."""

    def __init__(self, cost_class: float = 1.0,
                 cost_bbox: float = 5.0, cost_giou: float = 2.0):
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou

    @torch.no_grad()
    def forward(self, outputs: Dict[str, Tensor],
                targets: List[Dict[str, Tensor]]) -> List[Tuple[Tensor, Tensor]]:
        B = outputs['pred_logits'].shape[0]
        assert len(targets) == B, f"Batch mismatch: outputs B={B} but len(targets)={len(targets)}"

        matched = []
        for b in range(B):
            tgt_labels = targets[b]['labels']
            tgt_boxes  = targets[b]['boxes']

            # Classification cost: -softmax-prob of each gt class for each prediction
            probs = F.softmax(outputs['pred_logits'][b], dim=-1)    # [N, K+1]
            cost_class = -probs[:, tgt_labels]                       # [N, M]

            # L1 box cost on cxcywh
            cost_bbox = torch.cdist(outputs['pred_boxes'][b], tgt_boxes, p=1)   # [N, M]

            # GIoU cost (negated since high GIoU = good match)
            pred_xyxy = box_convert(outputs['pred_boxes'][b], 'cxcywh', 'xyxy')
            tgt_xyxy  = box_convert(tgt_boxes, 'cxcywh', 'xyxy')
            cost_giou = -generalized_box_iou(pred_xyxy, tgt_xyxy)    # [N, M]

            # Weighted sum and Hungarian assignment on CPU
            C = (self.cost_class * cost_class
                 + self.cost_bbox  * cost_bbox
                 + self.cost_giou  * cost_giou)
            pred_idx, tgt_idx = linear_sum_assignment(C.cpu())

            matched.append((
                torch.as_tensor(pred_idx, dtype=torch.long),
                torch.as_tensor(tgt_idx,  dtype=torch.long),
            ))

        return matched


# ---------------------------------------------------------------------------
# Loss (DetrLoss / SetCriterion)
# ---------------------------------------------------------------------------

class DetrLoss(nn.Module):
    """Class CE + L1 + GIoU on Hungarian-matched pairs. Auxiliary losses on every decoder layer."""

    def __init__(self,
                 num_classes: int = 3,
                 matcher: HungarianMatcher = None,
                 weight_class: float = 1.0,
                 weight_bbox: float = 5.0,
                 weight_giou: float = 2.0,
                 noobj_weight: float = 0.1):
        super().__init__()
        self.matcher = matcher
        self.num_classes = num_classes
        self.weight_class = weight_class
        self.weight_bbox  = weight_bbox
        self.weight_giou  = weight_giou

        # Class weights: down-weight 'no object' (class 0)
        class_w = torch.ones(num_classes + 1)
        class_w[0] = noobj_weight
        self.loss_class = nn.CrossEntropyLoss(weight=class_w)

    def _compute_losses(self, outputs: Dict[str, Tensor],
                        targets: List[Dict[str, Tensor]]) -> Dict[str, Tensor]:
        indices = self.matcher(outputs, targets)
        B, N = outputs['pred_logits'].shape[:2]
        device = outputs['pred_logits'].device

        # Classification: build [B, N] target class tensor (zeros = no object, fill matched)
        target_classes = torch.zeros(B, N, dtype=torch.long, device=device)
        for b, (pred_idx, tgt_idx) in enumerate(indices):
            target_classes[b, pred_idx] = targets[b]['labels'][tgt_idx]
        loss_class = self.loss_class(
            outputs['pred_logits'].transpose(1, 2),    # [B, K+1, N]
            target_classes,                             # [B, N]
        )

        # Box losses: only on matched pairs
        src_b, tgt_b = [], []
        for b, (p, t) in enumerate(indices):
            src_b.append(outputs['pred_boxes'][b, p])
            tgt_b.append(targets[b]['boxes'][t])
        src_boxes = torch.cat(src_b, dim=0)
        tgt_boxes = torch.cat(tgt_b, dim=0)
        total_M = max(src_boxes.shape[0], 1)

        loss_bbox = F.l1_loss(src_boxes, tgt_boxes, reduction='none').sum() / total_M

        src_xyxy = box_convert(src_boxes, 'cxcywh', 'xyxy')
        tgt_xyxy = box_convert(tgt_boxes, 'cxcywh', 'xyxy')
        loss_giou = (1 - torch.diag(generalized_box_iou(src_xyxy, tgt_xyxy))).sum() / total_M

        total = (self.weight_class * loss_class
                 + self.weight_bbox  * loss_bbox
                 + self.weight_giou  * loss_giou)
        return {
            'class': loss_class,
            'bbox':  loss_bbox,
            'giou':  loss_giou,
            'total': total,
        }

    def forward(self, outputs: Dict[str, Tensor],
                targets: List[Dict[str, Tensor]]) -> Dict[str, Tensor]:
        losses = self._compute_losses(outputs, targets)
        for aux in outputs.get('aux_outputs', []):
            aux_losses = self._compute_losses(aux, targets)
            for k in losses:
                losses[k] = losses[k] + aux_losses[k]
        return losses


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train_detr(num_steps: int = 500,
               batch_size: int = 8,
               lr: float = 1e-4,
               lr_backbone: float = 1e-5,
               weight_decay: float = 1e-4,
               clip_grad: float = 0.1,
               num_samples: int = 500,
               image_size: int = 128,
               num_classes: int = 3,
               num_queries: int = 100,
               noobj_weight: float = 0.1,
               backbone_name: str = 'resnet18',
               log_every: int = 20,
               device=None):
    """Standalone DETR training loop on the synthetic toy dataset."""
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    dataset = ToyDetectionDataset(
        num_samples=num_samples, image_size=image_size,
        min_objects=1, max_objects=5, num_classes=num_classes,
    )
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True, collate_fn=detr_collate,
    )

    model = DETR(
        num_classes=num_classes,
        num_queries=num_queries,
        backbone_name=backbone_name,
    ).to(device)
    matcher   = HungarianMatcher()
    criterion = DetrLoss(num_classes=num_classes, matcher=matcher,
                         noobj_weight=noobj_weight).to(device)

    backbone_params = [p for n, p in model.named_parameters()
                       if 'backbone' in n and p.requires_grad]
    other_params    = [p for n, p in model.named_parameters()
                       if 'backbone' not in n and p.requires_grad]
    optimizer = torch.optim.AdamW(
        [
            {'params': backbone_params, 'lr': lr_backbone},
            {'params': other_params,    'lr': lr},
        ],
        weight_decay=weight_decay,
    )

    model.train()
    step = 0
    while step < num_steps:
        for images, targets in loader:
            images = images.to(device)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            outputs = model(images)
            losses = criterion(outputs, targets)

            optimizer.zero_grad()
            losses['total'].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
            optimizer.step()

            if step % log_every == 0:
                print(
                    f"step {step:5d}  "
                    f"class={losses['class'].item():.3f}  "
                    f"bbox={losses['bbox'].item():.3f}  "
                    f"giou={losses['giou'].item():.3f}  "
                    f"total={losses['total'].item():.3f}"
                )

            step += 1
            if step >= num_steps:
                break

    return model


# ---------------------------------------------------------------------------
# Inference + visualization
# ---------------------------------------------------------------------------

@torch.no_grad()
def predict_and_visualize(model: DETR,
                          image: Tensor,
                          score_threshold: float = 0.7,
                          topk: int = None,
                          show_attention: bool = True,
                          class_names: List[str] = None):
    """
    Inference on one image. Plots boxes (and optional per-query attention overlays).

    Args:
        model:           trained DETR
        image:           [3, H, W] float tensor in [0, 1]
        score_threshold: keep predictions with max real-class probability >= this
        topk:            if set, ignore threshold and keep top-K by score
        show_attention:  if True, plot per-query attention maps from last decoder layer
        class_names:     optional list, e.g. ['no_object', 'red', 'green', 'blue']
    """
    model.eval()
    device = next(model.parameters()).device
    image_dev = image.to(device)

    # Hook the last decoder layer's cross-attention
    attn_store = {}

    def hook(mod, inp, out):
        attn_store['weights'] = out[1]   # [B, N, L]

    handle = model.decoder.layers[-1].cross_attn.register_forward_hook(hook)
    try:
        outputs = model(image_dev.unsqueeze(0))
    finally:
        handle.remove()

    logits = outputs['pred_logits'][0]      # [N, K+1]
    boxes  = outputs['pred_boxes'][0]       # [N, 4] cxcywh in [0, 1]
    probs = F.softmax(logits, dim=-1)        # [N, K+1]

    # Argmax over real classes only (skip class 0 = no object)
    real_probs = probs[:, 1:]                # [N, K]
    scores, classes_m1 = real_probs.max(dim=-1)
    classes = classes_m1 + 1                  # shift back to [1, K]

    if topk is not None:
        topk_scores, keep_idx = scores.topk(min(topk, scores.numel()))
        boxes_kept   = boxes[keep_idx].cpu()
        classes_kept = classes[keep_idx].cpu()
        scores_kept  = topk_scores.cpu()
    else:
        keep = scores >= score_threshold
        keep_idx = keep.nonzero(as_tuple=True)[0]
        boxes_kept   = boxes[keep_idx].cpu()
        classes_kept = classes[keep_idx].cpu()
        scores_kept  = scores[keep_idx].cpu()
    n_kept = len(keep_idx)

    H_img, W_img = image.shape[1:]
    img_np = image.permute(1, 2, 0).cpu().numpy().clip(0, 1)

    def label_for(c):
        if class_names is not None and 0 <= int(c) < len(class_names):
            return class_names[int(c)]
        return f'cls {int(c)}'

    def draw_boxes(ax):
        ax.imshow(img_np)
        for box, cls, score in zip(boxes_kept, classes_kept, scores_kept):
            cx, cy, bw, bh = box.tolist()
            x1, y1 = (cx - bw / 2) * W_img, (cy - bh / 2) * H_img
            ax.add_patch(patches.Rectangle(
                (x1, y1), bw * W_img, bh * H_img,
                linewidth=2, edgecolor='lime', facecolor='none',
            ))
            ax.text(x1, y1 - 4,
                    f'{label_for(cls)} ({score:.2f})',
                    color='white', fontsize=9,
                    bbox=dict(facecolor='black', alpha=0.6, pad=1))
        ax.axis('off')

    if not show_attention or n_kept == 0:
        fig, ax = plt.subplots(figsize=(6, 6))
        draw_boxes(ax)
        ax.set_title(f'predictions (kept {n_kept})')
        plt.tight_layout()
        plt.show()
    else:
        attn = attn_store['weights'][0]                    # [N, L]
        L = attn.shape[1]
        H_feat = W_feat = int(round(L ** 0.5))             # assumes square feature map

        n_cols = min(n_kept + 1, 6)
        n_rows = (n_kept + 1 + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))
        axes = axes.flatten() if (n_rows * n_cols) > 1 else [axes]

        draw_boxes(axes[0])
        axes[0].set_title(f'predictions (kept {n_kept})')

        for i, qi in enumerate(keep_idx):
            ax = axes[i + 1]
            attn_map = attn[qi].reshape(H_feat, W_feat).float()
            attn_up = F.interpolate(
                attn_map[None, None], size=(H_img, W_img),
                mode='bilinear', align_corners=False,
            )[0, 0].cpu()
            ax.imshow(img_np)
            ax.imshow(attn_up, alpha=0.55, cmap='jet')
            ax.set_title(f'q{qi.item()} → {label_for(classes_kept[i])} ({scores_kept[i]:.2f})')
            ax.axis('off')

        for j in range(n_kept + 1, len(axes)):
            axes[j].axis('off')

        plt.tight_layout()
        plt.show()

    return {
        'boxes':         boxes_kept,
        'classes':       classes_kept,
        'scores':        scores_kept,
        'query_indices': keep_idx.cpu(),
    }


# ---------------------------------------------------------------------------
# Quick smoke-test entry point
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    # Quick smoke test — instantiate everything and do a single forward + loss step.
    print("Quick smoke test...")
    ds = ToyDetectionDataset(num_samples=4, image_size=128, num_classes=3)
    images = torch.stack([ds[i][0] for i in range(2)], dim=0)
    targets = [ds[i][1] for i in range(2)]

    model = DETR(num_classes=3, num_queries=20, backbone_name='resnet18')
    matcher = HungarianMatcher()
    criterion = DetrLoss(num_classes=3, matcher=matcher)

    outputs = model(images)
    losses = criterion(outputs, targets)
    print(f"  pred_logits: {outputs['pred_logits'].shape}")
    print(f"  pred_boxes:  {outputs['pred_boxes'].shape}")
    print(f"  aux_outputs: {len(outputs['aux_outputs'])}")
    print(f"  losses: {{ {', '.join(f'{k}={v.item():.3f}' for k, v in losses.items())} }}")
    print("OK")
