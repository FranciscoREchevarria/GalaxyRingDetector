"""
RingDetectionZoobot: Custom PyTorch Lightning module for multilabel ring classification.

This module provides a fine-tunable model for detecting inner and outer rings in galaxy images
using a pretrained Zoobot encoder.
"""

import torch
from torch import nn
from torch.optim.lr_scheduler import CosineAnnealingLR, SequentialLR, LinearLR
import lightning.pytorch as pl
from timm.loss import AsymmetricLossMultiLabel
from torchmetrics import Accuracy, F1Score, FBetaScore, Precision, Recall, HammingDistance
from tqdm import tqdm


def tune_thresholds_on_val(
    model,
    val_dataloader,
    device,
    threshold_range=(0.2, 0.9),
    step=0.02,
    metric="f1",
):
    """
    Optimize inner and outer ring thresholds jointly on validation set via 2D grid search.

    Searches over (t_inner, t_outer) pairs to maximize the chosen metric (macro-averaged
    over both labels), capturing interactions between the two thresholds for multi-label decisions.

    Args:
        model: RingDetectionZoobot (or any with predict_proba and inner/outer_ring_threshold).
        val_dataloader: DataLoader yielding batches with 'image' and 'ring_class'.
        device: torch device.
        threshold_range: (low, high) for threshold grid.
        step: Grid step size.
        metric: Optimization objective (macro over labels). 'f1', 'f2' (recall-weighted),
                'recall', 'precision', or 'hamming'.

    Returns:
        (best_t_inner, best_t_outer), and sets model.inner_ring_threshold, model.outer_ring_threshold.
    """
    model.eval()
    all_probs = []
    all_labels = []
    with torch.no_grad():
        for batch in tqdm(val_dataloader, desc="Tuning thresholds"):
            x = batch['image'].to(device)
            y = batch['ring_class'].to(device)
            probs = model.predict_proba_tta(x)
            all_probs.append(probs.cpu())
            all_labels.append(y.cpu())
    probs = torch.cat(all_probs, dim=0)
    labels = torch.cat(all_labels, dim=0)

    low, high = threshold_range
    steps = max(1, int((high - low) / step) + 1)
    thresholds = [
        low + k * (high - low) / max(1, steps - 1)
        for k in range(steps)
    ]

    # Macro average over both labels so we have one scalar per (t_inner, t_outer)
    if metric == "recall":
        score_metric = Recall(task='multilabel', num_labels=2, average='macro')
    elif metric == "f2":
        score_metric = FBetaScore(task='multilabel', num_labels=2, average='macro', beta=2.0)
    elif metric == "f1":
        score_metric = F1Score(task='multilabel', num_labels=2, average='macro')
    elif metric == "precision":
        score_metric = Precision(task='multilabel', num_labels=2, average='macro')
    elif metric == "hamming":
        score_metric = HammingDistance(task='multilabel', num_labels=2, average='macro')
    else:
        raise ValueError(f"Unsupported metric: {metric}")

    # Joint 2D grid search over (t_inner, t_outer)
    best_t_inner, best_t_outer = 0.5, 0.5
    best_score = -1.0
    minimize_hamming = metric == "hamming"

    for t_inner in thresholds:
        for t_outer in thresholds:
            preds = (probs > torch.tensor([t_inner, t_outer])).float()
            score_metric.reset()
            score = score_metric(preds, labels).item()
            if minimize_hamming:
                score = -score  # so we maximize in the same block
            if score > best_score:
                best_score = score
                best_t_inner = t_inner
                best_t_outer = t_outer

    model.inner_ring_threshold = best_t_inner
    model.outer_ring_threshold = best_t_outer
    return best_t_inner, best_t_outer


def _focal_bce_loss(logits, targets, pos_weight=None, gamma=2.0):
    """Focal loss for multilabel: (1 - pt)^gamma * BCE, with optional pos_weight per label."""
    p = torch.sigmoid(logits)
    pt = torch.where(targets == 1, p, 1 - p)
    focal_weight = (1 - pt).clamp(min=1e-6).pow(gamma)
    bce = nn.functional.binary_cross_entropy_with_logits(logits, targets, reduction='none')
    loss = focal_weight * bce
    if pos_weight is not None:
        pw = pos_weight.to(logits.device).expand_as(targets)
        loss = loss * torch.where(targets == 1, pw, 1.0)
    return loss.mean()


class RingDetectionZoobot(pl.LightningModule):
    def __init__(
        self,
        encoder,
        encoder_dim: int = 512,
        hidden_dim: int = 256,
        dropout_rate: float = 0.4,
        encoder_lr: float = 1e-5,
        head_lr: float = 1e-4,
        weight_decay: float = 0.0,
        pos_weight=None,
        use_focal_loss: bool = False,
        focal_gamma: float = 2.0,
        use_head_batchnorm: bool = False,
        use_asl: bool = False,
        asl_gamma_neg: float = 4.0,
        asl_gamma_pos: float = 0.0,
        asl_clip: float = 0.05,
        scheduler_type: str = "plateau",
        warmup_epochs: int = 3,
        eta_min: float = 1e-7,
        **kwargs,
    ):
        """
        Custom Zoobot model for multilabel ring classification.
        
        Args:
            encoder: Pretrained Zoobot encoder (already instantiated).
            encoder_dim: Dimension of encoder feature output.
            hidden_dim: Hidden layer dimension in the classification head.
            dropout_rate: Dropout probability in the head.
            encoder_lr: Learning rate for encoder parameters during fine-tuning.
            head_lr: Learning rate for the classification head.
            weight_decay: Weight decay for the optimizer (AdamW).
            pos_weight: Optional tensor/array of shape [2] for BCEWithLogitsLoss
                        to handle class imbalance between inner/outer rings.
            use_focal_loss: If True, use focal loss instead of BCE (for hard/rare positives).
            focal_gamma: Gamma for focal loss (default 2.0).
            use_head_batchnorm: If True, add BatchNorm1d after first linear in head.
            use_asl: If True, use Asymmetric Loss (Ridnik et al., 2021) instead of
                     BCE/focal. Mutually exclusive with use_focal_loss.
            asl_gamma_neg: ASL focusing parameter for negatives (higher = more suppression
                          of easy negatives). Typical range: 2-6.
            asl_gamma_pos: ASL focusing parameter for positives. Usually 0 (no down-weighting
                          of hard positives).
            asl_clip: Probability margin for hard-thresholding easy negatives. Set 0 to disable.
            scheduler_type: LR scheduler strategy. "plateau" for ReduceLROnPlateau (reactive),
                           "cosine" for CosineAnnealingLR with linear warmup (smooth decay).
            warmup_epochs: Number of warmup epochs for the cosine scheduler. Ignored when
                          scheduler_type="plateau".
            eta_min: Minimum learning rate for cosine annealing. Ignored when
                    scheduler_type="plateau".
        """
        super().__init__()

        if use_asl and use_focal_loss:
            raise ValueError("use_asl and use_focal_loss are mutually exclusive")

        self.inner_ring_threshold = 0.5
        self.outer_ring_threshold = 0.5
        self.use_focal_loss = use_focal_loss
        self.focal_gamma = focal_gamma
        self.use_asl = use_asl
        self._pos_weight_tensor = torch.as_tensor(pos_weight, dtype=torch.float32) if pos_weight is not None else None

        # Save lightweight hyperparameters for reproducibility / checkpoints
        self.save_hyperparameters(ignore=["encoder"])

        # Store encoder without calling parent init
        self.encoder = encoder

        # Define classification head (optionally with BatchNorm)
        head_layers = [
            nn.Dropout(p=dropout_rate),
            nn.Linear(encoder_dim, hidden_dim),
            nn.ReLU(),
        ]
        if use_head_batchnorm:
            head_layers.append(nn.BatchNorm1d(hidden_dim))
        head_layers.extend([
            nn.Dropout(p=dropout_rate / 2),
            nn.Linear(hidden_dim, 2),
        ])
        self.head = nn.Sequential(*head_layers)

        # Loss and metrics
        if use_asl:
            self.loss_fn = AsymmetricLossMultiLabel(
                gamma_neg=asl_gamma_neg, gamma_pos=asl_gamma_pos, clip=asl_clip,
            )
        elif use_focal_loss:
            self.loss_fn = None  # computed via _focal_bce_loss in steps
        elif pos_weight is not None:
            self.loss_fn = nn.BCEWithLogitsLoss(pos_weight=self._pos_weight_tensor)
        else:
            self.loss_fn = nn.BCEWithLogitsLoss()

        # Core metrics
        self.train_accuracy = Accuracy(task='multilabel', num_labels=2)
        self.val_accuracy = Accuracy(task='multilabel', num_labels=2)

        # More informative multilabel metrics (macro over labels)
        self.train_f1_macro = F1Score(task='multilabel', num_labels=2, average='macro')
        self.val_f1_macro = F1Score(task='multilabel', num_labels=2, average='macro')
        self.train_precision_macro = Precision(task='multilabel', num_labels=2, average='macro')
        self.val_precision_macro = Precision(task='multilabel', num_labels=2, average='macro')
        self.train_recall_macro = Recall(task='multilabel', num_labels=2, average='macro')
        self.val_recall_macro = Recall(task='multilabel', num_labels=2, average='macro')
        self.train_f2_macro = FBetaScore(task='multilabel', num_labels=2, average='macro', beta=2.0)
        self.val_f2_macro = FBetaScore(task='multilabel', num_labels=2, average='macro', beta=2.0)
        

        # Optimizer / scheduler hyperparameters
        self.encoder_lr = encoder_lr
        self.head_lr = head_lr
        self.weight_decay = weight_decay
        self.scheduler_type = scheduler_type
        self.warmup_epochs = warmup_epochs
        self.eta_min = eta_min

        # Track freeze state
        self.encoder_frozen = True
        self.freeze_encoder()
    
    def freeze_encoder(self):
        """Freeze encoder parameters."""
        for param in self.encoder.parameters():
            param.requires_grad = False
        self.encoder_frozen = True
        print("✓ Encoder frozen")
    
    def unfreeze_encoder(self):
        """Unfreeze encoder parameters."""
        for param in self.encoder.parameters():
            param.requires_grad = True
        self.encoder_frozen = False
        print("✓ Encoder unfrozen")
    
    def forward(self, x):
        features = self.encoder(x)
        logits = self.head(features)
        return logits
    
    def training_step(self, batch, batch_idx):
        x, y = batch['image'], batch['ring_class']
        logits = self.forward(x)
        if self.loss_fn is not None:
            loss = self.loss_fn(logits, y)
        else:
            loss = _focal_bce_loss(logits, y, self._pos_weight_tensor, self.focal_gamma)
        
        #prediction threshold is performed by class, allowing for different thresholds for inner vs outer ring detection if desired
        threshold = torch.tensor([self.inner_ring_threshold, self.outer_ring_threshold], device=logits.device)

        preds = (logits.sigmoid() > threshold).float()
        acc = self.train_accuracy(preds, y)
        f1 = self.train_f1_macro(preds, y)
        prec = self.train_precision_macro(preds, y)
        rec = self.train_recall_macro(preds, y)
        f2 = self.train_f2_macro(preds, y)  # or val_f2_macro

        self.log('finetuning/train_loss', loss, on_epoch=True)
        self.log('finetuning/train_acc', acc, on_epoch=True)
        self.log('finetuning/train_f1_macro', f1, on_epoch=True)
        self.log('finetuning/train_precision_macro', prec, on_epoch=True)
        self.log('finetuning/train_recall_macro', rec, on_epoch=True)
        self.log('finetuning/train_f2_macro', f2, on_epoch=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch['image'], batch['ring_class']
        logits = self.forward(x)
        if self.loss_fn is not None:
            loss = self.loss_fn(logits, y)
        else:
            loss = _focal_bce_loss(logits, y, self._pos_weight_tensor, self.focal_gamma)
        
        #prediction threshold is performed by class, allowing for different thresholds for inner vs outer ring detection if desired
        threshold = torch.tensor([self.inner_ring_threshold, self.outer_ring_threshold], device=logits.device)

        preds = (logits.sigmoid() > threshold).float()
        acc = self.val_accuracy(preds, y)
        f1 = self.val_f1_macro(preds, y)
        prec = self.val_precision_macro(preds, y)
        rec = self.val_recall_macro(preds, y)
        f2 = self.val_f2_macro(preds, y)

        self.log('finetuning/val_loss', loss, on_epoch=True)
        self.log('finetuning/val_acc', acc, on_epoch=True)
        self.log('finetuning/val_f1_macro', f1, on_epoch=True)
        self.log('finetuning/val_precision_macro', prec, on_epoch=True)
        self.log('finetuning/val_recall_macro', rec, on_epoch=True)
        self.log('finetuning/val_f2_macro', f2, on_epoch=True)
        return loss
    
    def configure_optimizers(self):
        """Configure optimizer with separate parameter groups for head/encoder."""
        head_params = [p for p in self.head.parameters() if p.requires_grad]
        encoder_params = [p for p in self.encoder.parameters() if p.requires_grad]

        if not head_params and not encoder_params:
            raise ValueError("No trainable parameters found!")

        param_groups = []
        if head_params:
            param_groups.append({"params": head_params, "lr": self.head_lr})
        if encoder_params:
            param_groups.append({"params": encoder_params, "lr": self.encoder_lr})

        optimizer = torch.optim.AdamW(
            param_groups,
            weight_decay=self.weight_decay,
        )

        if self.scheduler_type == "cosine":
            max_epochs = self.trainer.max_epochs
            warmup_epochs = min(self.warmup_epochs, max_epochs - 1)
            cosine_epochs = max(1, max_epochs - warmup_epochs)

            warmup = LinearLR(
                optimizer, start_factor=0.3, total_iters=warmup_epochs,
            )
            cosine = CosineAnnealingLR(
                optimizer, T_max=cosine_epochs, eta_min=self.eta_min,
            )
            scheduler = SequentialLR(
                optimizer, schedulers=[warmup, cosine], milestones=[warmup_epochs],
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "epoch",
                },
            }

        # Default: ReduceLROnPlateau (scheduler_type="plateau")
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=3,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "finetuning/val_loss",
            },
        }
    
    def predict(self, x, threshold=0.5):
        probs = self.forward(x).sigmoid()
        return (probs > threshold).float()

    def predict_proba(self, x):
        """
        Get probability predictions for binary multilabel classification.
        
        Args:
            x: Input image tensor
            
        Returns:
            Probability tensor of shape (batch_size, 2) with values in [0, 1]
        """
        logits = self.forward(x)
        probabilities = logits.sigmoid()
        return probabilities

    def predict_proba_tta(self, x, n_rotations=4, flip=True):
        """
        Test-time augmentation: average predictions over rotations and optional flips.
        Views: 0°, 90°, 180°, 270° and, if flip=True, their horizontal flips (8 views).
        """
        self.eval()
        probs_list = []
        with torch.no_grad():
            for k in range(n_rotations):
                x_rot = torch.rot90(x, k=k, dims=(-2, -1))
                probs_list.append(self.predict_proba(x_rot))
            if flip:
                for k in range(n_rotations):
                    x_rot = torch.rot90(x, k=k, dims=(-2, -1))
                    x_flip = torch.flip(x_rot, dims=[-1])  # horizontal
                    probs_list.append(self.predict_proba(x_flip))
        return torch.stack(probs_list, dim=0).mean(dim=0)

    def batch_to_supervised_tuple(self, batch):
        """Convert batch dictionary to (x, y) tuple for training."""
        # Your dataset returns {'image': tensor, 'ring_class': tensor}
        x = batch['image']
        y = batch['ring_class']
        return x, y
