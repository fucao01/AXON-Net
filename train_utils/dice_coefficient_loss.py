import torch
import torch.nn as nn
import torch.nn.functional as F



def build_target(target: torch.Tensor, num_classes: int = 2, ignore_index: int = -100):
    """build target for dice coefficient"""
    dice_target = target.clone()
    if ignore_index >= 0:
        ignore_mask = torch.eq(target, ignore_index)
        dice_target[ignore_mask] = 0
        # [N, H, W] -> [N, H, W, C]
        dice_target = nn.functional.one_hot(dice_target, num_classes).float()
        dice_target[ignore_mask] = ignore_index
    else:
        dice_target = nn.functional.one_hot(dice_target, num_classes).float()

    return dice_target.permute(0, 3, 1, 2)


def dice_coeff(x: torch.Tensor, target: torch.Tensor, ignore_index: int = -100, epsilon=1e-6):
    # Average of Dice coefficient for all batches, or for a single mask
    d = 0.
    batch_size = x.shape[0]
    for i in range(batch_size):
        x_i = x[i].reshape(-1)
        t_i = target[i].reshape(-1)
        if ignore_index >= 0:
            # 找出mask中不为ignore_index的区域
            roi_mask = torch.ne(t_i, ignore_index)
            x_i = x_i[roi_mask]
            t_i = t_i[roi_mask]
        inter = torch.dot(x_i, t_i)
        sets_sum = torch.sum(x_i) + torch.sum(t_i)
        if sets_sum == 0:
            sets_sum = 2 * inter

        d += (2 * inter + epsilon) / (sets_sum + epsilon)

    return d / batch_size


def multiclass_dice_coeff(x: torch.Tensor, target: torch.Tensor, ignore_index: int = -100, epsilon=1e-6):
    """Average of Dice coefficient for all classes"""
    dice = 0.
    for channel in range(x.shape[1]):
        dice += dice_coeff(x[:, channel, ...], target[:, channel, ...], ignore_index, epsilon)

    return dice / x.shape[1]


def dice_loss(x: torch.Tensor, target: torch.Tensor, multiclass: bool = False, ignore_index: int = -100):
    # Dice loss (objective to minimize) between 0 and 1
    x = nn.functional.softmax(x, dim=1)
    fn = multiclass_dice_coeff if multiclass else dice_coeff
    return 1 - fn(x, target, ignore_index=ignore_index)

def _valid_mask(target: torch.Tensor, ignore_index: int):
    if ignore_index < 0:
        return torch.ones_like(target, dtype=torch.bool)
    return torch.ne(target, ignore_index)

def tversky_loss(x: torch.Tensor,
                 target: torch.Tensor,
                 num_classes: int,
                 alpha: float = 0.5,
                 beta: float = 0.5,
                 ignore_index: int = -100,
                 eps: float = 1e-6) -> torch.Tensor:
    """
    Multi-class Tversky loss.
    alpha: weight for FN, beta: weight for FP. Larger beta reduces FP (improves precision), larger alpha reduces FN (improves recall).
    """
    probs = F.softmax(x, dim=1)
    one_hot = build_target(target, num_classes=num_classes, ignore_index=ignore_index)
    mask = _valid_mask(target, ignore_index).unsqueeze(1)  # (N,1,H,W)

    probs = probs * mask
    one_hot = one_hot * mask

    tp = (probs * one_hot).sum(dim=(0, 2, 3))
    fp = (probs * (1 - one_hot)).sum(dim=(0, 2, 3))
    fn = ((1 - probs) * one_hot).sum(dim=(0, 2, 3))

    tversky = (tp + eps) / (tp + alpha * fn + beta * fp + eps)
    loss = 1.0 - tversky.mean()
    return loss

def focal_tversky_loss(x: torch.Tensor,
                       target: torch.Tensor,
                       num_classes: int,
                       alpha: float = 0.5,
                       beta: float = 0.5,
                       gamma: float = 1.33,
                       ignore_index: int = -100,
                       eps: float = 1e-6) -> torch.Tensor:
    tv = tversky_loss(x, target, num_classes=num_classes, alpha=alpha, beta=beta, ignore_index=ignore_index, eps=eps)
    # tv is already averaged (scalar). Focal Tversky scales hard examples with gamma.
    return torch.pow(tv, gamma)

def _foreground_prob(x: torch.Tensor) -> torch.Tensor:
    """Return foreground probability map as (N,1,H,W)."""
    probs = F.softmax(x, dim=1)
    # if multi-class (C>=2), take channel 1 as foreground; otherwise return as-is
    if probs.shape[1] > 1:
        return probs[:, 1:2, ...]
    return probs


def laplace_loss(x: torch.Tensor):
    # Apply Laplacian on foreground probability map
    x = _foreground_prob(x)
    laplace_filter = torch.tensor([
        [0, 1, 0],
        [1, -4, 1],
        [0, 1, 0]
    ], dtype=torch.float32, device=x.device).unsqueeze(0).unsqueeze(0)
    filtered = F.conv2d(x, weight=laplace_filter, padding=1)
    return torch.mean(torch.abs(filtered))

def lap_loss(x, target):
    """Laplacian loss between predicted foreground prob and GT mask."""
    pred = _foreground_prob(x)
    # build GT foreground mask (N,1,H,W)
    if pred.shape[1] == 1:
        gt = (target == 1).float().unsqueeze(1)
    else:
        gt = nn.functional.one_hot(target.clamp_min(0).long(), x.shape[1]).float().permute(0, 3, 1, 2)[:, 1:2, ...]
    laplace_filter = torch.tensor([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]], dtype=torch.float32, device=pred.device).unsqueeze(0).unsqueeze(0)
    pred_d2 = F.conv2d(pred, laplace_filter, padding=1)
    gt_d2 = F.conv2d(gt, laplace_filter, padding=1)
    return torch.mean(torch.abs(pred_d2 - gt_d2))


def sobel_loss(x, target):
    """Sobel edge loss between predicted foreground prob and GT mask."""
    pred = _foreground_prob(x)
    if pred.shape[1] == 1:
        gt = (target == 1).float().unsqueeze(1)
    else:
        gt = nn.functional.one_hot(target.clamp_min(0).long(), x.shape[1]).float().permute(0, 3, 1, 2)[:, 1:2, ...]
    device = pred.device
    sobel_x = torch.tensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)
    sobel_y = torch.tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)
    pred_sobel_x = F.conv2d(pred, sobel_x, padding=1)
    pred_sobel_y = F.conv2d(pred, sobel_y, padding=1)
    gt_sobel_x = F.conv2d(gt, sobel_x, padding=1)
    gt_sobel_y = F.conv2d(gt, sobel_y, padding=1)
    return (torch.abs(pred_sobel_x - gt_sobel_x) + torch.abs(pred_sobel_y - gt_sobel_y)).mean()




def structure_loss(x: torch.Tensor,
                   target: torch.Tensor,
                   ignore_index: int = -100,
                   kernel_size: int = 31,
                   factor: float = 5.0) -> torch.Tensor:
    """
    Structure loss adapted from sam2unet-main:
      - Weighted BCE with logits on foreground channel
      - Weighted IoU on sigmoid foreground prob

    x: logits with shape (N,C,H,W). If C>1 uses channel 1 as foreground.
    target: integer mask (N,H,W) with foreground==1, background==0, and ignore_index as specified.
    """
    assert kernel_size % 2 == 1, "kernel_size should be odd"
    if x.shape[1] > 1:
        logit = x[:, 1:2, ...]
    else:
        logit = x

    mask = (target == 1).float().unsqueeze(1)
    device = x.device
    valid = torch.ones_like(mask, dtype=torch.bool)
    if ignore_index >= 0:
        valid = (target != ignore_index).unsqueeze(1)

    # edge-aware weights
    pool = F.avg_pool2d(mask, kernel_size=kernel_size, stride=1, padding=kernel_size // 2)
    weit = 1.0 + factor * torch.abs(pool - mask)
    # zero-out ignored regions in both weights and terms
    weit = weit * valid.float()

    # weighted BCE with logits
    bce_map = F.binary_cross_entropy_with_logits(logit, mask, reduction='none')  # (N,1,H,W)
    bce_w = (weit * bce_map).sum(dim=(2, 3))
    denom = (weit).sum(dim=(2, 3)).clamp_min(1e-6)
    wbce = bce_w / denom

    # weighted IoU term
    prob = torch.sigmoid(logit)
    inter = ((prob * mask) * weit).sum(dim=(2, 3))
    union = ((prob + mask) * weit).sum(dim=(2, 3))
    wiou = 1.0 - (inter + 1.0) / (union - inter + 1.0)

    return (wbce + wiou).mean()


def _get_logits(inputs):
    if isinstance(inputs, dict):
        if "out" not in inputs:
            raise KeyError("Expected key 'out' in model output dict.")
        return inputs["out"]
    return inputs


def axonnet_loss(inputs,
                 target: torch.Tensor,
                 num_classes: int = 2,
                 ignore_index: int = -100,
                 loss_type: str = "dice",
                 w_ce: float = 1.0,
                 w_seg: float = 1.0,
                 w_sobel: float = 1.0) -> torch.Tensor:
    """
    Composite loss aligned with AXONNet outputs.
    inputs: logits tensor (N,C,H,W) or {"out": logits}
    target: integer mask (N,H,W)
    """
    x = _get_logits(inputs)
    ce = F.cross_entropy(x, target, ignore_index=ignore_index)

    if loss_type == "dice":
        dice_target = build_target(target, num_classes, ignore_index)
        seg_term = dice_loss(x, dice_target, multiclass=True, ignore_index=ignore_index)
    elif loss_type == "tversky":
        seg_term = tversky_loss(x, target, num_classes=num_classes, ignore_index=ignore_index)
    elif loss_type == "focal_tversky":
        seg_term = focal_tversky_loss(x, target, num_classes=num_classes, ignore_index=ignore_index)
    elif loss_type == "structure":
        seg_term = structure_loss(x, target, ignore_index=ignore_index)
    else:
        raise ValueError(f"Unsupported loss_type: {loss_type}")

    loss = w_ce * ce + w_seg * seg_term
    if w_sobel != 0.0:
        loss = loss + w_sobel * sobel_loss(x, target)
    return loss
