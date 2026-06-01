#!/usr/bin/env python3

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.ops as ops
from torchvision.transforms import v2
from skimage.feature import peak_local_max
import maxflow
import maxflow.fastmin
from PIL import Image

from src.model import VisualBackbone
from src.utils import get_features, rescale_tensor

device = 'cuda' if torch.cuda.is_available() else 'cpu'


# =============================================================================
# Template helpers
# =============================================================================

def template_area(bbox):
    x1, y1, x2, y2 = bbox
    return max(0, abs(x2 - x1)) * max(0, abs(y2 - y1))


def small_template_area_thresh(orig_w, orig_h, resize_dim=840, patch_size=14,
                                min_token_side=3):
    scale_x = orig_w / resize_dim
    scale_y = orig_h / resize_dim
    return (min_token_side * patch_size * scale_x) * (min_token_side * patch_size * scale_y)


def needs_hd(template_bboxes, orig_w, orig_h, resize_dim=840):
    thresh = small_template_area_thresh(orig_w, orig_h, resize_dim)
    return any(template_area(bb) < thresh for bb in template_bboxes)


def is_small_template(bbox, orig_w, orig_h, resize_dim=840):
    thresh = small_template_area_thresh(orig_w, orig_h, resize_dim)
    return template_area(bbox) < thresh


# =============================================================================
# Model initialisation
# =============================================================================

def initialize_model(resize_dim=840):
    model = VisualBackbone('dinov2_vitb14_reg', img_size=resize_dim).to(device).eval()
    transform = v2.Compose([
        v2.ToImage(),
        v2.Resize((resize_dim, resize_dim), antialias=True),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    ])
    return model, transform, resize_dim


# =============================================================================
# Utils
# =============================================================================

def _resize_conv_maps(conv_maps, h):
    max_h = max_w = h
    resized = []
    for m in conv_maps:
        orig_h, orig_w = m.shape[-2:]
        ph, pw = max_h - orig_h, max_w - orig_w
        pt, pb = ph // 2, ph - ph // 2
        pl, pr = pw // 2, pw - pw // 2
        padded = F.pad(m.unsqueeze(1), (pl, pr, pt, pb), value=0).squeeze(1)
        resized.append(rescale_tensor(padded))
    return torch.cat(resized, dim=0)


def _post_process_density_map(conv_maps, h):
    output = _resize_conv_maps(conv_maps, h)
    result = output.max(dim=0)
    return result.values, result.indices


def _post_processing(stacked_maps, optimal_labels):
    N, H, W = stacked_maps.shape
    output  = np.zeros((H, W), dtype=np.float32)
    indices = np.full((H, W), -1, dtype=np.int32)
    for lbl in range(N):
        mask = (optimal_labels == lbl)
        output[mask]  = stacked_maps[lbl][mask]
        indices[mask] = lbl
    return output, indices


def _normalization_factors(output, bboxes, similar_sizes=False):
    output = np.asarray(output.detach().cpu() if torch.is_tensor(output) else output)
    template_sums = []
    for x1, y1, x2, y2 in bboxes:
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        region = output[y1:y2, x1:x2]
        template_sums.append(float(region.sum()))

    if similar_sizes:
        avg   = float(np.mean(template_sums))
        return [(1.0 / avg)  for coeff in template_sums]
    return [(1.0 / coeff) for coeff in template_sums]


def _normalize_output_by_template(output, indices, norm_coeff):
    if not torch.is_tensor(output):
        output  = torch.tensor(output)
    if not torch.is_tensor(indices): 
        indices = torch.tensor(indices)
    output = output.cpu()
    indices = indices.cpu()
    normalized = torch.zeros_like(output)
    for i, c in enumerate(norm_coeff):
        mask = (indices == i)
        normalized[mask] = output[mask] * c
    return normalized


def _optimize_patch_labels(stacked_maps_sim, output_sim, maxima_coords,
                            indices_conv, conv_map,
                            alpha_dist=None, smoothness=0.1, n_iter=5):
    N, H, W = stacked_maps_sim.shape
    BG_LABEL = N
    n_labels = N + 1
    if alpha_dist is None:
        alpha_dist = 1.0 / np.sqrt((H - 1) ** 2 + (W - 1) ** 2)
    rows_grid, cols_grid = np.mgrid[0:H, 0:W]

    maxima_by_label = {n: [] for n in range(N)}
    for r, c in maxima_coords:
        lbl = int(indices_conv[r, c])
        maxima_by_label[lbl].append((r, c))

    D = np.full((H, W, n_labels), np.inf, dtype=np.float64)
    for lbl in range(N):
        sim = np.clip(stacked_maps_sim[lbl], 1e-6, 1.0) ** 3
        pts = maxima_by_label[lbl]
        dist_min = (
            np.stack([np.sqrt((rows_grid - r) ** 2 + (cols_grid - c) ** 2)
                      for r, c in pts]).min(axis=0)
            if pts else np.full((H, W), np.inf)
        )
        D[:, :, lbl] = (1.0 - sim) + alpha_dist * dist_min

    bg_prob = np.clip(1.0 - output_sim, 1e-6, 1.0) ** 3
    all_dists = (
        np.stack([np.sqrt((rows_grid - r) ** 2 + (cols_grid - c) ** 2)
                  for r, c in maxima_coords]).min(axis=0)
        if maxima_coords else np.ones((H, W))
    )
    D[:, :, BG_LABEL] = (1.0 - bg_prob) - alpha_dist * all_dists + 0.25 * (1 - conv_map)

    V = smoothness * (1.0 - np.eye(n_labels, dtype=np.float64))
    labels_init = np.argmin(D, axis=2).astype(np.int32)
    labels = maxflow.fastmin.aexpansion_grid(D, V, max_cycles=n_iter, labels=labels_init)
    return labels, BG_LABEL


def _density_vs_convmap_ratio(final_output, conv_map, high_thresh=0.7):
    final_output = np.asarray(final_output)
    conv_map     = np.asarray(conv_map)
    total = final_output.sum()
    return float(final_output[conv_map > high_thresh].sum() / total)


# =============================================================================
# Pipeline
# =============================================================================

def process_image(model, img: Image.Image, transform, template_bboxes: list,
                  resize_dim: int = 840,
                  use_local_max_as_pred: bool = False,
                  used_hd: bool = False):
    
    with torch.no_grad():
        feats = get_features(model, img, transform, ['vit_out'])
        feats = feats / feats.norm(dim=1, keepdim=True)

    _, C, Hf, Wf = feats.shape
    orig_w, orig_h = img.size

    areas = [template_area(bb) for bb in template_bboxes]
    all_small = all(is_small_template(bb, orig_w, orig_h, resize_dim) for bb in template_bboxes)
    ratio_thresh = 4.0 if all_small else 2.0
    similar_sizes = max(areas) / min(areas) <= ratio_thresh

    bboxes_feat = [
        [int(x1 * Wf / orig_w), int(y1 * Hf / orig_h),
         int(x2 * Wf / orig_w), int(y2 * Hf / orig_h)]
        for x1, y1, x2, y2 in template_bboxes
    ]

    cosine_maps, conv_maps = [], []
    for x1, y1, x2, y2 in bboxes_feat:
        cx = int(torch.clamp(torch.tensor((x1 + x2) // 2), 0, Wf - 1))
        cy = int(torch.clamp(torch.tensor((y1 + y2) // 2), 0, Hf - 1))
        cf = feats[0, :, cy, cx]
        cf = cf / (cf.norm() + 1e-6)
        cosine_maps.append(torch.einsum('c,chw->hw', cf, feats[0]).abs())

        out_size = (max(1, int(y2 - y1)), max(1, int(x2 - x1)))
        bbox_t   = torch.tensor([x1, y1, x2, y2], dtype=torch.float32).to(device)
        pooled   = ops.roi_align(feats, [bbox_t.unsqueeze(0)],
                                 output_size=out_size, spatial_scale=1.0)
        conv_layer = nn.Conv2d(C, 1, kernel_size=out_size, bias=False).to(device)
        conv_layer.weight = nn.Parameter(pooled)
        with torch.no_grad():
            conv_maps.append(conv_layer(feats[0]).abs())

    stacked_maps_sim = torch.stack(cosine_maps, dim=0)
    sim_map          = stacked_maps_sim.max(dim=0).values
    indices_sim      = stacked_maps_sim.max(dim=0).indices

    conv_map, indices_conv = _post_process_density_map(conv_maps, Hf)
    conv_map_np = conv_map.detach().cpu().numpy()

    maxima_coords = peak_local_max(conv_map_np, threshold_abs=0.1, min_distance=1)
    local_max     = [(r, c) for r, c in maxima_coords if conv_map[r, c] > 0.7]

    optimal_labels, BG_LABEL = _optimize_patch_labels(
        stacked_maps_sim.detach().cpu().numpy(),
        sim_map.detach().cpu().numpy(),
        local_max,
        indices_conv.detach().cpu().numpy(),
        conv_map_np,
    )

    output, indices = _post_processing(
        stacked_maps_sim.detach().cpu().numpy(), optimal_labels)
    factors      = _normalization_factors(output, bboxes_feat, similar_sizes)
    final_output = _normalize_output_by_template(output, torch.tensor(indices), factors)
    pred_count_density = float(final_output.sum())

    ratio = _density_vs_convmap_ratio(final_output.numpy(), conv_map_np)
    density_runaway = ratio < 0.1

    if use_local_max_as_pred:
        pred_count = float(len(local_max))
    elif used_hd and density_runaway:
        pred_count = float(len(local_max))
    else:
        density_runaway = False
        pred_count = pred_count_density

    arts = dict(
        output=output,
        indices=indices,
        final_output=final_output.detach().cpu().numpy(),
        bboxes_feat=bboxes_feat,
        stacked_maps_sim=stacked_maps_sim.detach().cpu().numpy(),
        sim_map=sim_map.detach().cpu().numpy(),
        conv_map=conv_map_np,
        local_max=local_max,
        optimal_labels=optimal_labels,
        BG_LABEL=BG_LABEL,
        indices_sim=indices_sim.detach().cpu().numpy(),
        indices_conv=indices_conv.detach().cpu().numpy(),
        pred_count_density=pred_count_density,
        n_peaks=len(local_max),
        density_vs_conv_ratio=ratio,
        density_runaway=density_runaway,
    )
    return pred_count, arts