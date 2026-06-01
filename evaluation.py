#!/usr/bin/env python3

import os
import random
import argparse
import csv
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import ListedColormap
from pathlib import Path
from PIL import Image
from tqdm import tqdm

from model import (
    initialize_model, process_image,
    needs_hd, is_small_template,
    TEMPLATE_COLORS,
)

# =============================================================================
# Dataset paths
# =============================================================================

FSC147_ROOT      = 'raid/datasets/FSC147_384_V2'
ANNOTATION_FILE  = os.path.join(FSC147_ROOT, 'annotation_FSC147_384.json')
DATA_SPLIT_FILE  = os.path.join(FSC147_ROOT, 'Train_Test_Val_FSC_147.json')

# =============================================================================
# Utils
# =============================================================================

def load_annotations(split):
    with open(ANNOTATION_FILE) as f:
        annotations = json.load(f)
    with open(DATA_SPLIT_FILE) as f:
        splits = json.load(f)
    img_dir = os.path.join(FSC147_ROOT, 'images_384_VarV2')
    all_images = splits[split]
    return annotations, all_images, img_dir

# =============================================================================
# Visualisation (Made with the help of AI agent CLAUDE)
# =============================================================================

TEMPLATE_COLORS = ['red', 'lime', 'blue', 'yellow', 'cyan', 'magenta']

def save_visualization(img, arts, template_bboxes, img_filename,
                        gt_count, pred_count, save_path,
                        used_hd=False, used_lm_fallback=False):
    N_t    = len(template_bboxes)
    cmap_t = ListedColormap(['#000000'] + TEMPLATE_COLORS[:N_t])
    KW     = dict(fontsize=9, pad=3)

    final_np       = arts['final_output']
    sm             = arts['sim_map']
    cm             = arts['conv_map']
    local_max      = arts['local_max']
    optimal_labels = arts['optimal_labels']
    BG_LABEL       = arts['BG_LABEL']
    indices_sim    = arts['indices_sim']
    indices_conv   = arts['indices_conv']
    rel_err        = abs(pred_count - gt_count) / max(gt_count, 1) * 100

    flags = []
    if used_hd:          flags.append('HD=1680px')
    if used_lm_fallback: flags.append(f"LM-fallback peaks={arts.get('n_peaks','')}")
    ratio = arts.get('density_vs_conv_ratio')
    if arts.get('density_runaway'):
        flags.append(f"RUNAWAY ratio={ratio:.3f}")
    elif ratio is not None:
        flags.append(f"ratio={ratio:.3f}")
    flag_str = '  [' + ' | '.join(flags) + ']' if flags else ''

    fig = plt.figure(figsize=(22, 9))
    gs  = fig.add_gridspec(2, 4, hspace=0.08, wspace=0.06,
                           left=0.02, right=0.98, top=0.90, bottom=0.02)
    ax_img     = fig.add_subplot(gs[:, 0])
    ax_sim     = fig.add_subplot(gs[0, 1])
    ax_simcmap = fig.add_subplot(gs[1, 1])
    ax_conv    = fig.add_subplot(gs[0, 2])
    ax_convcmap= fig.add_subplot(gs[1, 2])
    ax_final   = fig.add_subplot(gs[0, 3])
    ax_labels  = fig.add_subplot(gs[1, 3])

    ax_img.imshow(np.array(img))
    for i, (x1, y1, x2, y2) in enumerate(template_bboxes):
        c = TEMPLATE_COLORS[i % len(TEMPLATE_COLORS)]
        ax_img.add_patch(patches.Rectangle((x1, y1), x2-x1, y2-y1,
                         linewidth=2.5, edgecolor=c, facecolor='none'))
        ax_img.text(x1, y1-5, f'T{i}', color=c, fontsize=9, fontweight='bold',
                    bbox=dict(facecolor='black', alpha=0.45, pad=2, edgecolor='none'))
    ax_img.set_title('Input + Templates', **KW); ax_img.axis('off')

    im = ax_sim.imshow(sm, cmap='jet')
    ax_sim.set_title('Similarity map', **KW); ax_sim.axis('off')
    fig.colorbar(im, ax=ax_sim, fraction=0.046, pad=0.02).ax.tick_params(labelsize=7)

    ax_simcmap.imshow((indices_sim % N_t) + 1, cmap=cmap_t, vmin=0, vmax=N_t,
                      interpolation='nearest')
    ax_simcmap.set_title('Sim. dominant template', **KW); ax_simcmap.axis('off')

    im2 = ax_conv.imshow(cm, cmap='jet')
    if local_max:
        ys, xs = zip(*local_max)
        ax_conv.scatter(xs, ys, s=22, c='white', marker='+', linewidths=1.3)
    ax_conv.set_title(f'Conv map ({len(local_max)} maxima)', **KW); ax_conv.axis('off')
    fig.colorbar(im2, ax=ax_conv, fraction=0.046, pad=0.02).ax.tick_params(labelsize=7)

    ax_convcmap.imshow((indices_conv % N_t) + 1, cmap=cmap_t, vmin=0, vmax=N_t,
                       interpolation='nearest')
    if local_max:
        ax_convcmap.scatter(xs, ys, s=22, c='white', marker='+', linewidths=1.3)
    ax_convcmap.set_title('Conv dominant template', **KW); ax_convcmap.axis('off')

    im3 = ax_final.imshow(final_np, cmap='jet')
    ax_final.set_title(f'Final  pred={pred_count:.1f}  gt={gt_count:.0f}  err={rel_err:.1f}%',
                       fontsize=9, fontweight='bold', pad=3)
    ax_final.axis('off')
    fig.colorbar(im3, ax=ax_final, fraction=0.046, pad=0.02).ax.tick_params(labelsize=7)

    N        = BG_LABEL
    cmap_lbl = ListedColormap(['#000000'] + TEMPLATE_COLORS[:N])
    display  = np.where(optimal_labels == BG_LABEL, 0, optimal_labels + 1)
    ax_labels.imshow(display, cmap=cmap_lbl, vmin=0, vmax=N, interpolation='nearest')
    legend_lbl = [patches.Patch(facecolor='#000000', edgecolor='grey', label='BG')]
    for i in range(N):
        legend_lbl.append(patches.Patch(facecolor=TEMPLATE_COLORS[i % len(TEMPLATE_COLORS)],
                                        label=f'T{i}'))
    ax_labels.legend(handles=legend_lbl, fontsize=6, loc='upper right', framealpha=0.75)
    ax_labels.set_title(f'Label map  pred={pred_count:.1f}  gt={gt_count:.0f}',
                        fontsize=9, fontweight='bold', pad=3)
    ax_labels.axis('off')

    fig.suptitle(f'{img_filename}  GT={gt_count:.0f}  Pred={pred_count:.1f}'
                 f'  RelErr={rel_err:.1f}{flag_str}',
                 fontsize=11, fontweight='bold', y=0.97)
    plt.savefig(save_path, dpi=90, bbox_inches='tight')
    plt.close(fig)


def _rank_and_save(records, sort_key, ascending, top_n, out_dir, label, img_dir):
    sorted_recs = sorted(records, key=lambda r: r[sort_key],
                         reverse=not ascending)[:top_n]
    save_dir = out_dir / label
    save_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n  Saving {len(sorted_recs)} '{label}' visualisations")
    for rec in tqdm(sorted_recs, desc=label):
        try:
            img = Image.open(os.path.join(img_dir, rec['img_filename'])).convert('RGB')
        except Exception as e:
            print(f"    Could not reload {rec['img_filename']}: {e}")
            continue
        fname = (f"{rec['img_filename']}_err{rec['rel_err']:.1f}pct"
                 f"_pred{rec['pred_count']:.0f}_gt{rec['gt_count']}.png")
        save_visualization(img, rec['arts'], rec['converted_bboxes'],
                           rec['img_filename'], rec['gt_count'], rec['pred_count'],
                           save_path=str(save_dir / fname),
                           used_hd=rec['used_hd'],
                           used_lm_fallback=rec['used_lm_fallback'])


# =============================================================================
# Main (Baseline of the CountingDINO code + help of AI for generating the CSV file (CLAUDE))
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Evaluate counting model')
    parser.add_argument('--percentage', type=float, default=100)
    parser.add_argument('--split', type=str, default='test')
    parser.add_argument('--output_dir', type=str, default='eval_results')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--top_high_err_pct', type=int, default=100)
    parser.add_argument('--top_worst_n', type=int, default=10)
    parser.add_argument('--top_best_n', type=int, default=20)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    annotations, all_images, img_dir = load_annotations(args.split)

    n_eval   = max(1, int(len(all_images) * args.percentage / 100))
    selected = random.sample(all_images, n_eval)
    
    model_840,  transform_840,  _ = initialize_model(840)
    model_1680, transform_1680, _ = initialize_model(1680)

    maes, mses  = [], []
    all_records = []
    csv_path = out_dir / 'predictions.csv'
    with open(csv_path, 'w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(['image', 'gt_count', 'pred_count', 'abs_error',
                         'sq_error', 'rel_error_pct', 'n_templates',
                         'used_hd', 'used_lm_fallback',
                         'pred_density', 'n_peaks',
                         'density_vs_conv_ratio', 'density_runaway'])

        for img_filename in tqdm(selected, desc='Evaluating'):
            img_path = os.path.join(img_dir, img_filename)
            if not os.path.exists(img_path):
                print(f"  Not found: {img_path}")
                continue

            try:
                ann = annotations.get(img_filename)
            except FileNotFoundError:
                continue

            gt_count = len(ann['points'])
            converted_bboxes = []
            for box in ann['box_examples_coordinates']:
                xs = [p[0] for p in box]
                ys = [p[1] for p in box]
                if len(converted_bboxes) < 3:
                    converted_bboxes.append([min(xs), min(ys), max(xs), max(ys)])

            try:
                img = Image.open(img_path).convert('RGB')
                orig_w, orig_h = img.size

                used_hd = needs_hd(converted_bboxes, orig_w, orig_h, resize_dim=840)
                not_all_small = not all( is_small_template(bb, orig_w, orig_h, 840) for bb in converted_bboxes)
                used_lm_fallback = used_hd and not_all_small

                if used_hd:
                    model_u, transform_u, resize_u = model_1680, transform_1680, 1680
                else:
                    model_u, transform_u, resize_u = model_840,  transform_840,  840

                pred_count, arts = process_image(
                    model_u, img, transform_u, converted_bboxes, resize_u,
                    use_local_max_as_pred=used_lm_fallback, used_hd=used_hd)

            except Exception as e:
                print(f"  Error on {img_filename}: {e}")
                continue

            abs_err = abs(pred_count - gt_count)
            sq_err  = abs_err ** 2
            rel_err = abs_err / max(gt_count, 1) * 100
            maes.append(abs_err); mses.append(sq_err)

            writer.writerow([
                img_filename, gt_count, f'{pred_count:.2f}',
                f'{abs_err:.2f}', f'{sq_err:.2f}', f'{rel_err:.2f}',
                len(converted_bboxes), int(used_hd), int(used_lm_fallback),
                f"{arts.get('pred_count_density', pred_count):.2f}",
                arts.get('n_peaks', ''),
                f"{arts.get('density_vs_conv_ratio', -1):.4f}",
                int(arts.get('density_runaway', False)),
            ])

            all_records.append(dict(
                img_filename=img_filename, gt_count=gt_count,
                pred_count=pred_count, abs_err=abs_err, rel_err=rel_err,
                used_hd=used_hd, used_lm_fallback=used_lm_fallback,
                converted_bboxes=converted_bboxes, arts=arts,
            ))

    high_err = [r for r in all_records if r['rel_err'] > 100]
    if args.top_high_err_pct != 0:
        high_err = high_err[:args.top_high_err_pct]
    print(f"\n  Images rel_err > 100% : {len(high_err)}")
    _rank_and_save(high_err, 'rel_err', False, len(high_err),
                   out_dir, 'high_error_gt100pct', img_dir)
    _rank_and_save(all_records, 'abs_err', False, args.top_worst_n,
                   out_dir, f'worst_{args.top_worst_n}_abs_error', img_dir)
    _rank_and_save(all_records, 'rel_err', True,  args.top_best_n,
                   out_dir, f'best_{args.top_best_n}_rel_error', img_dir)
    runaway = [r for r in all_records if r['arts'].get('density_runaway', False)]
    print(f"  Density runaway : {len(runaway)}")
    _rank_and_save(runaway, 'rel_err', False, len(runaway),
                   out_dir, 'density_runaway', img_dir)

    if maes:
        mae  = np.mean(maes)
        rmse = np.sqrt(np.mean(mses))
        n_hd = sum(1 for r in all_records if r['used_hd'])
        print(f"\n{'='*50}")
        print(f"  MAE  : {mae:.2f}")
        print(f"  RMSE : {rmse:.2f}")
        print(f"  HD   : {n_hd}/{len(maes)}")
        print(f"  CSV  : {csv_path}")
        print(f"{'='*50}\n")

        with open(out_dir / 'summary.txt', 'w') as f:
            f.write(f"Dataset : {args.dataset}\nSplit : {args.split}\n"
                    f"N : {len(maes)}\nMAE : {mae:.4f}\nRMSE : {rmse:.4f}\n"
                    f"HD images : {n_hd}\n")
    else:
        print("  No images evaluated.")


if __name__ == '__main__':
    main()