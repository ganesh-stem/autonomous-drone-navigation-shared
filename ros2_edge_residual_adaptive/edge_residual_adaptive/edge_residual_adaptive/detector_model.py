"""
A small YOLO-style object detector for the aerial images (5 channels in, 12 x 12 grid out).

    model = Detector()                       # about 0.2 M parameters, built for a Jetson / Raspberry Pi class device
    out = model(x)                           # x: (B, 5, 96, 96)  ->  (B, 8, 12, 12)
    dets = decode(out, thresh)               # per image: [(class, score, x0, y0, x1, y1)]
    f1 = score_detections(dets, labels, iou=0.3)

Each grid cell predicts: objectness, x/y offset inside the cell, log width and height, and 3 class scores.
"""

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .aerial import SIZE

GRID = 12
STRIDE = SIZE // GRID
N_CLASSES = 3
IN_CHANNELS = 5
MODALITIES = {'fused': [0, 1, 2, 3, 4], 'rgb': [0, 1, 2], 'thermal': [3], 'lidar': [4], 'thermal+lidar': [3, 4]}


def block(cin, cout, stride):
    return nn.Sequential(nn.Conv2d(cin, cout, 3, stride, 1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True))


class Detector(nn.Module):
    def __init__(self):
        super().__init__()
        self.body = nn.Sequential(
            block(IN_CHANNELS, 24, 1), block(24, 32, 2),      # 96 -> 48
            block(32, 48, 2),                                  # 24
            block(48, 64, 2),                                  # 12
            block(64, 96, 1), block(96, 96, 1),
        )
        self.head = nn.Conv2d(96, 5 + N_CLASSES, 1)

    def forward(self, x):
        return self.head(self.body(x))


def encode(labels_batch):
    """Targets for a batch: obj (B,G,G), box (B,4,G,G) = tx, ty, log w, log h, cls (B,G,G) (long)."""
    B = len(labels_batch)
    obj = np.zeros((B, GRID, GRID), dtype=np.float32)
    box = np.zeros((B, 4, GRID, GRID), dtype=np.float32)
    cls = np.zeros((B, GRID, GRID), dtype=np.int64)
    for b, labels in enumerate(labels_batch):
        for c, x0, y0, x1, y1 in labels:
            cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
            gx, gy = min(GRID - 1, int(cx // STRIDE)), min(GRID - 1, int(cy // STRIDE))
            obj[b, gy, gx] = 1.0
            box[b, 0, gy, gx] = cx / STRIDE - gx
            box[b, 1, gy, gx] = cy / STRIDE - gy
            box[b, 2, gy, gx] = math.log(max(x1 - x0, 1.0) / STRIDE)
            box[b, 3, gy, gx] = math.log(max(y1 - y0, 1.0) / STRIDE)
            cls[b, gy, gx] = c
    return torch.from_numpy(obj), torch.from_numpy(box), torch.from_numpy(cls)


def loss_fn(out, obj, box, cls):
    o = out[:, 0]
    pos = obj > 0.5
    l_obj = F.binary_cross_entropy_with_logits(o, obj, pos_weight=torch.tensor(4.0))
    if pos.any():
        xy = torch.sigmoid(out[:, 1:3]).permute(0, 2, 3, 1)[pos]
        wh = out[:, 3:5].permute(0, 2, 3, 1)[pos]
        tgt = box.permute(0, 2, 3, 1)[pos]
        l_box = F.mse_loss(xy, tgt[:, :2]) * 4.0 + F.smooth_l1_loss(wh, tgt[:, 2:])
        l_cls = F.cross_entropy(out[:, 5:].permute(0, 2, 3, 1)[pos], cls[pos])
    else:
        l_box = l_cls = out.sum() * 0.0
    return l_obj * 4.0 + l_box + l_cls


def iou(a, b):
    ix0, iy0, ix1, iy1 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def decode(out, thresh=0.4, nms_iou=0.4):
    """out: (B, 8, G, G) tensor or array -> per image a list of (class, score, x0, y0, x1, y1)."""
    out = torch.as_tensor(out)
    obj = torch.sigmoid(out[:, 0])
    xy = torch.sigmoid(out[:, 1:3])
    wh = torch.exp(out[:, 3:5].clamp(-4, 3))
    probs = torch.softmax(out[:, 5:], dim=1)
    results = []
    for b in range(out.shape[0]):
        dets = []
        ys, xs = torch.nonzero(obj[b] * probs[b].max(dim=0).values > thresh, as_tuple=True)
        for gy, gx in zip(ys.tolist(), xs.tolist()):
            c = int(probs[b, :, gy, gx].argmax())
            score = float(obj[b, gy, gx] * probs[b, c, gy, gx])
            cx, cy = (gx + float(xy[b, 0, gy, gx])) * STRIDE, (gy + float(xy[b, 1, gy, gx])) * STRIDE
            w, h = float(wh[b, 0, gy, gx]) * STRIDE, float(wh[b, 1, gy, gx]) * STRIDE
            dets.append((c, score, cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2))
        dets.sort(key=lambda d: -d[1])
        keep = []
        for d in dets:
            if all(k[0] != d[0] or iou(k[2:], d[2:]) < nms_iou for k in keep):
                keep.append(d)
        results.append(keep)
    return results


def score_detections(dets_batch, labels_batch, iou_thresh=0.3):
    """Per class true positives, false positives and false negatives (a detection matches an unused label of its class at IoU >= iou_thresh)."""
    tp = np.zeros(N_CLASSES)
    fp = np.zeros(N_CLASSES)
    fn = np.zeros(N_CLASSES)
    for dets, labels in zip(dets_batch, labels_batch):
        used = set()
        for c, s, x0, y0, x1, y1 in dets:
            best, best_i = 0.0, -1
            for i, (lc, lx0, ly0, lx1, ly1) in enumerate(labels):
                if lc == c and i not in used:
                    v = iou((x0, y0, x1, y1), (lx0, ly0, lx1, ly1))
                    if v > best:
                        best, best_i = v, i
            if best >= iou_thresh:
                tp[c] += 1
                used.add(best_i)
            else:
                fp[c] += 1
        for i, l in enumerate(labels):
            if i not in used:
                fn[l[0]] += 1
    return tp, fp, fn


def f1_from(tp, fp, fn):
    p = tp / np.maximum(tp + fp, 1)
    r = tp / np.maximum(tp + fn, 1)
    return p, r, 2 * p * r / np.maximum(p + r, 1e-9)


def param_count(model):
    return sum(p.numel() for p in model.parameters())
