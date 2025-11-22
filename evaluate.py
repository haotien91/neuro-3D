import os
import argparse
import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree
from tqdm import tqdm


def load_points(ply_path):
    pcd = o3d.io.read_point_cloud(ply_path)
    pts = np.asarray(pcd.points, dtype=np.float32)
    return pts


def chamfer_distance(points_pred, points_gt):
    """
    points_pred, points_gt: (N, 3), (M, 3)
    return: scalar CD
    """
    tree_gt = cKDTree(points_gt)
    dists_pred_to_gt, _ = tree_gt.query(points_pred, k=1)
    tree_pred = cKDTree(points_pred)
    dists_gt_to_pred, _ = tree_pred.query(points_gt, k=1)

    cd = np.mean(dists_pred_to_gt ** 2) + np.mean(dists_gt_to_pred ** 2)
    return cd


def f1_score(points_pred, points_gt, thresh):
    """
    One-sided NN distances + threshold → precision / recall / F1
    """
    tree_gt = cKDTree(points_gt)
    dists_pred_to_gt, _ = tree_gt.query(points_pred, k=1)
    tree_pred = cKDTree(points_pred)
    dists_gt_to_pred, _ = tree_pred.query(points_gt, k=1)

    precision = np.mean(dists_pred_to_gt < thresh)
    recall = np.mean(dists_gt_to_pred < thresh)

    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)

    return precision, recall, f1


def infer_gt_name_from_pred(pred_filename):
    """
    pred: 100000-46_oil_lamp_08-0.ply
    GT:   oil_lamp_08.ply  （丟掉前面的 index 46）
    """
    base = os.path.basename(pred_filename)
    if not base.endswith(".ply"):
        return None
    base = base[:-4]  # 去掉 .ply

    parts = base.split("-")
    if len(parts) < 3:
        # 格式不符  <step>-<name>-<idx>
        return None

    mid = parts[1]  # e.g. "46_oil_lamp_08"
    segs = mid.split("_")
    if len(segs) < 2:
        # 沒有 '_' 就當作整個名字用
        gt_core = mid
    else:
        # 丟掉最前面的 index，例如 "46_oil_lamp_08" -> ["46","oil","lamp","08"]
        # 變成 "oil_lamp_08"
        gt_core = "_".join(segs[1:])

    return gt_core + ".ply"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pred_dir",
        type=str,
        required=True,
        help="directory of predicted point clouds (.ply)"
    )
    parser.add_argument(
        "--gt_dir",
        type=str,
        required=True,
        help="directory of ground-truth point clouds (.ply)"
    )
    parser.add_argument(
        "--dist_thresh",
        type=float,
        default=0.01,
        help="distance threshold for F1 (depends on point cloud scale)"
    )
    args = parser.parse_args()

    pred_files = [
        os.path.join(args.pred_dir, f)
        for f in os.listdir(args.pred_dir)
        if f.endswith(".ply")
    ]
    pred_files.sort()

    if len(pred_files) == 0:
        print(f"No .ply files found in {args.pred_dir}")
        return

    cds = []
    f1s = []
    precisions = []
    recalls = []
    skipped = 0

    for pred_path in tqdm(pred_files, desc="Evaluating"):
        gt_name = infer_gt_name_from_pred(pred_path)
        if gt_name is None:
            print(f"[WARN] Cannot parse GT name from {pred_path}, skip.")
            skipped += 1
            continue

        gt_path = os.path.join(args.gt_dir, gt_name)
        if not os.path.exists(gt_path):
            print(f"[WARN] GT file not found: {gt_path}, skip.")
            skipped += 1
            continue

        pts_pred = load_points(pred_path)
        pts_gt = load_points(gt_path)

        cd = chamfer_distance(pts_pred, pts_gt)
        prec, rec, f1 = f1_score(pts_pred, pts_gt, args.dist_thresh)

        cds.append(cd)
        f1s.append(f1)
        precisions.append(prec)
        recalls.append(rec)

    if len(cds) == 0:
        print("No valid pairs evaluated (check naming / paths).")
        return

    print("======================================")
    print(f"#Samples evaluated : {len(cds)} (skipped: {skipped})")
    print(f"Chamfer Distance   : {np.mean(cds):.6f}")
    print(f"Precision (th={args.dist_thresh}) : {np.mean(precisions):.4f}")
    print(f"Recall   (th={args.dist_thresh}) : {np.mean(recalls):.4f}")
    print(f"F1 Score (th={args.dist_thresh}) : {np.mean(f1s):.4f}")
    print("======================================")


if __name__ == "__main__":
    main()