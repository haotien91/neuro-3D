import os
import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree
from tqdm import tqdm
import argparse
import pandas as pd
import matplotlib.pyplot as plt

def load_points(ply_path):
    pcd = o3d.io.read_point_cloud(ply_path)
    pts = np.asarray(pcd.points, dtype=np.float32)
    return pts

def chamfer_distance(points_A, points_B):
    """
    Compute Chamfer Distance between two point clouds A and B.
    CD(A, B) = mean(min_dist(A, B)^2) + mean(min_dist(B, A)^2)
    """
    tree_B = cKDTree(points_B)
    dists_A_to_B, _ = tree_B.query(points_A, k=1)
    
    tree_A = cKDTree(points_A)
    dists_B_to_A, _ = tree_A.query(points_B, k=1)

    cd = np.mean(dists_A_to_B ** 2) + np.mean(dists_B_to_A ** 2)
    return cd

def analyze_and_plot(df, output_img, title_prefix=""):
    # Set pandas display options
    pd.set_option('display.max_colwidth', None)
    pd.set_option('display.width', 1000)

    print("\n=== Top 5 Minimum Chamfer Distance (Most Similar) ===")
    print(df.nsmallest(5, 'chamfer_distance')[['filename', 'chamfer_distance']].to_string(index=False))
    
    print("\n=== Top 5 Maximum Chamfer Distance (Most Different) ===")
    print(df.nlargest(5, 'chamfer_distance')[['filename', 'chamfer_distance']].to_string(index=False))

    # Create a figure with 2 subplots (Histogram and CDF)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # --- Plot 1: Histogram ---
    # Use more bins for better resolution
    counts, bins, patches = ax1.hist(df['chamfer_distance'], bins=50, color='skyblue', edgecolor='black', alpha=0.7)
    
    # Calculate and display bin width
    bin_width = bins[1] - bins[0]
    ax1.set_title(f'{title_prefix} Histogram (Bin Width: {bin_width:.5f})')
    ax1.set_xlabel('Chamfer Distance')
    ax1.set_ylabel('Count')
    ax1.grid(axis='y', alpha=0.3)
    
    # Add mean/median lines
    mean_val = df['chamfer_distance'].mean()
    median_val = df['chamfer_distance'].median()
    ax1.axvline(mean_val, color='red', linestyle='dashed', linewidth=1, label=f'Mean: {mean_val:.4f}')
    ax1.axvline(median_val, color='green', linestyle='dashed', linewidth=1, label=f'Median: {median_val:.4f}')
    ax1.legend()

    # --- Plot 2: CDF (Cumulative Distribution Function) ---
    # CDF is often better than histogram for continuous distributions as it avoids binning bias
    sorted_data = np.sort(df['chamfer_distance'])
    yvals = np.arange(len(sorted_data)) / float(len(sorted_data) - 1)
    
    ax2.plot(sorted_data, yvals, color='purple', linewidth=2)
    ax2.set_title(f'{title_prefix} CDF')
    ax2.set_xlabel('Chamfer Distance')
    ax2.set_ylabel('Cumulative Probability')
    ax2.grid(True, alpha=0.3, which='both')
    
    # Mark the mean on CDF
    ax2.axvline(mean_val, color='red', linestyle='dashed', linewidth=1, alpha=0.5)
    ax2.text(mean_val, 0.5, f' Mean: {mean_val:.4f}', color='red', rotation=90, verticalalignment='center')

    plt.tight_layout()
    plt.savefig(output_img)
    print(f"\nChart saved to {output_img} (Includes Histogram and CDF)")

def main():
    parser = argparse.ArgumentParser(description="Compare PLY files between two directories.")
    parser.add_argument("--dir1", type=str, required=True, help="First directory (e.g., Normal)")
    parser.add_argument("--dir2", type=str, required=True, help="Second directory (e.g., Random)")
    parser.add_argument("--output_csv", type=str, default="comparison_results.csv", help="Output CSV file")
    parser.add_argument("--output_img", type=str, default="chamfer_dist_distribution.png", help="Output Histogram Image file")
    parser.add_argument("--title", type=str, default="", help="Title prefix for the charts (e.g., sub03)")
    args = parser.parse_args()

    files1 = sorted([f for f in os.listdir(args.dir1) if f.endswith(".ply")])
    files2 = set(os.listdir(args.dir2))

    results = []
    
    print(f"Comparing {len(files1)} files...")
    
    for f in tqdm(files1):
        if f not in files2:
            print(f"Warning: {f} not found in {args.dir2}")
            continue
            
        path1 = os.path.join(args.dir1, f)
        path2 = os.path.join(args.dir2, f)
        
        pts1 = load_points(path1)
        pts2 = load_points(path2)
        
        cd = chamfer_distance(pts1, pts2)
        
        # Basic stats
        mean1 = np.mean(pts1, axis=0)
        mean2 = np.mean(pts2, axis=0)
        std1 = np.std(pts1, axis=0)
        std2 = np.std(pts2, axis=0)
        
        diff_mean = np.linalg.norm(mean1 - mean2)
        diff_std = np.linalg.norm(std1 - std2)

        results.append({
            "filename": f,
            "chamfer_distance": cd,
            "diff_mean": diff_mean,
            "diff_std": diff_std
        })

    df = pd.DataFrame(results)
    df.to_csv(args.output_csv, index=False)
    print(f"Results saved to {args.output_csv}")
    
    # Analyze and Plot
    analyze_and_plot(df, args.output_img, title_prefix=args.title)

if __name__ == "__main__":
    main()
