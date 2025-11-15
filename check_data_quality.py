"""
Data Quality Check Script for Vietnamese Sign Language Recognition
Analyzes:
- Sample counts per label
- Shape and missing values
- Landmark distribution
- Variance and outliers
- Visualize samples
"""

import numpy as np
import json
import os
import glob
import sys
from collections import defaultdict
import matplotlib.pyplot as plt

# Fix UTF-8 encoding for Windows console
if sys.platform == 'win32':
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'ignore')

# Configuration
DATA_PATH = 'Data'
LABEL_MAP_PATH = 'Logs/label_map.json'
OUTPUT_DIR = 'Logs/data_quality_report'

# Create output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 70)
print("DATA QUALITY CHECK REPORT")
print("=" * 70)

# 1. Load label map
print("\n[1/6] Loading label map...")
with open(LABEL_MAP_PATH, 'r', encoding='utf-8') as f:
    label_map = json.load(f)

print(f"   + Found {len(label_map)} labels")

# 2. Count samples per label
print("\n[2/6] Counting samples...")
label_counts = defaultdict(int)
all_files = []

for label_name, label_id in label_map.items():
    label_path = os.path.join(DATA_PATH, label_name)
    if os.path.exists(label_path):
        files = glob.glob(os.path.join(label_path, '*.npz'))
        label_counts[label_name] = len(files)
        all_files.extend(files)

# Print statistics table
print("\n   SAMPLE DISTRIBUTION:")
print("   " + "-" * 50)
print(f"   {'Label':<20} {'Samples':>15} {'%':>10}")
print("   " + "-" * 50)
total_samples = sum(label_counts.values())
for label, count in sorted(label_counts.items(), key=lambda x: x[1], reverse=True):
    percentage = (count / total_samples) * 100
    print(f"   {label:<20} {count:>15} {percentage:>9.1f}%")
print("   " + "-" * 50)
print(f"   {'TOTAL':<20} {total_samples:>15} {100.0:>9.1f}%")
print("   " + "-" * 50)

# Check balance
min_count = min(label_counts.values())
max_count = max(label_counts.values())
imbalance_ratio = max_count / min_count if min_count > 0 else float('inf')

if imbalance_ratio > 1.5:
    print(f"\n   WARNING: Imbalanced dataset (ratio: {imbalance_ratio:.2f}x)")
    print(f"      Min label: {min_count} samples")
    print(f"      Max label: {max_count} samples")
else:
    print(f"\n   OK: Balanced dataset (ratio: {imbalance_ratio:.2f}x)")

# 3. Check shape and missing values
print("\n[3/6] Checking shapes and missing values...")
shape_errors = []
label_errors = []
missing_values = []
all_shapes = set()

for i, file_path in enumerate(all_files):
    try:
        data = np.load(file_path)
        
        # Check keys
        if 'sequence' not in data or 'label' not in data:
            shape_errors.append(f"{file_path}: Missing 'sequence' or 'label' key")
            continue
        
        seq = data['sequence']
        lbl = data['label']
        
        # Check shape
        all_shapes.add(seq.shape)
        if seq.shape != (60, 201):
            shape_errors.append(f"{file_path}: Shape {seq.shape} (expected (60, 201))")
        
        # Check missing values (NaN, Inf)
        if np.isnan(seq).any():
            missing_values.append(f"{file_path}: Contains NaN values")
        if np.isinf(seq).any():
            missing_values.append(f"{file_path}: Contains Inf values")
        
        # Check label validity
        if lbl < 0 or lbl >= len(label_map):
            label_errors.append(f"{file_path}: Invalid label {lbl} (max: {len(label_map)-1})")
            
    except Exception as e:
        shape_errors.append(f"{file_path}: Error loading - {str(e)}")

print(f"\n   Shapes found: {all_shapes}")
print(f"   + Checked {len(all_files)} files")

if shape_errors:
    print(f"\n   ERROR: {len(shape_errors)} shape errors:")
    for err in shape_errors[:5]:  # Show first 5
        print(f"      - {err}")
    if len(shape_errors) > 5:
        print(f"      ... and {len(shape_errors) - 5} more errors")
else:
    print(f"   OK: All files have correct shape (60, 201)")

if missing_values:
    print(f"\n   ERROR: {len(missing_values)} files with missing values:")
    for err in missing_values[:5]:
        print(f"      - {err}")
else:
    print(f"   OK: No NaN/Inf values")

if label_errors:
    print(f"\n   ERROR: {len(label_errors)} label errors:")
    for err in label_errors[:5]:
        print(f"      - {err}")
else:
    print(f"   OK: All labels are valid")

# 4. Analyze landmark distribution
print("\n[4/6] Analyzing landmark distribution...")

# Sample 50 files randomly
sample_files = np.random.choice(all_files, min(50, len(all_files)), replace=False)

zero_counts = []  # Count keypoints = 0 in each sample
means = []
stds = []

for file_path in sample_files:
    try:
        data = np.load(file_path)
        seq = data['sequence']
        
        # Count zero keypoints (missing)
        zero_count = np.sum(np.all(seq == 0, axis=0))
        zero_counts.append(zero_count)
        
        # Calculate mean and std
        means.append(np.mean(seq))
        stds.append(np.std(seq))
        
    except:
        continue

if zero_counts:
    avg_zeros = np.mean(zero_counts)
    max_zeros = np.max(zero_counts)
    
    print(f"\n   Analysis on {len(sample_files)} samples:")
    print(f"      - Average {avg_zeros:.1f}/201 keypoints are zero")
    print(f"      - Maximum {max_zeros}/201 keypoints are zero")
    print(f"      - Mean value: {np.mean(means):.4f} +/- {np.std(means):.4f}")
    print(f"      - Std value: {np.mean(stds):.4f} +/- {np.std(stds):.4f}")
    
    if avg_zeros > 50:
        print(f"\n   WARNING: Too many missing keypoints!")
        print(f"      -> Check original videos, Mediapipe might not detect well")
    elif avg_zeros > 20:
        print(f"\n   WARNING: Some keypoints are missing (acceptable)")
    else:
        print(f"\n   OK: Most keypoints are detected well")

# 5. Analyze variance within each label
print("\n[5/6] Analyzing variance within each label...")

label_variances = {}

for label_name in label_map.keys():
    label_path = os.path.join(DATA_PATH, label_name)
    if not os.path.exists(label_path):
        continue
    
    files = glob.glob(os.path.join(label_path, '*.npz'))
    
    # Sample 20 files to calculate variance
    sample_files = files[:min(20, len(files))]
    sequences = []
    
    for file_path in sample_files:
        try:
            data = np.load(file_path)
            sequences.append(data['sequence'].flatten())
        except:
            continue
    
    if len(sequences) > 1:
        sequences = np.array(sequences)
        variance = np.mean(np.var(sequences, axis=0))
        label_variances[label_name] = variance

print("\n   VARIANCE PER LABEL:")
print("   " + "-" * 50)
print(f"   {'Label':<20} {'Variance':>15} {'Status':>12}")
print("   " + "-" * 50)

avg_var = np.mean(list(label_variances.values()))

for label, var in sorted(label_variances.items(), key=lambda x: x[1], reverse=True):
    if var < avg_var * 0.5:
        status = "Too Low"
    elif var > avg_var * 2:
        status = "Too High"
    else:
        status = "Good"
    print(f"   {label:<20} {var:>15.6f} {status:>12}")

print("   " + "-" * 50)
print(f"   Average: {avg_var:.6f}")
print("\n   Interpretation:")
print("      - Too low variance -> Data too similar (lack diversity)")
print("      - Too high variance -> Data inconsistent (has outliers)")

# 6. Create visualization
print("\n[6/6] Creating visualization...")

fig, axes = plt.subplots(2, 2, figsize=(15, 12))
fig.suptitle('DATA QUALITY REPORT', fontsize=16, fontweight='bold')

# Plot 1: Sample counts
ax1 = axes[0, 0]
labels = list(label_counts.keys())
counts = list(label_counts.values())
bars = ax1.bar(labels, counts, color='steelblue', alpha=0.8)
ax1.set_xlabel('Label', fontsize=12)
ax1.set_ylabel('Number of samples', fontsize=12)
ax1.set_title('Sample Distribution', fontsize=13, fontweight='bold')
ax1.tick_params(axis='x', rotation=45)

# Add values on bars
for bar in bars:
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height,
             f'{int(height)}',
             ha='center', va='bottom', fontsize=10)

# Plot 2: Zero counts distribution
ax2 = axes[0, 1]
if zero_counts:
    ax2.hist(zero_counts, bins=20, color='coral', alpha=0.7, edgecolor='black')
    ax2.axvline(np.mean(zero_counts), color='red', linestyle='--', 
                linewidth=2, label=f'Mean: {np.mean(zero_counts):.1f}')
    ax2.set_xlabel('Number of zero keypoints', fontsize=12)
    ax2.set_ylabel('Number of samples', fontsize=12)
    ax2.set_title('Missing Keypoints Distribution', fontsize=13, fontweight='bold')
    ax2.legend()

# Plot 3: Variance comparison
ax3 = axes[1, 0]
labels_var = list(label_variances.keys())
variances = list(label_variances.values())
bars = ax3.barh(labels_var, variances, color='forestgreen', alpha=0.8)
ax3.set_xlabel('Variance', fontsize=12)
ax3.set_ylabel('Label', fontsize=12)
ax3.set_title('Variance per Label', fontsize=13, fontweight='bold')

# Add average line
ax3.axvline(avg_var, color='red', linestyle='--', 
            linewidth=2, label=f'Avg: {avg_var:.4f}')
ax3.legend()

# Plot 4: Sample mean/std distribution
ax4 = axes[1, 1]
if means and stds:
    ax4.scatter(means, stds, alpha=0.6, s=50, c='purple')
    ax4.set_xlabel('Mean value', fontsize=12)
    ax4.set_ylabel('Std value', fontsize=12)
    ax4.set_title('Mean vs Std Distribution', fontsize=13, fontweight='bold')
    ax4.grid(True, alpha=0.3)

plt.tight_layout()
report_path = os.path.join(OUTPUT_DIR, 'data_quality_report.png')
plt.savefig(report_path, dpi=150, bbox_inches='tight')
print(f"   + Saved visualization: {report_path}")

# 7. Create summary report JSON
summary = {
    "total_samples": total_samples,
    "num_labels": len(label_map),
    "label_distribution": dict(label_counts),
    "imbalance_ratio": float(imbalance_ratio),
    "shape_errors": len(shape_errors),
    "missing_values": len(missing_values),
    "label_errors": len(label_errors),
    "avg_zero_keypoints": float(np.mean(zero_counts)) if zero_counts else 0,
    "label_variances": {k: float(v) for k, v in label_variances.items()},
    "avg_variance": float(avg_var) if label_variances else 0
}

summary_path = os.path.join(OUTPUT_DIR, 'data_quality_summary.json')
with open(summary_path, 'w', encoding='utf-8') as f:
    json.dump(summary, f, ensure_ascii=False, indent=2)

print(f"   + Saved summary: {summary_path}")

# 8. Final summary
print("\n" + "=" * 70)
print("DATA QUALITY ASSESSMENT SUMMARY")
print("=" * 70)

score = 100  # Maximum score

# Deduct points based on issues
if imbalance_ratio > 2:
    score -= 20
    print("X Dataset severely imbalanced (-20 points)")
elif imbalance_ratio > 1.5:
    score -= 10
    print("! Dataset slightly imbalanced (-10 points)")
else:
    print("+ Dataset well balanced")

if shape_errors:
    score -= 15
    print(f"X {len(shape_errors)} files have shape errors (-15 points)")
else:
    print("+ All files have correct shape")

if missing_values:
    score -= 15
    print(f"X {len(missing_values)} files have missing values (-15 points)")
else:
    print("+ No missing values")

if zero_counts:
    if avg_zeros > 50:
        score -= 25
        print(f"X Too many missing keypoints ({avg_zeros:.1f}/201) (-25 points)")
    elif avg_zeros > 20:
        score -= 10
        print(f"! Some keypoints missing ({avg_zeros:.1f}/201) (-10 points)")
    else:
        print(f"+ Keypoints detected well ({avg_zeros:.1f}/201 missing)")

if total_samples < 500:
    score -= 15
    print(f"! Dataset is small ({total_samples} samples) (-15 points)")
elif total_samples < 1000:
    score -= 5
    print(f"! Dataset is moderate ({total_samples} samples) (-5 points)")
else:
    print(f"+ Dataset has good size ({total_samples} samples)")

print("\n" + "=" * 70)
print(f"OVERALL QUALITY SCORE: {score}/100")
print("=" * 70)

if score >= 80:
    print("+ Data quality is GOOD - Ready for training!")
elif score >= 60:
    print("! Data quality is MODERATE - Should improve")
else:
    print("X Data quality is POOR - Needs serious improvement")

print("\nRECOMMENDATIONS:")
if total_samples < 500:
    print("   1. Collect more original videos (target: >20 videos/label)")
if imbalance_ratio > 1.5:
    print("   2. Balance dataset by adding samples to minority labels")
if zero_counts and avg_zeros > 20:
    print("   3. Improve video quality (lighting, angle, background)")
    print("   4. Ensure hands/person are fully in frame")
if shape_errors or missing_values:
    print("   5. Re-run create_data_augment.py to fix data errors")

print("\n+ Complete! See details in:", OUTPUT_DIR)
print("=" * 70)
