"""
preprocess_datasets_v10.py - FINAL VERSION 10 (10 Coarse Classes)
Saves preprocessed segments as .ply files

- Toronto-3D: Merges L001-L004, deduplicates on (x,y,z), saves x,y,z,r,g,b,intensity,label
- Paris-Lille-3D: Uses Lille1.ply, uses 10 coarse classes, remaps to 5 unified classes
- Output field 'label' contains unified 5-class labels for modeling
- Splits into 8 segments (7 train, 1 test)
"""

import numpy as np
from pathlib import Path
from tqdm import tqdm
from plyfile import PlyData, PlyElement
import argparse

# Paris-Lille-3D: 10 coarse classes to 5 unified classes mapping
PARIS_LILLE_10_TO_5_MAP = {
    0: 4,  # unclassified → Unclassified
    1: 0,  # ground → Ground
    2: 1,  # building → Building
    3: 4,  # pole - road sign - traffic light → Unclassified
    4: 4,  # bollard - small pole → Unclassified
    5: 4,  # trash can → Unclassified
    6: 4,  # barrier → Unclassified
    7: 4,  # pedestrian → Unclassified
    8: 2,  # car → Vehicle
    9: 3   # natural - vegetation → Vegetation
}

Toronto_3D_LABEL_MAP = {
    1: 0, 2: 0,  # Ground
    4: 1,        # Building
    7: 2,        # Vehicle
    3: 3,        # Vegetation
    0: 4, 5: 4, 6: 4, 8: 4  # Unclassified
}

CLASS_NAMES = ['Ground', 'Building', 'Vehicle', 'Vegetation', 'Unclassified']


def load_toronto_files(data_dir):
    """Load and merge all Toronto-3D files with deduplication"""
    print("Loading Toronto-3D files (L001-L004)...")
    
    files = [Path(data_dir) / f"L00{i}.ply" for i in range(1, 5)]
    files = [f for f in files if f.exists()]
    
    if not files:
        raise FileNotFoundError(f"No Toronto-3D files found in {data_dir}")
    
    print(f"Found {len(files)} files: {[f.name for f in files]}")
    
    all_xyz, all_rgb, all_intensity, all_labels = [], [], [], []
    
    for fpath in files:
        print(f"  Loading {fpath.name}...")
        ply = PlyData.read(str(fpath))
        v = ply['vertex']
        
        xyz = np.c_[v['x'], v['y'], v['z']]
        rgb = np.c_[
            v['red'] if 'red' in v.data.dtype.names else np.zeros(len(v), dtype=np.uint8),
            v['green'] if 'green' in v.data.dtype.names else np.zeros(len(v), dtype=np.uint8),
            v['blue'] if 'blue' in v.data.dtype.names else np.zeros(len(v), dtype=np.uint8)
        ]
        intensity = v['scalar_Intensity'] if 'scalar_Intensity' in v.data.dtype.names else v['intensity']
        label = v['scalar_Label'] if 'scalar_Label' in v.data.dtype.names else v['label']
        
        all_xyz.append(xyz)
        all_rgb.append(rgb)
        all_intensity.append(intensity)
        all_labels.append(label)
    
    xyz = np.vstack(all_xyz)
    rgb = np.vstack(all_rgb)
    intensity = np.hstack(all_intensity)
    labels = np.hstack(all_labels)
    
    print(f"Before deduplication: {len(xyz):,} points")
    
    xyz_rounded = np.round(xyz * 100).astype(np.int32)
    _, unique_idx = np.unique(xyz_rounded, axis=0, return_index=True)
    unique_idx = np.sort(unique_idx)
    
    xyz = xyz[unique_idx]
    rgb = rgb[unique_idx]
    intensity = intensity[unique_idx]
    labels = labels[unique_idx]
    
    print(f"After deduplication: {len(xyz):,} points")
    
    return xyz, rgb, intensity, labels


def load_paris_file(data_dir):
    """Load Paris-Lille-3D Lille1.ply with 10 coarse classes
    
    Uses the 'class' field which contains 10 coarse class labels (0-9)
    """
    print("Loading Paris-Lille-3D (Lille1.ply)...")
    
    fpath = Path(data_dir) / "Lille1.ply"
    if not fpath.exists():
        fpath = Path(data_dir) / "lille1.ply"
    if not fpath.exists():
        raise FileNotFoundError(f"Lille1.ply not found in {data_dir}")
    
    ply = PlyData.read(str(fpath))
    v = ply['vertex']
    
    xyz = np.c_[v['x'], v['y'], v['z']]
    intensity = v['reflectance'] if 'reflectance' in v.data.dtype.names else v['intensity']
    
    # Use 'class' field which contains 10 coarse classes
    coarse_class = v['class'] if 'class' in v.data.dtype.names else np.zeros(len(v), dtype=np.int32)
    
    # Print unique coarse class values
    unique_classes = np.unique(coarse_class)
    print(f"10 Coarse class labels found: {unique_classes}")
    print(f"Total unique classes: {len(unique_classes)}")
    
    # Print distribution of 10 coarse classes
    print("\n10 Coarse Classes Distribution:")
    coarse_names = ['unclassified', 'ground', 'building', 'pole/sign/light', 
                    'bollard/small pole', 'trash can', 'barrier', 'pedestrian', 
                    'car', 'natural/vegetation']
    for i in range(10):
        count = np.sum(coarse_class == i)
        pct = (count / len(coarse_class)) * 100 if len(coarse_class) > 0 else 0
        print(f"  {i}: {coarse_names[i]}: {count:,} ({pct:.2f}%)")
    
    print(f"\nBefore deduplication: {len(xyz):,} points")
    
    # Deduplicate
    xyz_rounded = np.round(xyz * 100).astype(np.int32)
    _, unique_idx = np.unique(xyz_rounded, axis=0, return_index=True)
    unique_idx = np.sort(unique_idx)
    
    xyz = xyz[unique_idx]
    intensity = intensity[unique_idx]
    coarse_class = coarse_class[unique_idx]
    
    print(f"After deduplication: {len(xyz):,} points")
    
    return xyz, intensity, coarse_class


def remap_labels(labels, label_map):
    """Remap labels to unified 5-class system"""
    remapped = np.full(len(labels), 4, dtype=np.int32)
    
    for orig, unified in label_map.items():
        mask = labels == orig
        remapped[mask] = unified
    
    # Count unmapped
    unmapped_mask = np.ones(len(labels), dtype=bool)
    for orig in label_map.keys():
        unmapped_mask &= (labels != orig)
    
    num_unmapped = unmapped_mask.sum()
    if num_unmapped > 0:
        unmapped_labels = np.unique(labels[unmapped_mask])
        print(f"\nWarning: {num_unmapped} points with unmapped labels: {unmapped_labels}")
        print("These will be assigned to 'Unclassified'")
    
    print("\n5 Unified Classes Distribution:")
    total = len(remapped)
    for i, name in enumerate(CLASS_NAMES):
        count = np.sum(remapped == i)
        pct = (count / total) * 100 if total > 0 else 0
        print(f"  {i}: {name}: {count:,} ({pct:.2f}%)")
    
    return remapped


def split_by_trajectory(xyz, labels, num_segments=8):
    """Split point cloud into equal segments along Y-axis"""
    print(f"\nSplitting into {num_segments} segments...")
    
    y_vals = xyz[:, 1]
    y_min, y_max = y_vals.min(), y_vals.max()
    total_length = y_max - y_min
    seg_len = total_length / num_segments
    
    print(f"  Trajectory length: {total_length:.2f} m")
    print(f"  Segment length: {seg_len:.2f} m")
    
    segments = []
    for i in range(num_segments):
        y_start = y_min + i * seg_len
        y_end = y_min + (i + 1) * seg_len if i < num_segments - 1 else y_max + 1
        
        mask = (y_vals >= y_start) & (y_vals < y_end)
        segments.append(mask)
        
        print(f"  Segment {i+1}: {mask.sum():,} points")
    
    return segments


def save_toronto_ply(xyz, rgb, intensity, labels, out_path):
    """Save Toronto-3D segment as .ply file"""
    print(f"Saving {out_path.name}...")
    
    vertex = np.zeros(len(xyz), dtype=[
        ('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
        ('red', 'u1'), ('green', 'u1'), ('blue', 'u1'),
        ('intensity', 'f4'),
        ('label', 'i4')  # Unified 5-class label for modeling
    ])
    
    vertex['x'] = xyz[:, 0]
    vertex['y'] = xyz[:, 1]
    vertex['z'] = xyz[:, 2]
    vertex['red'] = rgb[:, 0]
    vertex['green'] = rgb[:, 1]
    vertex['blue'] = rgb[:, 2]
    vertex['intensity'] = intensity
    vertex['label'] = labels
    
    el = PlyElement.describe(vertex, 'vertex')
    PlyData([el]).write(str(out_path))


def save_paris_ply(xyz, intensity, unified_labels, out_path):
    """Save Paris-Lille-3D segment as .ply file
    
    Output fields:
    - x, y, z: coordinates
    - intensity: reflectance
    - label: unified 5-class label for modeling (0-4)
    """
    print(f"Saving {out_path.name}...")
    
    vertex = np.zeros(len(xyz), dtype=[
        ('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
        ('intensity', 'f4'),
        ('label', 'i4')  # Unified 5-class label for modeling
    ])
    
    vertex['x'] = xyz[:, 0]
    vertex['y'] = xyz[:, 1]
    vertex['z'] = xyz[:, 2]
    vertex['intensity'] = intensity
    vertex['label'] = unified_labels
    
    el = PlyElement.describe(vertex, 'vertex')
    PlyData([el]).write(str(out_path))


def save_metadata(out_dir, dataset_name, segments, labels):
    """Save preprocessing metadata"""
    import json
    
    metadata = {
        'dataset': dataset_name,
        'version': 'v10_10coarse_to_5unified',
        'output_format': '.ply',
        'deduplication': 'Applied at 1cm precision',
        'num_segments': len(segments),
        'train_segments': 7,
        'test_segments': 1,
        'unified_classes': 5,
        'class_names': CLASS_NAMES,
        'paris_lille_mapping': '10 coarse classes → 5 unified classes',
        'paris_lille_fields': 'x, y, z, intensity, label (unified 0-4)',
        'toronto_fields': 'x, y, z, red, green, blue, intensity, label (unified 0-4)',
        'segments': []
    }
    
    for i, seg_mask in enumerate(segments):
        seg_labels = labels[seg_mask]
        class_dist = {CLASS_NAMES[j]: int((seg_labels == j).sum()) for j in range(5)}
        
        metadata['segments'].append({
            'id': i,
            'split': 'train' if i < 7 else 'test',
            'num_points': int(seg_mask.sum()),
            'class_distribution': class_dist
        })
    
    with open(out_dir / 'metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\n✓ Metadata saved")


def main():
    parser = argparse.ArgumentParser(description='Preprocess datasets (v10) - 10 coarse to 5 unified')
    parser.add_argument('--dataset', required=True, choices=['Paris-Lille-3D', 'Toronto-3D', 'both'])
    parser.add_argument('--data_dir', required=True)
    parser.add_argument('--output_dir', default='./data/preprocessed')
    parser.add_argument('--num_segments', type=int, default=8)
    parser.add_argument('--train_segments', type=int, default=6)  # 6 train
    parser.add_argument('--val_segments', type=int, default=1)    # 1 validation
    parser.add_argument('--test_segments', type=int, default=1)   # 1 test
    args = parser.parse_args()

    datasets = ['Paris-Lille-3D', 'Toronto-3D'] if args.dataset == 'both' else [args.dataset]

    for dataset_name in datasets:
        print(f"\n{'='*80}")
        print(f"Processing {dataset_name.upper()} (10 Coarse → 5 Unified)")
        print(f"{'='*80}\n")

        base_out_dir = Path(args.output_dir) / dataset_name
        train_out_dir = base_out_dir / 'train'
        val_out_dir   = base_out_dir / 'val'
        test_out_dir  = base_out_dir / 'test'
        for d in [train_out_dir, val_out_dir, test_out_dir]:
            d.mkdir(parents=True, exist_ok=True)

        if dataset_name == 'Toronto-3D':
            data_path = Path(args.data_dir) / 'Toronto-3D'
            xyz, rgb, intensity, labels = load_toronto_files(data_path)
            labels = remap_labels(labels, Toronto_3D_LABEL_MAP)
            segments = split_by_trajectory(xyz, labels, args.num_segments)

            for i, seg_mask in enumerate(segments):
                # Select folder based on index
                if i < args.train_segments:
                    split = 'train'
                    out_dir = train_out_dir
                elif i < args.train_segments + args.val_segments:
                    split = 'val'
                    out_dir = val_out_dir
                else:
                    split = 'test'
                    out_dir = test_out_dir

                fname = f"Toronto-3D_{split}_segment_{i:02d}.ply"
                save_toronto_ply(
                    xyz[seg_mask], rgb[seg_mask], intensity[seg_mask], labels[seg_mask],
                    out_dir / fname
                )

            save_metadata(base_out_dir, dataset_name, segments, labels)

        else:  # Paris-Lille-3D
            data_path = Path(args.data_dir) / 'Paris-Lille-3D'
            xyz, intensity, coarse_class = load_paris_file(data_path)
            unified_labels = remap_labels(coarse_class, PARIS_LILLE_10_TO_5_MAP)
            segments = split_by_trajectory(xyz, unified_labels, args.num_segments)

            for i, seg_mask in enumerate(segments):
                if i < args.train_segments:
                    split = 'train'
                    out_dir = train_out_dir
                elif i < args.train_segments + args.val_segments:
                    split = 'val'
                    out_dir = val_out_dir
                else:
                    split = 'test'
                    out_dir = test_out_dir

                fname = f"Paris-Lille-3D_{split}_segment_{i:02d}.ply"
                save_paris_ply(
                    xyz[seg_mask], intensity[seg_mask], unified_labels[seg_mask],
                    out_dir / fname
                )

            save_metadata(base_out_dir, dataset_name, segments, unified_labels)

        print(f"\n{'='*80}")
        print(f"✓ Complete! Output folders created:")
        print(f"  • Train: {train_out_dir}")
        print(f"  • Val:   {val_out_dir}")
        print(f"  • Test:  {test_out_dir}")
        print(f"{'='*80}\n")


if __name__ == '__main__':
    main()
