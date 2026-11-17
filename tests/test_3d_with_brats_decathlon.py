"""
Batch validation script for Decathlon Task01_BrainTumour format.

Splits each 4D (240,240,155,4) BRATS_XXX.nii.gz file into 4 separate
single-modality NIfTI files [FLAIR, T1, T1c, T2] and sends them through
the live Flask API (/api/predict/segment) exactly like the Streamlit UI does.

Usage:
    python tests/test_3d_with_brats_decathlon.py \
        --data_dir data/brats_dataset/Task01_BrainTumour_subset \
        --num_cases 5 \
        --api_url http://localhost:5000/api

File location: tests/test_3d_with_brats_decathlon.py
"""

import argparse
import json
import os
import sys
import tempfile
import time
from pathlib import Path

import nibabel as nib
import numpy as np
import requests

# Decathlon Task01_BrainTumour channel order (official MSD spec):
# 0 = FLAIR, 1 = T1w, 2 = T1gd (T1c), 3 = T2w
CHANNEL_TO_MODALITY = {0: 'FLAIR', 1: 'T1', 2: 'T1c', 3: 'T2'}


def split_4d_volume(image_path: str, temp_dir: str) -> dict:
    """
    Split a Decathlon 4D NIfTI (H, W, D, 4) into 4 separate single-modality
    NIfTI files matching the API's expected field names.

    Returns:
        dict: {'T1': path, 'T2': path, 'FLAIR': path, 'T1c': path}
    """
    img = nib.load(image_path)
    data = img.get_fdata()
    affine = img.affine
    header = img.header

    if data.ndim != 4 or data.shape[-1] != 4:
        raise ValueError(f"Expected 4D volume with 4 channels, got shape {data.shape}")

    file_paths = {}
    case_name = Path(image_path).stem.replace('.nii', '')

    for channel_idx, modality in CHANNEL_TO_MODALITY.items():
        channel_data = data[..., channel_idx].astype(np.float32)
        channel_img = nib.Nifti1Image(channel_data, affine, header)

        out_path = os.path.join(temp_dir, f"{case_name}_{modality}.nii.gz")
        nib.save(channel_img, out_path)
        file_paths[modality] = out_path

    return file_paths


def load_label_stats(label_path: str) -> dict:
    """Extract ground-truth volume statistics from the Decathlon label mask."""
    label_img = nib.load(label_path)
    label_data = label_img.get_fdata()

    voxel_volume_mm3 = np.prod(label_img.header.get_zooms()[:3])

    return {
        'edema_voxels': int(np.sum(label_data == 1)),
        'non_enhancing_voxels': int(np.sum(label_data == 2)),
        'enhancing_voxels': int(np.sum(label_data == 3)),
        'whole_tumor_voxels': int(np.sum(label_data > 0)),
        'voxel_volume_mm3': float(voxel_volume_mm3),
    }


def run_segmentation(api_url: str, file_paths: dict, timeout: int = 120) -> dict:
    """POST the 4 modality files to the Flask API and return the parsed response."""
    files = {}
    file_handles = []

    try:
        for modality, path in file_paths.items():
            fh = open(path, 'rb')
            file_handles.append(fh)
            files[modality] = (os.path.basename(path), fh, 'application/gzip')

        start = time.time()
        response = requests.post(
            f"{api_url}/predict/segment",
            files=files,
            params={'include_3d': 'true'},
            timeout=timeout
        )
        elapsed = time.time() - start

        response.raise_for_status()
        result = response.json()
        result['_elapsed_seconds'] = elapsed
        return result

    finally:
        for fh in file_handles:
            fh.close()


def run_batch_test(data_dir: str, num_cases: int, api_url: str, output_path: str):
    images_dir = os.path.join(data_dir, 'imagesTr')
    labels_dir = os.path.join(data_dir, 'labelsTr')

    if not os.path.isdir(images_dir):
        print(f"❌ imagesTr directory not found: {images_dir}")
        sys.exit(1)

    case_files = sorted([f for f in os.listdir(images_dir) if f.endswith('.nii.gz')])
    if num_cases > 0:
        case_files = case_files[:num_cases]

    print(f"📋 Found {len(case_files)} case(s) to test")
    print(f"🎯 API endpoint: {api_url}/predict/segment")
    print("=" * 70)

    results = []
    n_success = 0
    n_failed = 0

    for i, case_file in enumerate(case_files, 1):
        case_name = case_file.replace('.nii.gz', '')
        image_path = os.path.join(images_dir, case_file)
        label_path = os.path.join(labels_dir, case_file)

        print(f"\n[{i}/{len(case_files)}] {case_name}")

        case_result = {'case': case_name, 'status': 'unknown'}

        with tempfile.TemporaryDirectory() as temp_dir:
            try:
                print("  📦 Splitting 4D volume into modalities...")
                file_paths = split_4d_volume(image_path, temp_dir)

                if os.path.exists(label_path):
                    print("  📊 Loading ground-truth label stats...")
                    case_result['ground_truth'] = load_label_stats(label_path)

                print("  🚀 Sending to API (segmentation + 3D mesh)...")
                api_result = run_segmentation(api_url, file_paths)

                elapsed = api_result.get('_elapsed_seconds', 0)
                print(f"  ✅ Success in {elapsed:.1f}s")

                case_result['status'] = 'success'
                case_result['elapsed_seconds'] = elapsed

                # Extract key metrics (guard against missing keys defensively)
                data = api_result.get('data', api_result)
                seg = data.get('segmentation', {})
                quality = data.get('quality', {})
                geometry = data.get('3d_geometry', {})

                case_result['tumor_volume_voxels'] = seg.get('tumor_volume_voxels')
                case_result['model_confidence'] = seg.get('confidence')
                case_result['segmentation_quality'] = quality.get('quality_label') or quality.get('label')

                if geometry:
                    tumor_info = geometry.get('tumor_mesh_info', {})
                    edema_info = geometry.get('edema_mesh_info', {})
                    case_result['tumor_mesh'] = {
                        'vertices': tumor_info.get('vertex_count'),
                        'faces': tumor_info.get('face_count'),
                        'attention_mean': tumor_info.get('attention_mean'),
                        'attention_max': tumor_info.get('attention_max'),
                    }
                    case_result['edema_mesh'] = {
                        'vertices': edema_info.get('vertex_count'),
                        'faces': edema_info.get('face_count'),
                    }
                    print(f"     Tumor mesh: {tumor_info.get('vertex_count')} verts, "
                          f"{tumor_info.get('face_count')} faces")
                    print(f"     Attention: mean={tumor_info.get('attention_mean')}, "
                          f"max={tumor_info.get('attention_max')}")

                n_success += 1

            except requests.exceptions.RequestException as e:
                print(f"  ❌ API request failed: {e}")
                case_result['status'] = 'api_error'
                case_result['error'] = str(e)
                n_failed += 1

            except Exception as e:
                print(f"  ❌ Failed: {e}")
                case_result['status'] = 'error'
                case_result['error'] = str(e)
                n_failed += 1

        results.append(case_result)

    # === Summary ===
    print("\n" + "=" * 70)
    print("📊 BATCH TEST SUMMARY")
    print("=" * 70)
    print(f"Total cases tested: {len(case_files)}")
    print(f"✅ Successful: {n_success}")
    print(f"❌ Failed: {n_failed}")
    print(f"Success rate: {100 * n_success / len(case_files):.1f}%")

    successful_results = [r for r in results if r['status'] == 'success']
    if successful_results:
        avg_time = np.mean([r['elapsed_seconds'] for r in successful_results])
        print(f"Average processing time: {avg_time:.1f}s per case")

        vertex_counts = [r['tumor_mesh']['vertices'] for r in successful_results
                          if r.get('tumor_mesh', {}).get('vertices')]
        if vertex_counts:
            print(f"Tumor mesh vertices: min={min(vertex_counts)}, "
                  f"max={max(vertex_counts)}, mean={np.mean(vertex_counts):.0f}")

    # Save full results
    output_data = {
        'test_run_timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'data_dir': data_dir,
        'api_url': api_url,
        'total_cases': len(case_files),
        'successful': n_success,
        'failed': n_failed,
        'success_rate_pct': round(100 * n_success / len(case_files), 1) if case_files else 0,
        'results': results,
    }

    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)

    print(f"\n💾 Full results saved to: {output_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Batch test BraTS Decathlon cases against live API')
    parser.add_argument('--data_dir', type=str,
                         default='data/brats_dataset/Task01_BrainTumour_subset',
                         help='Path to Task01_BrainTumour_subset directory (containing imagesTr/labelsTr)')
    parser.add_argument('--num_cases', type=int, default=5,
                         help='Number of cases to test (0 = all cases)')
    parser.add_argument('--api_url', type=str, default='http://localhost:5000/api',
                         help='Base URL of the Flask API')
    parser.add_argument('--output', type=str, default='brats_3d_test_results.json',
                         help='Output JSON file path')

    args = parser.parse_args()

    run_batch_test(args.data_dir, args.num_cases, args.api_url, args.output)
