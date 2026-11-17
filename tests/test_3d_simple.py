#!/usr/bin/env python
"""
Simple 3D Mesh Generation Test
Tests that mesh_generator.py works correctly
"""

import sys
from pathlib import Path
import numpy as np

# Add src/ to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

print("=" * 70)
print("🧪 SIMPLE 3D MESH GENERATION TEST")
print("=" * 70)

# Test 1: Import mesh_generator
print("\n[TEST 1] Importing mesh_generator...")
try:
    from xai_explainers.mesh_generator import VolumetricMeshGenerator
    print("✅ PASS: mesh_generator imported successfully")
except ImportError as e:
    print(f"❌ FAIL: Could not import mesh_generator: {e}")
    sys.exit(1)

# Test 2: Import skimage
print("\n[TEST 2] Importing skimage.measure.marching_cubes...")
try:
    from skimage.measure import marching_cubes
    print("✅ PASS: marching_cubes imported successfully")
except ImportError as e:
    print(f"❌ FAIL: Could not import marching_cubes: {e}")
    sys.exit(1)

# Test 3: Create a simple synthetic tumor
print("\n[TEST 3] Creating synthetic tumor volume...")
try:
    volume_shape = (128, 128, 128)
    tumor_mask = np.zeros(volume_shape, dtype=np.float32)
    
    # Create a small sphere
    center = np.array([64, 64, 64])
    radius = 10
    for d in range(max(0, center[0]-radius), min(volume_shape[0], center[0]+radius+1)):
        for h in range(max(0, center[1]-radius), min(volume_shape[1], center[1]+radius+1)):
            for w in range(max(0, center[2]-radius), min(volume_shape[2], center[2]+radius+1)):
                dist = np.sqrt((d-center[0])**2 + (h-center[1])**2 + (w-center[2])**2)
                if dist <= radius:
                    tumor_mask[d, h, w] = min(1.0, 1.0 - dist/radius)
    
    print(f"✅ PASS: Created synthetic tumor ({tumor_mask.sum():.0f} voxels)")
except Exception as e:
    print(f"❌ FAIL: Could not create synthetic tumor: {e}")
    sys.exit(1)

# Test 4: Extract isosurface
print("\n[TEST 4] Extracting isosurface with marching cubes...")
try:
    binary = tumor_mask > 0.5
    verts, faces = marching_cubes(binary, level=0.5)[:2]
    verts = verts.astype(np.float32)
    faces = faces.astype(np.uint32)
    print(f"✅ PASS: Isosurface extracted ({len(verts):,} vertices, {len(faces):,} faces)")
except Exception as e:
    print(f"❌ FAIL: Could not extract isosurface: {e}")
    sys.exit(1)

# Test 5: Generate glTF
print("\n[TEST 5] Generating glTF binary file...")
try:
    mesh_gen = VolumetricMeshGenerator()
    gltf_bytes = mesh_gen.mesh_to_gltf_binary(verts, faces, name="test_tumor")
    
    # Verify glTF magic number
    magic = int.from_bytes(gltf_bytes[0:4], 'little')
    version = int.from_bytes(gltf_bytes[4:8], 'little')
    
    if magic != 0x46546C67:  # 'glTF'
        raise ValueError(f"Invalid magic number: {hex(magic)}")
    
    if version != 2:
        raise ValueError(f"Invalid version: {version}")
    
    print(f"✅ PASS: glTF generated ({len(gltf_bytes):,} bytes, magic: OK, version: 2)")
except Exception as e:
    print(f"❌ FAIL: Could not generate glTF: {e}")
    sys.exit(1)

# Test 6: Test color mapping
print("\n[TEST 6] Testing Grad-CAM color mapping...")
try:
    attention_map = tumor_mask  # Use tumor mask as dummy attention
    vertex_colors = mesh_gen.attention_to_vertex_colors(
        np.random.rand(len(verts)).astype(np.float32),
        colormap='jet'
    )
    
    if vertex_colors.shape != (len(verts), 3):
        raise ValueError(f"Wrong color shape: {vertex_colors.shape}")
    
    print(f"✅ PASS: Attention color mapping works ({vertex_colors.shape} colors)")
except Exception as e:
    print(f"❌ FAIL: Could not map attention colors: {e}")
    sys.exit(1)

# Test 7: End-to-end pipeline
print("\n[TEST 7] Testing end-to-end 3D generation pipeline...")
try:
    edema_mask = np.zeros_like(tumor_mask)
    attention_map = tumor_mask
    
    geometry = mesh_gen.generate_full_segmentation_3d(
        tumor_mask=tumor_mask,
        edema_mask=edema_mask,
        attention_map=attention_map,
        tumor_threshold=0.5,
        edema_threshold=0.5
    )
    
    if not geometry.get('tumor_mesh_gltf'):
        raise ValueError("No tumor mesh in geometry")
    
    print(f"✅ PASS: End-to-end pipeline successful")
    print(f"   - Tumor mesh: {len(geometry['tumor_mesh_gltf'])} bytes (base64)")
    print(f"   - Tumor vertices: {geometry['tumor_mesh_info']['vertex_count']:,}")
    print(f"   - Tumor faces: {geometry['tumor_mesh_info']['face_count']:,}")
except Exception as e:
    print(f"❌ FAIL: End-to-end pipeline failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Summary
print("\n" + "=" * 70)
print("✅ ALL TESTS PASSED!")
print("=" * 70)
print("\n🎉 Your 3D visualization backend is working correctly!")
print("\nNext steps:")
print("1. Start the Flask API: python src/app.py")
print("2. Test with real MRI data: curl -X POST http://localhost:5000/api/predict/segment ...")
print("3. Check response has '3d_available': true")
