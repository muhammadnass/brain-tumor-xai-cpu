"""
3D Mesh generation for brain tumor segmentation visualization
File location: src/xai_explainers/mesh_generator.py

Converts segmentation probability maps and attention maps to interactive
3D geometry (isosurfaces), with per-vertex attention value mapping.

Key design decisions:
- Marching cubes for fast isosurface extraction (CPU-compatible)
- glTF binary format for compact, web-ready geometry
- Graceful degradation: mesh generation fails → fall back to 2D
- Vertex count limited to ~50k per mesh (web performance target)

Coordinate system note
----------------------
skimage.measure.marching_cubes always returns vertex coordinates in the
**original voxel space** of the input array, regardless of the step_size
parameter.  With step_size=4 on a 128³ volume, vertices are positioned at
multiples of 4 in the range [0, 127] — NOT in a down-sampled [0, 31] space.
Therefore NO post-multiplication by step_size is needed or correct.
The previous `verts * step_size` scaling was a bug that pushed vertices
outside the attention map bounds (0-508 instead of 0-127), causing every
attention lookup to land at the clipped edge of the volume (all zeros).
"""

import numpy as np
import logging
import base64
import struct
import io
import json
from typing import Optional, Tuple, Dict, Any
from pathlib import Path

logger = logging.getLogger(__name__)

try:
    from scipy.ndimage import binary_erosion, binary_dilation
    from skimage.measure import marching_cubes
    SKIMAGE_AVAILABLE = True
except ImportError:
    logger.warning("⚠️ scikit-image not available; mesh generation disabled")
    SKIMAGE_AVAILABLE = False


class VolumetricMeshGenerator:
    """
    Convert 3D segmentation volumes + attention maps to interactive
    web-ready mesh geometry.
    """

    # glTF constants for binary format
    GLTF_MAGIC = 0x46546C67  # 'glTF' in ASCII
    GLTF_VERSION = 2
    COMPONENT_TYPE_FLOAT = 5126
    COMPONENT_TYPE_UNSIGNED_INT = 5125  # glTF spec: uint32 = 5125
    TYPE_VEC3 = 'VEC3'
    TYPE_SCALAR = 'SCALAR'

    def __init__(self, target_vertex_count: int = 50000, enable_smoothing: bool = True):
        """
        Args:
            target_vertex_count: Approximate max vertices per mesh (for simplification)
            enable_smoothing: Apply Laplacian smoothing to meshes
        """
        self.target_vertex_count = target_vertex_count
        self.enable_smoothing = enable_smoothing

        if not SKIMAGE_AVAILABLE:
            logger.warning("Mesh generation requires scikit-image and scipy")

    def segment_to_isosurface(
        self,
        segmentation_mask: np.ndarray,
        threshold: float = 0.5,
        min_component_size: int = 50,
        step_size: int = 1,
    ) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """
        Extract isosurface from segmentation probability map.

        Args:
            segmentation_mask: [D, H, W] float array
            threshold: Binarization threshold
            min_component_size: Drop components below this voxel count
            step_size: Marching-cubes subsampling factor.
                       1 = full resolution
                       2 = half resolution  (full-LOD, ~10-30 s on CPU)
                       4 = quarter res      (preview LOD, <3 s on CPU)

        Returns:
            (vertices, faces) or None

        Note: vertices are in the original voxel coordinate space [0, N-1],
        regardless of step_size.  Do NOT multiply by step_size.
        """
        if not SKIMAGE_AVAILABLE:
            logger.warning("skimage not available; skipping mesh generation")
            return None

        try:
            binary = segmentation_mask > threshold
            if binary.sum() == 0:
                logger.warning("No voxels above threshold; skipping mesh")
                return None

            if min_component_size > 0:
                from scipy.ndimage import label, sum as ndi_sum
                labeled, num_features = label(binary)
                if num_features > 0:
                    sizes = ndi_sum(binary, labeled, range(num_features + 1))
                    binary = np.isin(labeled, np.where(sizes >= min_component_size)[0])

            try:
                verts, faces = marching_cubes(
                    binary, level=0.5, step_size=step_size
                )[:2]
            except ValueError:
                logger.warning("Marching cubes failed (degenerate surface)")
                return None

            # skimage.measure.marching_cubes returns vertices in the
            # original voxel coordinate space [0, N-1] for all step_size values.
            # The step_size controls sampling density only, NOT coordinates.
            # Do NOT multiply by step_size — that was a bug causing out-of-bounds
            # attention map lookups (vertices at 0-508 clipped to edge = zeros).
            verts = verts.astype(np.float32)
            faces = faces.astype(np.uint32)

            logger.info(
                f"✅ Isosurface (step_size={step_size}): "
                f"{len(verts):,} verts, {len(faces):,} faces"
            )
            return verts, faces

        except Exception as e:
            logger.error(f"❌ Isosurface extraction failed: {e}")
            return None

    def map_attention_to_vertices(
        self,
        vertices: np.ndarray,
        attention_map: np.ndarray
    ) -> np.ndarray:
        """
        Sample attention values at mesh vertex positions.

        Args:
            vertices: [n_vertices, 3] vertex coordinates (voxel space)
            attention_map: [D, H, W] attention heatmap

        Returns:
            vertex_attention: [n_vertices] attention values (0-1 float)
        """
        try:
            from scipy.ndimage import gaussian_filter
            # Gaussian-smooth the (sparse) attention map so that the signal
            # spreads to neighbouring voxels.  sigma=2.0 voxels is a
            # conservative choice that blurs ~4-5 voxels around each peak.
            smoothed = gaussian_filter(
                attention_map.astype(np.float32), sigma=2.0
            )
            # Re-normalise after smoothing so the range stays 0-1
            s_max = smoothed.max()
            if s_max > 1e-8:
                smoothed = smoothed / s_max

            # Vectorised nearest-neighbour sampling
            shape = np.array(attention_map.shape, dtype=np.int32) - 1
            verts_int = np.clip(
                np.round(vertices).astype(np.int32), 0, shape
            )
            vertex_attention = smoothed[
                verts_int[:, 0], verts_int[:, 1], verts_int[:, 2]
            ].astype(np.float32)

            logger.info(
                f"✅ Attention mapped to vertices. "
                f"Mean: {vertex_attention.mean():.4f}, "
                f"Max:  {vertex_attention.max():.4f}"
            )
            return vertex_attention

        except Exception as e:
            logger.error(f"❌ Attention mapping failed: {e}")
            return np.ones(len(vertices), dtype=np.float32) * 0.5

    def attention_to_vertex_colors(
        self,
        vertex_attention: np.ndarray,
        colormap: str = 'jet'
    ) -> np.ndarray:
        """
        Convert attention scalars to RGB colors.

        Args:
            vertex_attention: [n_vertices] values (0-1 float)
            colormap: 'jet', 'hot', 'viridis', 'turbo', etc.

        Returns:
            colors: [n_vertices, 3] uint8 RGB values
        """
        try:
            if colormap == 'jet':
                colors = self._colormap_jet(vertex_attention)
            elif colormap == 'hot':
                colors = self._colormap_hot(vertex_attention)
            elif colormap == 'turbo':
                colors = self._colormap_turbo(vertex_attention)
            else:
                gray = (vertex_attention * 255).astype(np.uint8)
                colors = np.stack([gray, gray, gray], axis=1)

            return colors

        except Exception as e:
            logger.error(f"❌ Colormap failed: {e}")
            return np.tile([255, 0, 0], (len(vertex_attention), 1)).astype(np.uint8)

    @staticmethod
    def _colormap_jet(values: np.ndarray) -> np.ndarray:
        """Jet colormap: blue → cyan → green → yellow → red (vectorised)."""
        v = np.clip(np.asarray(values, dtype=np.float32), 0.0, 1.0)
        r = np.zeros_like(v)
        g = np.zeros_like(v)
        b = np.zeros_like(v)

        m1 = v < 0.125
        m2 = (v >= 0.125) & (v < 0.375)
        m3 = (v >= 0.375) & (v < 0.625)
        m4 = (v >= 0.625) & (v < 0.875)
        m5 = v >= 0.875

        b[m1] = v[m1] / 0.125
        g[m2] = (v[m2] - 0.125) / 0.25
        b[m2] = 1.0
        r[m3] = (v[m3] - 0.375) / 0.25
        g[m3] = 1.0
        b[m3] = 1.0 - (v[m3] - 0.375) / 0.25
        r[m4] = 1.0
        g[m4] = 1.0 - (v[m4] - 0.625) / 0.25
        r[m5] = 1.0 - (v[m5] - 0.875) / 0.125

        return np.stack([
            (r * 255).astype(np.uint8),
            (g * 255).astype(np.uint8),
            (b * 255).astype(np.uint8),
        ], axis=1)

    @staticmethod
    def _colormap_hot(values: np.ndarray) -> np.ndarray:
        """Hot colormap: black → red → yellow → white"""
        n = len(values)
        colors = np.zeros((n, 3), dtype=np.uint8)

        for i, v in enumerate(values):
            if v < 0.33:
                r = int((v / 0.33) * 255)
                g, b = 0, 0
            elif v < 0.66:
                r = 255
                g = int(((v - 0.33) / 0.33) * 255)
                b = 0
            else:
                r, g = 255, 255
                b = int(((v - 0.66) / 0.34) * 255)

            colors[i] = [r, g, b]

        return colors

    @staticmethod
    def _colormap_turbo(values: np.ndarray) -> np.ndarray:
        """Turbo colormap approximation"""
        n = len(values)
        colors = np.zeros((n, 3), dtype=np.uint8)

        for i, v in enumerate(values):
            if v < 0.2:
                r, g, b = 0, int((v / 0.2) * 128), 255
            elif v < 0.4:
                r, g, b = 0, 128 + int(((v - 0.2) / 0.2) * 127), 255 - int(((v - 0.2) / 0.2) * 127)
            elif v < 0.6:
                r, g, b = int(((v - 0.4) / 0.2) * 255), 255, 0
            elif v < 0.8:
                r, g, b = 255, 255 - int(((v - 0.6) / 0.2) * 127), 0
            else:
                r, g, b = 255, int((1 - v) / 0.2 * 127), 0

            colors[i] = [r, g, b]

        return colors

    def mesh_to_gltf_binary(
        self,
        vertices: np.ndarray,
        faces: np.ndarray,
        colors: Optional[np.ndarray] = None,
        name: str = "mesh"
    ) -> bytes:
        """
        Export mesh to glTF binary format (.glb).

        Args:
            vertices: [n_vertices, 3] float32 vertex positions
            faces: [n_faces, 3] uint32 triangle indices
            colors: [n_vertices, 3] uint8 RGB colors (optional)
            name: Mesh name

        Returns:
            glb_bytes: Complete glTF binary file
        """
        try:
            n_vertices = len(vertices)
            n_faces = len(faces)

            positions_accessor = {
                "bufferView": 0,
                "byteOffset": 0,
                "componentType": self.COMPONENT_TYPE_FLOAT,
                "count": n_vertices,
                "type": self.TYPE_VEC3,
                "min": vertices.min(axis=0).tolist(),
                "max": vertices.max(axis=0).tolist()
            }

            indices_accessor = {
                "bufferView": 1,
                "byteOffset": 0,
                "componentType": self.COMPONENT_TYPE_UNSIGNED_INT,
                "count": n_faces * 3,
                "type": self.TYPE_SCALAR
            }

            accessors = [positions_accessor, indices_accessor]

            vertices_byte_length = len(vertices) * 12
            faces_byte_length = faces.size * 4

            buffer_views = [
                {"buffer": 0, "byteOffset": 0, "byteLength": vertices_byte_length},
                {"buffer": 0, "byteOffset": vertices_byte_length, "byteLength": faces_byte_length}
            ]

            color_accessor_idx = None
            if colors is not None:
                color_accessor_idx = len(accessors)
                color_offset = vertices_byte_length + faces_byte_length
                accessors.append({
                    "bufferView": 2,
                    "byteOffset": 0,
                    "componentType": 5121,
                    "count": n_vertices,
                    "type": "VEC3",
                    "normalized": True
                })
                buffer_views.append({
                    "buffer": 0,
                    "byteOffset": color_offset,
                    "byteLength": len(colors) * 3
                })

            primitive = {
                "attributes": {"POSITION": 0},
                "indices": 1,
                "mode": 4
            }
            if color_accessor_idx is not None:
                primitive["attributes"]["COLOR_0"] = color_accessor_idx

            mesh = {
                "name": name,
                "primitives": [primitive]
            }

            gltf_json = {
                "asset": {"generator": "neurosurgery-xai-mesh", "version": "2.0"},
                "scene": 0,
                "scenes": [{"nodes": [0]}],
                "nodes": [{"mesh": 0}],
                "meshes": [mesh],
                "buffers": [{"byteLength": 0}],
                "bufferViews": buffer_views,
                "accessors": accessors
            }

            buffer_data = io.BytesIO()
            buffer_data.write(vertices.astype(np.float32).tobytes())
            faces_uint32 = faces.astype(np.uint32)
            buffer_data.write(faces_uint32.tobytes())

            if colors is not None:
                buffer_data.write(colors.tobytes())

            buffer_bytes = buffer_data.getvalue()
            gltf_json["buffers"][0]["byteLength"] = len(buffer_bytes)

            json_bytes = json.dumps(gltf_json).encode('utf-8')
            json_padding = (4 - len(json_bytes) % 4) % 4
            json_bytes += b' ' * json_padding
            json_chunk_size = len(json_bytes)

            buffer_padding = (4 - len(buffer_bytes) % 4) % 4
            buffer_bytes += b'\x00' * buffer_padding
            buffer_chunk_size = len(buffer_bytes)

            total_length = 12 + 8 + json_chunk_size + 8 + buffer_chunk_size

            glb_header = struct.pack('<III', self.GLTF_MAGIC, self.GLTF_VERSION, total_length)
            json_chunk = struct.pack('<II', json_chunk_size, 0x4E4F534A) + json_bytes
            buffer_chunk = struct.pack('<II', buffer_chunk_size, 0x004E4942) + buffer_bytes

            glb_bytes = glb_header + json_chunk + buffer_chunk

            logger.info(f"✅ glTF generated: {len(glb_bytes):,} bytes")
            return glb_bytes

        except Exception as e:
            logger.error(f"❌ glTF generation failed: {e}")
            raise

    def mesh_to_gltf_base64(
        self,
        vertices: np.ndarray,
        faces: np.ndarray,
        colors: Optional[np.ndarray] = None,
        name: str = "mesh"
    ) -> str:
        """Export mesh to base64-encoded glTF for JSON response."""
        glb_bytes = self.mesh_to_gltf_binary(vertices, faces, colors, name)
        return base64.b64encode(glb_bytes).decode('utf-8')

    def generate_full_segmentation_3d(
        self,
        tumor_mask: np.ndarray,
        edema_mask: np.ndarray,
        attention_map: np.ndarray,
        tumor_threshold: float = 0.5,
        edema_threshold: float = 0.5,
        step_size: int = 1,
    ) -> Dict[str, Any]:
        """
        Complete pipeline: segmentation → isosurfaces → attention mapping → glTF.
        step_size=4 → fast preview LOD (sent first)
        step_size=2 → full LOD (sent as background update)
        """
        geometry_data = {}

        # === Tumor mesh ===
        logger.info(f"Generating tumor mesh (step_size={step_size})…")
        tumor_iso = self.segment_to_isosurface(
            tumor_mask, threshold=tumor_threshold, step_size=step_size
        )

        if tumor_iso is not None:
            tumor_verts, tumor_faces = tumor_iso
            tumor_attention = self.map_attention_to_vertices(tumor_verts, attention_map)
            tumor_colors = self.attention_to_vertex_colors(tumor_attention, colormap='jet')
            tumor_gltf = self.mesh_to_gltf_base64(tumor_verts, tumor_faces, tumor_colors, "tumor")
            geometry_data['tumor_mesh_gltf'] = tumor_gltf
            geometry_data['tumor_mesh_info'] = {
                'vertex_count': len(tumor_verts),
                'face_count': len(tumor_faces),
                'attention_mean': float(tumor_attention.mean()),
                'attention_max': float(tumor_attention.max()),
            }
        else:
            logger.warning("⚠️ Tumor mesh generation failed; will use 2D fallback")
            geometry_data['tumor_mesh_gltf'] = None

        # === Edema mesh ===
        logger.info(f"Generating edema mesh (step_size={step_size})…")
        edema_iso = self.segment_to_isosurface(
            edema_mask, threshold=edema_threshold, step_size=step_size
        )

        if edema_iso is not None:
            edema_verts, edema_faces = edema_iso
            edema_colors = np.tile([255, 165, 0], (len(edema_verts), 1)).astype(np.uint8)
            edema_gltf = self.mesh_to_gltf_base64(edema_verts, edema_faces, edema_colors, "edema")
            geometry_data['edema_mesh_gltf'] = edema_gltf
            geometry_data['edema_mesh_info'] = {
                'vertex_count': len(edema_verts),
                'face_count': len(edema_faces),
            }
        else:
            logger.warning("⚠️ Edema mesh generation failed; will use 2D fallback")
            geometry_data['edema_mesh_gltf'] = None

        # === Metadata ===
        geometry_data['metadata'] = {
            'voxel_spacing': [1.0, 1.0, 1.0],
            'volume_shape': tumor_mask.shape,
            'mesh_generation_method': 'marching_cubes',
            'step_size': step_size,
            'lod': 'preview' if step_size >= 4 else ('full' if step_size <= 2 else 'medium'),
            'colormap': 'jet',
        }

        logger.info("✅ 3D geometry generation complete")
        return geometry_data


def create_mesh_generator(target_vertex_count: int = 50000) -> VolumetricMeshGenerator:
    """Factory function"""
    return VolumetricMeshGenerator(target_vertex_count=target_vertex_count)


def generate_skull_surface(
    skull_volume: np.ndarray,
    step_size: int = 4,
    skin_threshold: float = 0.05,
    min_component_size: int = 500,
) -> Optional[bytes]:
    """
    Extract the outer head/scalp surface from a non-skull-stripped T1 scan.
    Returns glb bytes or None.
    """
    if not SKIMAGE_AVAILABLE:
        logger.warning("scikit-image not available — skull surface disabled")
        return None

    try:
        lo, hi = skull_volume.min(), skull_volume.max()
        if hi <= lo:
            return None
        vol_norm = (skull_volume - lo) / (hi - lo)

        # Otsu threshold separates head from air robustly (a fixed 0.05
        # caught background noise and produced a box-shaped shell).
        from skimage.filters import threshold_otsu
        from scipy.ndimage import binary_fill_holes, binary_opening
        thr = max(threshold_otsu(vol_norm), skin_threshold)
        binary = vol_norm > thr
        binary = binary_opening(binary, iterations=2)
        binary = np.stack([binary_fill_holes(s) for s in binary])

        if binary.sum() < min_component_size:
            logger.warning("Skull volume: too few voxels above threshold")
            return None

        from scipy.ndimage import label, sum as ndi_sum
        labeled, n = label(binary)
        if n > 1:
            sizes   = np.array([(labeled == i).sum() for i in range(1, n + 1)])
            largest = np.argmax(sizes) + 1
            binary  = (labeled == largest)

        try:
            verts, faces = marching_cubes(
                binary, level=0.5, step_size=step_size
            )[:2]
        except ValueError:
            logger.warning("Skull marching cubes failed (degenerate surface)")
            return None

        # Vertices are already in original voxel space — no step_size scaling needed.
        verts = verts.astype(np.float32)
        faces = faces.astype(np.uint32)

        grey = np.full((len(verts), 3), 160, dtype=np.uint8)

        gen = VolumetricMeshGenerator()
        glb = gen.mesh_to_gltf_binary(verts, faces, colors=grey, name="skull")

        logger.info(
            f"Skull surface: {len(verts):,} verts, {len(faces):,} faces "
            f"(step_size={step_size})"
        )
        return glb

    except Exception as exc:
        logger.error(f"Skull surface generation failed: {exc}")
        return None


def align_head_to_brain(moving: np.ndarray, brain_ref: np.ndarray) -> np.ndarray:
    """
    Register a non-skull-stripped head scan (`moving`, e.g. BrainWeb T1) into
    the space of the skull-stripped BraTS volume (`brain_ref`).  Both inputs
    are already resampled to the same 128³ grid.

    1. SimpleITK affine registration (Mattes mutual information, moments
       initialisation) if SimpleITK is installed.
    2. Fallback: centroid + per-axis scale alignment so the moving brain
       region (head mask eroded to ~85 % extent) matches the BraTS brain mask.

    Returns the resampled `moving` volume in `brain_ref` space.
    """
    moving = moving.astype(np.float32)
    brain_ref = brain_ref.astype(np.float32)

    # ---- 1. SimpleITK affine registration ----
    try:
        import SimpleITK as sitk
        fixed_img = sitk.GetImageFromArray(brain_ref)
        moving_img = sitk.GetImageFromArray(moving)
        fixed_mask = sitk.GetImageFromArray(
            brain_mask_from_volume(brain_ref).astype(np.uint8))

        init = sitk.CenteredTransformInitializer(
            fixed_img, moving_img, sitk.AffineTransform(3),
            sitk.CenteredTransformInitializerFilter.MOMENTS)

        reg = sitk.ImageRegistrationMethod()
        reg.SetMetricAsMattesMutualInformation(numberOfHistogramBins=32)
        reg.SetMetricFixedMask(fixed_mask)
        reg.SetMetricSamplingStrategy(reg.RANDOM)
        reg.SetMetricSamplingPercentage(0.15, seed=42)
        reg.SetInterpolator(sitk.sitkLinear)
        reg.SetOptimizerAsRegularStepGradientDescent(
            learningRate=1.0, minStep=1e-3, numberOfIterations=200)
        reg.SetOptimizerScalesFromPhysicalShift()
        reg.SetShrinkFactorsPerLevel([4, 2, 1])
        reg.SetSmoothingSigmasPerLevel([2, 1, 0])
        reg.SetInitialTransform(init, inPlace=False)
        tx = reg.Execute(fixed_img, moving_img)

        out = sitk.Resample(moving_img, fixed_img, tx, sitk.sitkLinear,
                            float(moving.min()))
        logger.info(f"✅ Head scan registered (SimpleITK affine, "
                    f"metric={reg.GetMetricValue():.4f})")
        return sitk.GetArrayFromImage(out).astype(np.float32)
    except ImportError:
        logger.info("SimpleITK not installed — using moment-based alignment")
    except Exception as exc:
        logger.warning(f"SimpleITK registration failed ({exc}) — falling back")

    # ---- 2. Moment / bounding-box fallback ----
    from scipy.ndimage import affine_transform, binary_fill_holes
    from skimage.filters import threshold_otsu

    ref_mask = brain_mask_from_volume(brain_ref)
    norm = (moving - moving.min()) / (float(np.ptp(moving)) or 1.0)
    head = norm > threshold_otsu(norm)
    head = np.stack([binary_fill_holes(s) for s in head])
    if ref_mask.sum() == 0 or head.sum() == 0:
        return moving

    def stats(m):
        idx = np.argwhere(m)
        return idx.mean(0), idx.max(0) - idx.min(0) + 1

    c_ref, e_ref = stats(ref_mask)
    c_mov, e_mov = stats(head)
    # Brain occupies ~85 % of the head's extent along each axis
    scale = (e_mov * 0.85) / e_ref          # output→input scaling
    offset = c_mov - scale * c_ref
    out = affine_transform(moving, scale, offset=offset, order=1,
                           mode='constant', cval=float(moving.min()))
    logger.info(f"✅ Head scan aligned (moments): scale={np.round(1/scale, 2)}")
    return out.astype(np.float32)


def brain_mask_from_volume(vol: np.ndarray) -> np.ndarray:
    """
    Brain mask that works for both raw (background=0) and z-score normalised
    (background = constant negative value) skull-stripped volumes.
    Background value is taken from the volume corner, which is always outside
    the head in BraTS.
    """
    bg = vol[0, 0, 0]
    rng = float(vol.max() - vol.min()) or 1.0
    return np.abs(vol - bg) > 1e-3 * rng


def generate_brain_surface(
    t1_volume: np.ndarray,
    step_size: int = 6,
    min_component_size: int = 5000,
) -> Optional[bytes]:
    """
    Extract a smooth outer brain surface from a skull-stripped T1/FLAIR channel.

    Strategy:
      1. Threshold to get brain mask
      2. Keep only largest connected component
      3. binary_fill_holes — fills ventricles / internal CSF so the surface
         is a solid convex-ish shell, not a sponge of sulcal fragments
      4. Strong morphological closing to smooth remaining sulci
      5. Marching cubes at step_size=6 — coarser mesh, faster, less noisy

    This gives a single clean semi-transparent shell (like an OR skull-opening
    view) rather than thousands of gyral fragments that obscure the tumour.
    """
    if not SKIMAGE_AVAILABLE:
        return None

    try:
        from scipy.ndimage import (
            label, binary_fill_holes, binary_closing,
            generate_binary_structure,
        )

        brain_mask = brain_mask_from_volume(t1_volume)

        if brain_mask.sum() < min_component_size or brain_mask.mean() > 0.9:
            logger.warning("Brain surface: mask empty or covers whole volume")
            return None

        # 1. Keep only the largest connected component (whole brain)
        labeled, n = label(brain_mask)
        if n > 1:
            sizes   = np.array([(labeled == i).sum() for i in range(1, n + 1)])
            largest = np.argmax(sizes) + 1
            brain_mask = (labeled == largest)

        # 2. Fill internal voids (ventricles, sulci) → solid hull
        # Process slice-by-slice in all three orientations for speed
        for ax in range(3):
            for sl in range(brain_mask.shape[ax]):
                idx = [slice(None)] * 3
                idx[ax] = sl
                brain_mask[tuple(idx)] = binary_fill_holes(brain_mask[tuple(idx)])

        # 3. 3-D closing with 26-connectivity kernel to smooth surface lumps
        struct_el = generate_binary_structure(3, 2)   # 26-neighbours
        brain_mask = binary_closing(brain_mask, structure=struct_el, iterations=3)

        try:
            verts, faces = marching_cubes(
                brain_mask.astype(np.uint8), level=0.5, step_size=step_size
            )[:2]
        except ValueError:
            logger.warning("Brain surface marching cubes failed")
            return None

        verts = verts.astype(np.float32)
        faces = faces.astype(np.uint32)

        # Cortical pink (real brain tissue colour under OR lighting)
        cortical_pink = np.array([210, 160, 155], dtype=np.uint8)
        colors = np.tile(cortical_pink, (len(verts), 1)).astype(np.uint8)

        gen = VolumetricMeshGenerator()
        glb = gen.mesh_to_gltf_binary(verts, faces, colors=colors, name="brain")

        logger.info(
            f"✅ Brain surface: {len(verts):,} verts, {len(faces):,} faces "
            f"(step_size={step_size})"
        )
        return glb

    except Exception as exc:
        logger.error(f"Brain surface generation failed: {exc}")
        return None


def generate_ventricles(
    t1_volume: np.ndarray,
    brain_mask: Optional[np.ndarray] = None,
    step_size: int = 4,
    csf_percentile: float = 12.0,
    min_component_size: int = 200,
) -> Optional[bytes]:
    """
    Extract ventricular CSF surface from a skull-stripped T1 volume.

    Ventricles appear dark in T1-weighted MRI (CSF signal).  Within the
    skull-stripped brain, voxels below ~12th percentile of brain intensity
    are typically CSF.  We keep only the largest dark connected component
    near the brain centre to avoid scalp/air artefacts.

    Args:
        t1_volume:   [D, H, W] skull-stripped T1 channel.
        brain_mask:  Optional [D, H, W] bool mask of brain region.  If None,
                     derived from t1_volume > 0.
        step_size:   Marching-cubes step.
        csf_percentile: Intensity percentile threshold within brain for CSF.
        min_component_size: Drop CSF fragments below this count.

    Returns:
        glb_bytes or None.
    """
    if not SKIMAGE_AVAILABLE:
        return None

    try:
        if brain_mask is None:
            brain_mask = brain_mask_from_volume(t1_volume)

        brain_voxels = t1_volume[brain_mask]
        if len(brain_voxels) < 1000:
            return None

        csf_threshold = np.percentile(brain_voxels, csf_percentile)

        # CSF: dark T1 intensity within brain
        csf_mask = (t1_volume <= csf_threshold) & brain_mask

        if csf_mask.sum() < min_component_size:
            logger.info("No significant ventricular CSF found")
            return None

        # Ventricles are deep structures: drop sulcal CSF near the cortex by
        # keeping only voxels ≥ ~12 voxels (~22 mm) from the brain surface.
        from scipy.ndimage import label, distance_transform_edt, binary_fill_holes
        filled = np.stack([binary_fill_holes(s) for s in brain_mask])
        depth = distance_transform_edt(filled)
        csf_mask &= depth >= 12

        # Keep up to the 2 largest deep components (left/right lateral)
        labeled, n = label(csf_mask)
        if n == 0:
            return None
        sizes = np.bincount(labeled.ravel())[1:]
        keep = np.argsort(sizes)[::-1][:2] + 1
        keep = [k for k in keep if sizes[k - 1] >= min_component_size // 2]
        csf_mask = np.isin(labeled, keep)

        if csf_mask.sum() < min_component_size:
            return None

        # Smooth the binary mask so the surface isn't voxel-blocky
        from scipy.ndimage import gaussian_filter
        csf_smooth = gaussian_filter(csf_mask.astype(np.float32), sigma=1.2)

        try:
            verts, faces = marching_cubes(
                csf_smooth, level=0.5, step_size=step_size
            )[:2]
        except ValueError:
            return None

        verts = verts.astype(np.float32)
        faces = faces.astype(np.uint32)

        # Cerulean blue for CSF/ventricles
        csf_blue = np.array([100, 180, 240], dtype=np.uint8)
        colors = np.tile(csf_blue, (len(verts), 1)).astype(np.uint8)

        gen = VolumetricMeshGenerator()
        glb = gen.mesh_to_gltf_binary(verts, faces, colors=colors, name="ventricles")

        logger.info(
            f"✅ Ventricles: {len(verts):,} verts, {len(faces):,} faces"
        )
        return glb

    except Exception as exc:
        logger.error(f"Ventricle extraction failed: {exc}")
        return None


if __name__ == "__main__":
    print("✅ Mesh generation module loaded successfully")
