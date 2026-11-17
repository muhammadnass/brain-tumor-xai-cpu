"""
Surgical Planning 3-D Viewer  ·  Neurosurgery XAI
==================================================
Renders a WebGL scene (Three.js r165) with:
  • Brain parenchyma  — semi-transparent pink cortex
  • Ventricles        — semi-transparent cerulean CSF
  • Edema / peri-tumoral zone  — translucent orange
  • Tumour            — attention-coloured jet (blue-green-red = low→high)
  • Scalp / skull     — thin wireframe outer shell (toggleable)
  • Orientation axes  — L/R  A/P  S/I corners
  • Resection margin halos  — 5 mm green, 10 mm orange wireframe rings
  • Lighting          — hemisphere + two directional lights for 3-D depth
  • All materials     — MeshPhongMaterial with computed normals

Coordinate system
-----------------
All meshes exported from mesh_generator.py are in the same 128³ voxel
coordinate system (each axis 0 – 127).  The viewer auto-centres and
auto-scales every geometry at load time so the combined scene fits
comfortably in the frustum.

Controls
--------
  Left-drag   → orbit          Scroll → zoom
  Right-drag  → pan            Checkboxes → show/hide structures
  Double-click tumour → measure resection sphere

Usage
-----
    from streamlit_3d_viewer import render_3d_viewer
    render_3d_viewer(geometry_data, height=700)
"""

from __future__ import annotations

import base64
import json
import streamlit.components.v1 as components


def render_3d_viewer(geometry_data: dict | None, height: int = 720) -> None:
    """
    Render the interactive 3-D surgical planning viewer.

    Parameters
    ----------
    geometry_data : dict
        Keys accepted:
          tumor_mesh_gltf       – base64 GLB (attention-coloured)
          edema_mesh_gltf       – base64 GLB (orange)
          brain_mesh_gltf       – base64 GLB (cortical pink, semi-transparent)
          ventricle_mesh_gltf   – base64 GLB (cerulean, semi-transparent)
          skull_mesh_gltf       – base64 GLB (grey wireframe)
          tumor_mesh_info       – {vertex_count, attention_mean, attention_max}
    height : int
        Pixel height of the viewer iframe.
    """
    if not geometry_data:
        components.html(
            "<div style='padding:20px;font-family:sans-serif;color:#aaa'>"
            "No 3-D geometry available — run segmentation first.</div>",
            height=120,
        )
        return

    def _b64(key: str) -> str:
        v = geometry_data.get(key)
        return v if v else ""

    tumor_b64    = _b64("tumor_mesh_gltf")
    edema_b64    = _b64("edema_mesh_gltf")
    brain_b64    = _b64("brain_mesh_gltf")
    ventricle_b64= _b64("ventricle_mesh_gltf")
    skull_b64    = _b64("skull_mesh_gltf")

    info = geometry_data.get("tumor_mesh_info", {})
    attn_mean = info.get("attention_mean", 0.0)
    attn_max  = info.get("attention_max",  0.0)
    n_verts   = info.get("vertex_count", 0)
    n_faces   = info.get("face_count", 0)

    meta = geometry_data.get("metadata", {})
    lod  = meta.get("lod", "preview")

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1.0">
<title>Surgical 3-D Viewer</title>
<style>
  html, body, #canvas-container {{
    background:#0a0e1a !important;
  }}
  * {{ margin:0; padding:0; box-sizing:border-box; }}
  body {{ background:#0a0e1a; font-family:'Segoe UI',Arial,sans-serif;
         overflow:hidden; color:#e0e6f0; }}
  #canvas-container {{ position:relative; width:100%; height:{height}px;
                       background:#0a0e1a; }}
  canvas {{ display:block; width:100%!important; height:100%!important;
            background:#0a0e1a; }}

  /* ── Top-right metrics panel ── */
  #metrics {{
    position:absolute; top:10px; right:10px;
    background:rgba(10,14,26,.85); border:1px solid #2a3a5c;
    border-radius:8px; padding:10px 14px; min-width:180px;
    font-size:12px; line-height:1.8; backdrop-filter:blur(4px);
  }}
  #metrics h4 {{ color:#7eb8f7; margin-bottom:6px; font-size:13px; }}
  .metric-row {{ display:flex; justify-content:space-between; gap:12px; }}
  .metric-val {{ color:#a8f0c8; font-weight:600; }}

  /* ── Controls panel (left) ── */
  #controls {{
    position:absolute; top:10px; left:10px;
    background:rgba(10,14,26,.85); border:1px solid #2a3a5c;
    border-radius:8px; padding:10px 14px; min-width:160px;
    font-size:12px; line-height:2; backdrop-filter:blur(4px);
  }}
  #controls h4 {{ color:#7eb8f7; margin-bottom:4px; font-size:13px; }}
  .ctrl-row {{ display:flex; align-items:center; gap:8px; cursor:pointer; }}
  .ctrl-row:hover {{ color:#fff; }}
  .dot {{ width:10px; height:10px; border-radius:50%; flex-shrink:0; }}

  /* ── Bottom status bar ── */
  #statusbar {{
    position:absolute; bottom:10px; left:50%; transform:translateX(-50%);
    background:rgba(10,14,26,.8); border:1px solid #2a3a5c;
    border-radius:20px; padding:5px 18px; font-size:11px;
    color:#8899bb; white-space:nowrap; pointer-events:none;
  }}

  /* ── Orientation legend ── */
  #orient {{
    position:absolute; bottom:10px; right:10px;
    background:rgba(10,14,26,.75); border:1px solid #2a3a5c;
    border-radius:8px; padding:6px 10px; font-size:11px;
    line-height:1.7; color:#99aacc;
  }}

  /* ── Loading overlay ── */
  #loading {{
    position:absolute; inset:0; display:flex; align-items:center;
    justify-content:center; background:rgba(10,14,26,.92);
    font-size:15px; color:#7eb8f7; z-index:10;
    flex-direction:column; gap:12px;
  }}
  .spinner {{
    width:36px; height:36px; border:3px solid #2a3a5c;
    border-top-color:#7eb8f7; border-radius:50%;
    animation:spin .8s linear infinite;
  }}
  @keyframes spin {{ to {{ transform:rotate(360deg); }} }}
</style>
</head>
<body>
<div id="canvas-container">
  <div id="loading"><div class="spinner"></div><span>Loading 3-D scene…</span></div>
  <canvas id="c"></canvas>

  <div id="metrics">
    <h4>🔬 Tumour Analysis</h4>
    <div class="metric-row"><span>Attention Mean</span><span class="metric-val" id="m-attn-mean">{attn_mean:.4f}</span></div>
    <div class="metric-row"><span>Attention Max</span> <span class="metric-val" id="m-attn-max">{attn_max:.4f}</span></div>
    <div class="metric-row"><span>Vertices</span>      <span class="metric-val">{n_verts:,}</span></div>
    <div class="metric-row"><span>Faces</span>         <span class="metric-val">{n_faces:,}</span></div>
    <div class="metric-row"><span>LOD</span>           <span class="metric-val">{lod}</span></div>
    <hr style="border-color:#2a3a5c;margin:6px 0">
    <div class="metric-row"><span>Tumour Radius</span><span class="metric-val" id="m-radius">—</span></div>
    <div class="metric-row"><span>5 mm margin r</span><span class="metric-val" id="m-r5">—</span></div>
    <div class="metric-row"><span>10 mm margin r</span><span class="metric-val" id="m-r10">—</span></div>
  </div>

  <div id="controls">
    <h4>🧠 Structures</h4>
    <label class="ctrl-row" id="tog-brain">
      <input type="checkbox" checked id="cb-brain">
      <span class="dot" style="background:#d2a09b;opacity:.7;border:1px solid #fff3"></span>
      Brain cortex
    </label>
    <label class="ctrl-row" id="tog-vent">
      <input type="checkbox" checked id="cb-vent">
      <span class="dot" style="background:#64b4f0"></span>
      Ventricles
    </label>
    <label class="ctrl-row" id="tog-edema">
      <input type="checkbox" checked id="cb-edema">
      <span class="dot" style="background:#ffa520"></span>
      Edema
    </label>
    <label class="ctrl-row" id="tog-tumor">
      <input type="checkbox" checked id="cb-tumor">
      <span class="dot" style="background:#ff3322"></span>
      Tumour
    </label>
    <label class="ctrl-row" id="tog-skull">
      <input type="checkbox" id="cb-skull">
      <span class="dot" style="background:#e8dccb;border:1px solid #fff5"></span>
      Scalp / skull
    </label>
    <label class="ctrl-row" id="tog-margin">
      <input type="checkbox" checked id="cb-margin">
      <span class="dot" style="background:#22ee44;border:1px solid #fff3"></span>
      Margins
    </label>
    <hr style="border-color:#2a3a5c;margin:4px 0">
    <label class="ctrl-row">
      <input type="checkbox" id="cb-xray">
      <span class="dot" style="background:none;border:1px solid #aaa"></span>
      X-ray mode
    </label>
    <button id="btn-reset" style="margin-top:6px;width:100%;background:#1a2540;
      border:1px solid #2a3a5c;color:#aabbdd;border-radius:5px;
      padding:4px;cursor:pointer;font-size:11px">⟳ Reset Camera</button>
  </div>

  <div id="statusbar" id="statusbar">Drag to orbit · Scroll to zoom · Right-drag to pan</div>

  <div id="orient">
    <b style="color:#ccd">Orientations</b><br>
    <span style="color:#f88">R</span>→Right &nbsp;
    <span style="color:#f88">L</span>→Left<br>
    <span style="color:#8f8">A</span>→Anterior &nbsp;
    <span style="color:#8f8">P</span>→Posterior<br>
    <span style="color:#88f">S</span>→Superior &nbsp;
    <span style="color:#88f">I</span>→Inferior
  </div>
</div>

<!-- Three.js r165 from cdnjs -->
<script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r134/three.min.js"></script>

<script>
// ──────────────────────────────────────────────────────────────────────────
// Mesh data (base64-encoded GLB)
// ──────────────────────────────────────────────────────────────────────────
const MESHES = {{
  tumor:     "{tumor_b64}",
  edema:     "{edema_b64}",
  brain:     "{brain_b64}",
  ventricle: "{ventricle_b64}",
  skull:     "{skull_b64}",
}};

// ──────────────────────────────────────────────────────────────────────────
// Scene setup
// ──────────────────────────────────────────────────────────────────────────
const canvas    = document.getElementById('c');
const container = document.getElementById('canvas-container');
const W = () => container.clientWidth;
const H = () => container.clientHeight;

const renderer = new THREE.WebGLRenderer({{ canvas, antialias:true, alpha:false }});
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
renderer.setSize(W(), H());
renderer.setClearColor(0x0a0e1a, 1);
renderer.shadowMap.enabled = true;
renderer.shadowMap.type = THREE.PCFSoftShadowMap;

const scene  = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(50, W()/H(), 0.1, 5000);
camera.position.set(0, 0, 300);

// ── Lighting ──────────────────────────────────────────────────────────────
// Hemisphere: warm sky, cool ground — gives natural brain-tissue shading
const hemi = new THREE.HemisphereLight(0xd0e8ff, 0x203050, 0.6);
scene.add(hemi);

// Two directional lights simulate OR theatre overhead + fill
const dir1 = new THREE.DirectionalLight(0xfff5e0, 1.0);
dir1.position.set(200, 300, 200);
dir1.castShadow = true;
dir1.shadow.mapSize.set(1024, 1024);
scene.add(dir1);

const dir2 = new THREE.DirectionalLight(0xc0d8ff, 0.4);
dir2.position.set(-150, -100, 100);
scene.add(dir2);

const ambient = new THREE.AmbientLight(0x202840, 0.4);
scene.add(ambient);

// ── Structure groups ───────────────────────────────────────────────────────
const groups = {{
  brain:     new THREE.Group(),
  ventricle: new THREE.Group(),
  edema:     new THREE.Group(),
  tumor:     new THREE.Group(),
  skull:     new THREE.Group(),
  margins:   new THREE.Group(),
  axes:      new THREE.Group(),
}};
for (const g of Object.values(groups)) scene.add(g);

// ──────────────────────────────────────────────────────────────────────────
// GLB loader (tiny inline parser — handles our specific simple GLB format)
// ──────────────────────────────────────────────────────────────────────────
function base64ToBuffer(b64) {{
  const bin = atob(b64);
  const arr = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) arr[i] = bin.charCodeAt(i);
  return arr.buffer;
}}

function parseGLB(buffer) {{
  const view = new DataView(buffer);
  // Header: magic, version, total length
  const magic = view.getUint32(0, true);
  if (magic !== 0x46546C67) throw new Error('Not a GLB file');
  // JSON chunk
  const jsonLen  = view.getUint32(12, true);
  const jsonText = new TextDecoder().decode(new Uint8Array(buffer, 20, jsonLen));
  const gltf     = JSON.parse(jsonText);
  // Binary chunk
  const binOff   = 20 + jsonLen;
  const binLen   = view.getUint32(binOff, true);
  const bin      = buffer.slice(binOff + 8, binOff + 8 + binLen);
  return {{ gltf, bin }};
}}

function buildMeshFromGLB(b64, options = {{}}) {{
  if (!b64) return null;
  try {{
    const {{ gltf, bin }} = parseGLB(base64ToBuffer(b64));
    const prim    = gltf.meshes[0].primitives[0];

    // Positions
    const posAcc  = gltf.accessors[prim.attributes.POSITION];
    const posBV   = gltf.bufferViews[posAcc.bufferView];
    const posArr  = new Float32Array(bin, posBV.byteOffset, posAcc.count * 3);

    // Indices
    const idxAcc  = gltf.accessors[prim.indices];
    const idxBV   = gltf.bufferViews[idxAcc.bufferView];
    const idxArr  = new Uint32Array(bin, idxBV.byteOffset, idxAcc.count);

    // Colours (optional)
    let colArr = null;
    if (prim.attributes.COLOR_0 !== undefined) {{
      const colAcc = gltf.accessors[prim.attributes.COLOR_0];
      const colBV  = gltf.bufferViews[colAcc.bufferView];
      colArr = new Uint8Array(bin, colBV.byteOffset, colAcc.count * 3);
    }}

    const geo = new THREE.BufferGeometry();
    geo.setAttribute('position', new THREE.BufferAttribute(posArr.slice(), 3));
    geo.setIndex(new THREE.BufferAttribute(idxArr.slice(), 1));

    if (colArr) {{
      const floatColors = new Float32Array(colArr.length);
      for (let i = 0; i < colArr.length; i++) floatColors[i] = colArr[i] / 255;
      geo.setAttribute('color', new THREE.BufferAttribute(floatColors, 3));
    }}

    // Compute smooth vertex normals for lighting
    geo.computeVertexNormals();

    const matOpts = {{
      vertexColors: colArr != null,
      side: THREE.DoubleSide,
      shininess: options.shininess ?? 40,
      specular: new THREE.Color(0x334466),
      ...options.mat,
    }};
    if (!matOpts.color && !matOpts.vertexColors) matOpts.color = 0xcccccc;

    const mat  = new THREE.MeshPhongMaterial(matOpts);
    const mesh = new THREE.Mesh(geo, mat);
    return mesh;
  }} catch (e) {{
    console.error('GLB parse error:', e);
    return null;
  }}
}}

// ──────────────────────────────────────────────────────────────────────────
// Load all meshes
// ──────────────────────────────────────────────────────────────────────────
let tumorModel = null;

function loadAll() {{
  // Transparent rendering order: outermost first so depth compositing works.
  // renderOrder: 1=brain(outermost), 2=edema, 3=ventricles, 4=tumor(opaque)

  // 1. Brain cortex — semi-transparent outer shell
  const brainMesh = buildMeshFromGLB(MESHES.brain, {{
    shininess: 25,
    mat: {{
      color: new THREE.Color(0xd2a09b),
      vertexColors: false,
      transparent: true,
      opacity: 0.28,
      depthWrite: false,
      side: THREE.BackSide,   // render back-faces only for clean inside-view
    }}
  }});
  if (brainMesh) {{
    brainMesh.renderOrder = 1;
    groups.brain.add(brainMesh);
    // Front-face pass at lower opacity for outer surface highlight
    const brainFront = buildMeshFromGLB(MESHES.brain, {{
      shininess: 40,
      mat: {{
        color: new THREE.Color(0xd2a09b),
        vertexColors: false,
        transparent: true,
        opacity: 0.18,
        depthWrite: false,
        side: THREE.FrontSide,
      }}
    }});
    if (brainFront) {{ brainFront.renderOrder = 2; groups.brain.add(brainFront); }}
  }}

  // 2. Ventricles — cerulean, semi-transparent
  const ventMesh = buildMeshFromGLB(MESHES.ventricle, {{
    shininess: 60,
    mat: {{
      color: new THREE.Color(0x64b4f0),
      vertexColors: false,
      transparent: true,
      opacity: 0.55,
      depthWrite: false,
    }}
  }});
  if (ventMesh) {{ ventMesh.renderOrder = 3; groups.ventricle.add(ventMesh); }}

  // 3. Edema — orange translucent
  const edemaMesh = buildMeshFromGLB(MESHES.edema, {{
    shininess: 20,
    mat: {{
      color: new THREE.Color(0xff8c00),
      vertexColors: false,
      transparent: true,
      opacity: 0.50,
      depthWrite: false,
    }}
  }});
  if (edemaMesh) {{ edemaMesh.renderOrder = 3; groups.edema.add(edemaMesh); }}

  // 4. Tumour — attention-coloured, fully opaque, glowing
  const tumorMesh = buildMeshFromGLB(MESHES.tumor, {{
    shininess: 80,
    mat: {{
      vertexColors: true,
      transparent: false,
      emissive: new THREE.Color(0x110800),
      emissiveIntensity: 0.3,
    }}
  }});
  if (tumorMesh) {{
    tumorMesh.renderOrder = 4;
    groups.tumor.add(tumorMesh);
    tumorModel = tumorMesh;
  }}

  // 5. Scalp / skull — faint translucent shell (registered to BraTS space)
  const skullMesh = buildMeshFromGLB(MESHES.skull, {{
    mat: {{
      color: new THREE.Color(0xe8dccb),   // bone/scalp tone
      vertexColors: false,
      transparent: true,
      opacity: 0.10,
      depthWrite: false,
      side: THREE.FrontSide,
      wireframe: false,
    }}
  }});
  if (skullMesh) {{
    skullMesh.renderOrder = 0;
    groups.skull.add(skullMesh);
    groups.skull.visible = false; // hidden by default, user toggles
  }}

  // ── Centre & scale all geometry uniformly ─────────────────────────────
  centreAndScaleScene();

  // ── Orientation axes (after centering so they stay at origin) ─────────
  addOrientationAxes();

  // ── Resection margin halos (after centering) ──────────────────────────
  if (tumorModel) addMarginSpheres();

  // ── Fit camera ────────────────────────────────────────────────────────
  fitCamera();

  document.getElementById('loading').style.display = 'none';
  document.getElementById('statusbar').textContent =
    'Drag to orbit · Scroll to zoom · Right-drag to pan';
}}

// ──────────────────────────────────────────────────────────────────────────
// Centring + uniform scale
// ──────────────────────────────────────────────────────────────────────────
function centreAndScaleScene() {{
  // Compute bounding box of all visible geometry (excl. axes/margins)
  const box = new THREE.Box3();
  for (const key of ['brain','ventricle','edema','tumor','skull']) {{
    groups[key].traverse(c => {{ if (c.isMesh) box.expandByObject(c); }});
  }}
  if (box.isEmpty()) return;

  const centre = new THREE.Vector3();
  box.getCenter(centre);
  const size   = new THREE.Vector3();
  box.getSize(size);
  const maxDim = Math.max(size.x, size.y, size.z);

  // Target: fit into a 100-unit bounding cube centred at origin
  const targetSize = 100;
  const scale = maxDim > 0 ? targetSize / maxDim : 1;

  for (const key of ['brain','ventricle','edema','tumor','skull']) {{
    const g = groups[key];
    g.traverse(c => {{
      if (c.isMesh) {{
        c.geometry.translate(-centre.x, -centre.y, -centre.z);
        c.geometry.scale(scale, scale, scale);
      }}
    }});
  }}

  // Store scale factor for margin sphere computation
  window._meshScale = scale;
  window._meshCentre = centre;

  const sizeStr = (maxDim).toFixed(1);
  const camDist = (targetSize * 2.5).toFixed(1);
  document.getElementById('statusbar').textContent =
    `Scene: ${{sizeStr}} vox → scaled ×${{scale.toFixed(3)}} | cam dist: ${{camDist}}`;
}}

// ──────────────────────────────────────────────────────────────────────────
// Camera fit
// ──────────────────────────────────────────────────────────────────────────
function fitCamera() {{
  const box = new THREE.Box3();
  for (const key of ['brain','ventricle','edema','tumor']) {{
    groups[key].traverse(c => {{ if (c.isMesh) box.expandByObject(c); }});
  }}
  if (box.isEmpty()) return;

  const size  = new THREE.Vector3();
  const cent  = new THREE.Vector3();
  box.getSize(size);
  box.getCenter(cent);
  const radius = size.length() / 2;
  const fovRad = (camera.fov * Math.PI) / 180;
  const dist   = (radius / Math.sin(fovRad / 2)) * 1.4;

  camera.position.set(cent.x, cent.y, cent.z + dist);
  controls.target.copy(cent);
  controls.minDistance = dist * 0.05;
  controls.maxDistance = dist * 5;
  controls.update();
}}

// ──────────────────────────────────────────────────────────────────────────
// Orientation axes
// ──────────────────────────────────────────────────────────────────────────
function addOrientationAxes() {{
  const ax = new THREE.AxesHelper(28);
  groups.axes.add(ax);

  // Labelled sprites
  const labels = [
    {{ text:'R', pos:[ 32,  0,  0], color:'#ff6666' }},
    {{ text:'L', pos:[-32,  0,  0], color:'#ff6666' }},
    {{ text:'A', pos:[  0, 32,  0], color:'#66ff66' }},
    {{ text:'P', pos:[  0,-32,  0], color:'#66ff66' }},
    {{ text:'S', pos:[  0,  0, 32], color:'#6688ff' }},
    {{ text:'I', pos:[  0,  0,-32], color:'#6688ff' }},
  ];
  for (const lbl of labels) {{
    const cvs = document.createElement('canvas');
    cvs.width = cvs.height = 64;
    const ctx = cvs.getContext('2d');
    ctx.font = 'bold 40px Arial';
    ctx.fillStyle = lbl.color;
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillText(lbl.text, 32, 32);
    const tex  = new THREE.CanvasTexture(cvs);
    // sizeAttenuation must stay true: with false, scale is in screen units
    // and 8 = 8x viewport height, which smeared the letters across the canvas.
    const mat  = new THREE.SpriteMaterial({{ map:tex, sizeAttenuation:true,
                                            depthTest:false, transparent:true }});
    const spr  = new THREE.Sprite(mat);
    spr.position.set(...lbl.pos);
    spr.scale.set(8, 8, 1);
    spr.renderOrder = 10;
    groups.axes.add(spr);
  }}
}}

// ──────────────────────────────────────────────────────────────────────────
// Resection margin halos  (computed in scaled scene-space)
// ──────────────────────────────────────────────────────────────────────────
function addMarginSpheres() {{
  const box = new THREE.Box3().setFromObject(tumorModel);
  const centre = new THREE.Vector3();
  box.getCenter(centre);
  const sphere = new THREE.Sphere();
  box.getBoundingSphere(sphere);
  const r = sphere.radius;

  // 1 voxel ≈ 1.875 mm for a 128³ resample from 240 mm FOV
  // After our uniform normalisation to 100 units:
  //   1 voxel_scaled = (100 / maxDim_vox) scaled units
  //   1 mm = 1 / 1.875 voxels = (100 / maxDim_vox) / 1.875 scaled units
  // But we just want margin relative to tumour size so we use a fixed
  // fraction: 5mm ≈ 12% of typical tumour diameter in glioblastoma
  // Alternatively: mm → vox → scaled_units
  const mmPerVox   = 1.875;           // BraTS 128³ ≈ 240mm / 128
  const scale      = window._meshScale || 1;
  const scaledPerMM = scale / mmPerVox;

  const r5  = r + 5  * scaledPerMM;
  const r10 = r + 10 * scaledPerMM;

  document.getElementById('m-radius').textContent = r.toFixed(1) + ' u';
  document.getElementById('m-r5').textContent     = r5.toFixed(1) + ' u';
  document.getElementById('m-r10').textContent    = r10.toFixed(1) + ' u';

  function makeHalo(radius, hexColor) {{
    const geo  = new THREE.SphereGeometry(radius, 32, 24);
    const mat  = new THREE.MeshBasicMaterial({{
      color: hexColor, wireframe: true, transparent: true, opacity: 0.35,
    }});
    const mesh = new THREE.Mesh(geo, mat);
    mesh.position.copy(centre);
    return mesh;
  }}

  groups.margins.add(makeHalo(r5,  0x22ee44));  // green  — 5 mm
  groups.margins.add(makeHalo(r10, 0xff8800));  // orange — 10 mm

  const rVox5  = (r5 / scale).toFixed(1);
  const rVox10 = (r10 / scale).toFixed(1);
  console.log(`Margin halos: 5mm r=${{rVox5}}vox, 10mm r=${{rVox10}}vox`);
}}

// ──────────────────────────────────────────────────────────────────────────
// Orbit controls (minimal inline implementation)
// ──────────────────────────────────────────────────────────────────────────
const controls = (function() {{
  let _drag = false, _rDrag = false;
  let _ox = 0, _oy = 0;
  let theta = 0, phi = Math.PI / 2;
  let radius = 300;
  const target = new THREE.Vector3();
  const up = new THREE.Vector3(0, 1, 0);

  function update() {{
    const x = target.x + radius * Math.sin(phi) * Math.sin(theta);
    const y = target.y + radius * Math.cos(phi);
    const z = target.z + radius * Math.sin(phi) * Math.cos(theta);
    camera.position.set(x, y, z);
    camera.lookAt(target);
  }}

  canvas.addEventListener('mousedown', e => {{
    if (e.button === 0) {{ _drag = true; _ox = e.clientX; _oy = e.clientY; }}
    if (e.button === 2) {{ _rDrag = true; _ox = e.clientX; _oy = e.clientY; }}
  }});
  window.addEventListener('mousemove', e => {{
    if (_drag) {{
      const dx = (e.clientX - _ox) * 0.006;
      const dy = (e.clientY - _oy) * 0.006;
      theta -= dx;
      phi    = Math.max(0.05, Math.min(Math.PI - 0.05, phi + dy));
      _ox = e.clientX; _oy = e.clientY;
      update();
    }}
    if (_rDrag) {{
      const dx = (e.clientX - _ox) * 0.15;
      const dy = (e.clientY - _oy) * 0.15;
      const right = new THREE.Vector3().crossVectors(
        new THREE.Vector3().subVectors(camera.position, target).normalize(), up
      ).normalize();
      const upVec = up.clone();
      target.addScaledVector(right, -dx);
      target.addScaledVector(upVec,  dy);
      _ox = e.clientX; _oy = e.clientY;
      update();
    }}
  }});
  window.addEventListener('mouseup', () => {{ _drag = false; _rDrag = false; }});
  canvas.addEventListener('wheel', e => {{
    radius *= 1 + e.deltaY * 0.001;
    radius  = Math.max(controls.minDistance || 1, Math.min(controls.maxDistance || 9999, radius));
    update();
    e.preventDefault();
  }}, {{ passive: false }});
  canvas.addEventListener('contextmenu', e => e.preventDefault());

  // Touch
  let _tc = null;
  canvas.addEventListener('touchstart', e => {{
    if (e.touches.length === 1) {{
      _drag = true; _ox = e.touches[0].clientX; _oy = e.touches[0].clientY;
    }}
  }});
  canvas.addEventListener('touchmove', e => {{
    if (_drag && e.touches.length === 1) {{
      const dx = (e.touches[0].clientX - _ox) * 0.006;
      const dy = (e.touches[0].clientY - _oy) * 0.006;
      theta -= dx;
      phi    = Math.max(0.05, Math.min(Math.PI - 0.05, phi + dy));
      _ox = e.touches[0].clientX; _oy = e.touches[0].clientY;
      update();
    }}
    e.preventDefault();
  }}, {{ passive: false }});
  canvas.addEventListener('touchend', () => {{ _drag = false; }});

  update();
  return {{
    target,
    minDistance: 1,
    maxDistance: 9999,
    update,
    reset() {{
      theta = 0; phi = Math.PI / 2; radius = 300;
      target.set(0, 0, 0);
      update();
      fitCamera();
    }}
  }};
}})();

// ──────────────────────────────────────────────────────────────────────────
// UI toggle handlers
// ──────────────────────────────────────────────────────────────────────────
document.getElementById('cb-brain').addEventListener('change', e =>
  {{ groups.brain.visible = e.target.checked; }});
document.getElementById('cb-vent').addEventListener('change', e =>
  {{ groups.ventricle.visible = e.target.checked; }});
document.getElementById('cb-edema').addEventListener('change', e =>
  {{ groups.edema.visible = e.target.checked; }});
document.getElementById('cb-tumor').addEventListener('change', e =>
  {{ groups.tumor.visible = e.target.checked; }});
document.getElementById('cb-skull').addEventListener('change', e =>
  {{ groups.skull.visible = e.target.checked; }});
document.getElementById('cb-margin').addEventListener('change', e =>
  {{ groups.margins.visible = e.target.checked; }});

document.getElementById('cb-xray').addEventListener('change', e => {{
  // X-ray mode: make all surfaces mostly transparent, show internal anatomy
  const xray = e.target.checked;
  let brainPass = 0;
  groups.brain.traverse(c => {{
    if (c.material) {{
      // Two brain passes: back-face (0.28 normal), front-face (0.18 normal)
      c.material.opacity = xray ? (brainPass === 0 ? 0.06 : 0.04) : (brainPass === 0 ? 0.28 : 0.18);
      brainPass++;
    }}
  }});
  groups.edema.traverse(c => {{ if (c.material) c.material.opacity = xray ? 0.15 : 0.50; }});
  groups.ventricle.traverse(c => {{ if (c.material) c.material.opacity = xray ? 0.30 : 0.55; }});
  groups.tumor.traverse(c => {{ if (c.material) {{ c.material.transparent = xray; c.material.opacity = xray ? 0.85 : 1.0; }} }});
}});

document.getElementById('btn-reset').addEventListener('click', () => {{
  controls.reset();
}});

// ──────────────────────────────────────────────────────────────────────────
// Resize handler
// ──────────────────────────────────────────────────────────────────────────
window.addEventListener('resize', () => {{
  renderer.setSize(W(), H());
  camera.aspect = W() / H();
  camera.updateProjectionMatrix();
}});

// ──────────────────────────────────────────────────────────────────────────
// Render loop
// ──────────────────────────────────────────────────────────────────────────
function animate() {{
  requestAnimationFrame(animate);
  renderer.render(scene, camera);
}}
animate();

// ──────────────────────────────────────────────────────────────────────────
// Kick off loading
// ──────────────────────────────────────────────────────────────────────────
setTimeout(loadAll, 80);   // brief delay so canvas is laid out
</script>
</body>
</html>"""

    components.html(html, height=height + 10, scrolling=False)
