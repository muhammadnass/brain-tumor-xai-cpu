"""
3D Viewer with base64-encoded Three.js libraries (handles special chars)
File location: frontend/streamlit_3d_viewer.py

Reads local libraries, encodes as base64, decodes in JS to avoid f-string issues
"""

import streamlit as st
import base64
import os
from typing import Optional, Dict, Any


def render_3d_viewer(
    tumor_gltf_b64: str,
    edema_gltf_b64: Optional[str] = None,
    skull_gltf_b64: Optional[str] = None,
    attention_stats: Optional[Dict[str, Any]] = None,
    height: int = 700,
    key: str = "3d_viewer"
) -> None:
    """
    Render 3D viewer with base64-encoded local libraries
    """
    
    # Get paths to local libraries
    base_dir = os.path.dirname(os.path.abspath(__file__))
    three_path = os.path.join(base_dir, 'libs', 'three.min.js')
    loader_path = os.path.join(base_dir, 'libs', 'GLTFLoader.js')
    
    # Read and encode library files as base64
    three_b64 = ""
    loader_b64 = ""
    
    try:
        if os.path.exists(three_path):
            with open(three_path, 'rb') as f:
                three_b64 = base64.b64encode(f.read()).decode('utf-8')
            st.sidebar.success("✅ three.min.js loaded")
        else:
            st.sidebar.error(f"❌ three.min.js not found")
            return
    except Exception as e:
        st.sidebar.error(f"❌ Error reading three.min.js: {e}")
        return
    
    try:
        if os.path.exists(loader_path):
            with open(loader_path, 'rb') as f:
                loader_b64 = base64.b64encode(f.read()).decode('utf-8')
            st.sidebar.success("✅ GLTFLoader.js loaded")
        else:
            st.sidebar.error(f"❌ GLTFLoader.js not found")
            return
    except Exception as e:
        st.sidebar.error(f"❌ Error reading GLTFLoader.js: {e}")
        return
    
    # Build HTML with base64-encoded libraries
    html_template = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
        <style>
            * {{ margin: 0; padding: 0; }}
            body {{ 
                background: #0a0a0a;
                font-family: monospace;
                overflow: hidden;
            }}
            #canvas {{ 
                display: block;
                width: 100%;
                height: 100%;
            }}
            #overlay {{
                position: absolute;
                top: 0;
                left: 0;
                right: 0;
                bottom: 0;
                background: rgba(10, 10, 10, 0.95);
                display: flex;
                flex-direction: column;
                justify-content: center;
                align-items: center;
                color: #0f0;
                z-index: 1000;
                padding: 20px;
                text-align: center;
            }}
            .hidden {{ display: none !important; }}
            #info {{ 
                position: absolute;
                top: 10px;
                left: 10px;
                background: rgba(0,0,0,0.95);
                color: #0f0;
                padding: 15px;
                border-radius: 8px;
                font-size: 12px;
                z-index: 100;
                max-width: 300px;
                border: 1px solid rgba(0,255,0,0.3);
            }}
            .stat {{ margin: 6px 0; padding-bottom: 4px; border-bottom: 1px solid rgba(0,255,0,0.1); }}
            .label {{ color: #888; font-size: 9px; }}
            .value {{ color: #0f0; font-weight: bold; }}
            #controls {{
                position: absolute;
                top: 10px;
                right: 10px;
                background: rgba(0,0,0,0.95);
                color: #0f0;
                padding: 15px;
                border-radius: 8px;
                font-size: 11px;
                z-index: 100;
                border: 1px solid rgba(0,255,0,0.3);
            }}
            button {{
                background: rgba(0,255,0,0.15);
                border: 1px solid #0f0;
                color: #0f0;
                padding: 8px;
                border-radius: 4px;
                cursor: pointer;
                width: 100%;
                margin-top: 10px;
            }}
            button:hover {{ background: rgba(0,255,0,0.25); }}
            #debug {{ 
                position: absolute;
                bottom: 10px;
                left: 10px;
                background: rgba(0,0,0,0.95);
                color: #888;
                padding: 10px;
                border-radius: 4px;
                font-size: 10px;
                max-width: 400px;
                max-height: 200px;
                overflow-y: auto;
                border: 1px solid rgba(0,255,0,0.2);
            }}
            .debug-line {{ margin: 2px 0; }}
            .debug-ok {{ color: #0f0; }}
            .debug-wait {{ color: #ff0; }}
            .debug-err {{ color: #f00; }}
        </style>
    </head>
    <body>
        <canvas id="canvas"></canvas>
        
        <div id="overlay">
            <div style="font-size: 24px; margin-bottom: 20px;">⏳</div>
            <div style="font-size: 16px; margin-bottom: 20px;">Initializing 3D Viewer</div>
            <div id="overlay-status" style="font-size: 12px; color: #888;">Loading embedded libraries...</div>
        </div>
        
        <div id="info" class="hidden">
            <div style="text-align: center; margin-bottom: 8px; font-weight: bold; border-bottom: 1px solid rgba(0,255,0,0.2); padding-bottom: 6px;">
                🧠 3D Model
            </div>
            <div class="stat">
                <div class="label">Tumor Vertices</div>
                <div class="value">{attention_stats.get('vertex_count', 0) if attention_stats else 0:,}</div>
            </div>
            <div class="stat">
                <div class="label">Tumor Faces</div>
                <div class="value">{attention_stats.get('face_count', 0) if attention_stats else 0:,}</div>
            </div>
            <div class="stat">
                <div class="label">Attention Mean</div>
                <div class="value">{attention_stats.get('attention_mean', 0) if attention_stats else 0:.3f}</div>
            </div>
            <div class="stat">
                <div class="label">Attention Max</div>
                <div class="value">{attention_stats.get('attention_max', 0) if attention_stats else 0:.3f}</div>
            </div>
            <div class="stat" style="border-bottom: none;">
                <div class="label" style="margin-bottom: 4px;" id="colormap-mode-label">Attention Colormap (absolute 0&ndash;1)</div>
                <div style="height: 10px; border-radius: 3px; background: linear-gradient(to right,
                    rgb(0,0,255) 0%, rgb(0,255,255) 25%, rgb(0,255,0) 50%,
                    rgb(255,255,0) 75%, rgb(255,0,0) 100%);"></div>
                <div style="display: flex; justify-content: space-between; font-size: 8px; color: #888; margin-top: 2px;">
                    <span id="colormap-min-label">Low</span><span id="colormap-max-label">High</span>
                </div>
            </div>
        </div>
        
        <div id="controls" class="hidden">
            <strong>Controls</strong>
            <div style="margin-top: 8px; font-size: 10px; line-height: 1.6;">
                🖱️ Drag = Rotate<br>
                📜 Scroll = Zoom
            </div>
            <div style="margin-top: 10px; font-size: 11px; border-top: 1px solid rgba(0,255,0,0.2); padding-top: 8px;">
                <label style="display: block; cursor: pointer; margin-bottom: 4px;">
                    <input type="checkbox" checked onchange="if(window.toggleTumor) toggleTumor(this.checked)"> Tumor
                </label>
                {'<label style="display: block; cursor: pointer;"><input type="checkbox" checked onchange="if(window.toggleEdema) toggleEdema(this.checked)"> Edema</label>' if edema_gltf_b64 else ''}
            </div>
            <div style="margin-top: 8px; font-size: 11px; border-top: 1px solid rgba(0,255,0,0.2); padding-top: 8px;">
                <label style="display: block; cursor: pointer;" title="Stretch colors to this mesh's own min/max attention range, instead of the fixed 0-1 scale">
                    <input type="checkbox" onchange="if(window.toggleRelativeColormap) toggleRelativeColormap(this.checked)"> Relative colormap
                </label>
            </div>
            <button onclick="if(window.resetView) resetView();">↻ Reset</button>
        </div>
        
        <div id="debug">
            <div class="debug-line debug-wait">⏳ Initializing...</div>
        </div>
        
        <script>
            const debugDiv = document.getElementById('debug');
            const debugLines = [];
            
            function log(msg, type = 'normal') {{
                const timestamp = new Date().toLocaleTimeString();
                const fullMsg = `[${{timestamp}}] ${{msg}}`;
                debugLines.push({{'msg': fullMsg, 'type': type}});
                
                if (debugLines.length > 25) debugLines.shift();
                
                debugDiv.innerHTML = debugLines.map(l => 
                    `<div class="debug-line debug-${{l.type}}">${{l.msg}}</div>`
                ).join('');
                debugDiv.scrollTop = debugDiv.scrollHeight;
                
                console.log(fullMsg);
            }}
            
            log('🚀 Page loaded (base64 mode)');
            
            // Decode and execute base64-encoded libraries
            function decodeAndExecute(base64Str, name) {{
                try {{
                    log(`📦 Decoding ${{name}}...`);
                    const binaryString = atob(base64Str);
                    const bytes = new Uint8Array(binaryString.length);
                    for (let i = 0; i < binaryString.length; i++) {{
                        bytes[i] = binaryString.charCodeAt(i);
                    }}
                    const decoder = new TextDecoder();
                    const code = decoder.decode(bytes);
                    log(`✅ Decoded ${{name}}`, 'ok');
                    return code;
                }} catch (e) {{
                    log(`❌ Decode error for ${{name}}: ${{e.message}}`, 'err');
                    return null;
                }}
            }}
            
            // Decode libraries
            const threeCode = decodeAndExecute('{three_b64}', 'three.min.js');
            const loaderCode = decodeAndExecute('{loader_b64}', 'GLTFLoader.js');
            
            if (!threeCode || !loaderCode) {{
                log('❌ Failed to decode libraries', 'err');
            }} else {{
                // Execute libraries
                log('▶️ Executing three.min.js...');
                eval(threeCode);
                log('✅ three.min.js executed', 'ok');
                
                log('▶️ Executing GLTFLoader.js...');
                eval(loaderCode);
                log('✅ GLTFLoader.js executed', 'ok');
                
                // Initialize viewer
                setTimeout(() => initViewer(), 100);
            }}
            
            let scene, camera, renderer, group;
            
            function showError(msg) {{
                document.getElementById('overlay-status').textContent = msg;
                log('❌ ' + msg, 'err');
            }}
            
            function initViewer() {{
                try {{
                    log('🎬 Initializing viewer...');
                    
                    if (typeof THREE === 'undefined') {{
                        throw new Error('THREE not available after eval');
                    }}
                    
                    // Hide overlay
                    document.getElementById('overlay').classList.add('hidden');
                    document.getElementById('info').classList.remove('hidden');
                    document.getElementById('controls').classList.remove('hidden');
                    
                    const canvas = document.getElementById('canvas');
                    
                    scene = new THREE.Scene();
                    scene.background = new THREE.Color(0x0a0a0a);
                    log('✅ Scene created', 'ok');
                    
                    camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 1000);
                    camera.position.set(50, 50, 50);
                    camera.lookAt(0, 0, 0);
                    log('✅ Camera created', 'ok');
                    
                    renderer = new THREE.WebGLRenderer({{
                        canvas: canvas,
                        antialias: true,
                        alpha: false,
                        precision: 'lowp'
                    }});
                    renderer.setSize(window.innerWidth, window.innerHeight);
                    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 1.5));
                    log('✅ Renderer created', 'ok');
                    
                    group = new THREE.Group();
                    scene.add(group);
                    
                    scene.add(new THREE.AmbientLight(0xffffff, 0.8));
                    const dirLight = new THREE.DirectionalLight(0xffffff, 1.0);
                    dirLight.position.set(100, 100, 100);
                    scene.add(dirLight);
                    const dirLight2 = new THREE.DirectionalLight(0xffffff, 0.5);
                    dirLight2.position.set(-100, -50, -100);
                    scene.add(dirLight2);
                    log('✅ Lights added', 'ok');
                    
                    // Controls
                    let dragging = false;
                    let mouse = {{ x: 0, y: 0 }};
                    
                    canvas.addEventListener('mousedown', () => dragging = true);
                    canvas.addEventListener('mouseup', () => dragging = false);
                    canvas.addEventListener('mousemove', (e) => {{
                        if (dragging) {{
                            group.rotation.y += (e.clientX - mouse.x) * 0.01;
                            group.rotation.x += (e.clientY - mouse.y) * 0.01;
                        }}
                        mouse = {{ x: e.clientX, y: e.clientY }};
                    }});
                    
                    canvas.addEventListener('wheel', (e) => {{
                        e.preventDefault();
                        const dir = camera.position.clone().normalize();
                        const dist = camera.position.length();
                        const newDist = Math.max(10, Math.min(500, dist + e.deltaY * 0.2));
                        camera.position.copy(dir.multiplyScalar(newDist));
                    }});
                    
                    log('✅ Controls added', 'ok');
                    
                    // Load meshes (tumor + optional edema)
                    const loader = new THREE.GLTFLoader();
                    const hasEdema = {'true' if edema_gltf_b64 else 'false'};
                    const totalToLoad = hasEdema ? 2 : 1;
                    let loadedCount = 0;
                    let tumorModel = null;
                    let edemaModel = null;
                    
                    function fitCameraToAll() {{
                        const box = new THREE.Box3().setFromObject(group);
                        const size = box.getSize(new THREE.Vector3());
                        const center = box.getCenter(new THREE.Vector3());
                        const maxDim = Math.max(size.x, size.y, size.z);
                        const fov = camera.fov * Math.PI / 180;
                        const cameraDist = Math.abs(maxDim / 2 / Math.tan(fov / 2)) * 1.5;
                        
                        group.position.set(-center.x, -center.y, -center.z);
                        
                        camera.position.set(cameraDist * 0.7, cameraDist * 0.7, cameraDist);
                        camera.lookAt(0, 0, 0);
                        camera.near = Math.max(0.1, cameraDist / 100);
                        camera.far = cameraDist * 10;
                        camera.updateProjectionMatrix();
                        
                        window._defaultCamDist = cameraDist;
                        
                        log(`📐 Combined mesh size: ${{maxDim.toFixed(1)}}, cam dist: ${{cameraDist.toFixed(1)}}`, 'ok');
                    }}
                    
                    function onMeshLoaded() {{
                        loadedCount++;
                        if (loadedCount === totalToLoad) {{
                            fitCameraToAll();
                            log('✅ All meshes loaded and camera fitted', 'ok');
                        }}
                    }}
                    
                    function applyNormalsFix(model, label) {{
                        model.traverse((node) => {{
                            if (node.isMesh) {{
                                if (!node.geometry.attributes.normal) {{
                                    node.geometry.computeVertexNormals();
                                    log(`🔧 Computed vertex normals for ${{label}}`, 'ok');
                                }}
                            }}
                        }});
                    }}
                    
                    // --- Jet colormap forward/inverse (must match mesh_generator.py exactly) ---
                    function jetForward(v) {{
                        v = Math.max(0, Math.min(1, v));
                        let r, g, b;
                        if (v < 0.125) {{ r = 0; g = 0; b = (v / 0.125); }}
                        else if (v < 0.375) {{ r = 0; g = ((v - 0.125) / 0.25); b = 1; }}
                        else if (v < 0.625) {{ r = ((v - 0.375) / 0.25); g = 1; b = (1 - (v - 0.375) / 0.25); }}
                        else if (v < 0.875) {{ r = 1; g = (1 - (v - 0.625) / 0.25); b = 0; }}
                        else {{ r = (1 - (v - 0.875) / 0.125); g = 0; b = 0; }}
                        return [r, g, b];
                    }}
                    
                    function jetInverse(r, g, b) {{
                        // Recover approximate original 0-1 value from a jet-colormapped RGB.
                        // Matches the 5-segment piecewise structure of jetForward above.
                        //
                        // DEFENSIVE FIX: Three.js's GLTFLoader has been observed returning
                        // raw 0-255 component values from BufferAttribute.getX/Y/Z() instead
                        // of the normalized 0-1 floats implied by the accessor's
                        // "normalized: true" flag. Auto-detect and correct for this so the
                        // function works correctly regardless of which representation we get.
                        if (r > 1.5 || g > 1.5 || b > 1.5) {{
                            r = r / 255; g = g / 255; b = b / 255;
                        }}
                        
                        const eps = 0.02;
                        if (r <= eps && g <= eps) {{
                            return (b) * 0.125;
                        }} else if (r <= eps && b >= 1 - eps) {{
                            return 0.125 + g * 0.25;
                        }} else if (g >= 1 - eps && b > eps) {{
                            return 0.375 + r * 0.25;
                        }} else if (r >= 1 - eps && b <= eps) {{
                            return 0.625 + (1 - g) * 0.25;
                        }} else {{
                            return 0.875 + (1 - r) * 0.125;
                        }}
                    }}
                    
                    // Tumor mesh (jet colormap by attention, from COLOR_0 vertex data)
                    const tumorUrl = 'data:model/gltf-binary;base64,{tumor_gltf_b64}';
                    log('📦 Loading tumor mesh...', 'wait');
                    
                    loader.load(tumorUrl,
                        (gltf) => {{
                            tumorModel = gltf.scene;
                            applyNormalsFix(tumorModel, 'tumor');
                            tumorModel.traverse((node) => {{
                                if (node.isMesh) {{
                                    const hasVertexColors = !!node.geometry.attributes.color;
                                    
                                    if (hasVertexColors) {{
                                        log('🎨 Using attention-based vertex colors (jet colormap)', 'ok');
                                        node.material = new THREE.MeshPhongMaterial({{
                                            vertexColors: true,
                                            color: 0xffffff,
                                            emissive: 0x000000,
                                            shininess: 25,
                                            side: THREE.DoubleSide
                                        }});
                                        
                                        // Precompute attention proxy per vertex (inverse jet) and
                                        // keep a backup of the original absolute-scale colors so we
                                        // can restore them when the relative toggle is switched off.
                                        const colorAttr = node.geometry.attributes.color;
                                        const n = colorAttr.count;
                                        const proxy = new Float32Array(n);
                                        
                                        // Detect whether getX/Y/Z returns 0-1 normalized floats or
                                        // raw 0-255 values (observed to vary), so we can write colors
                                        // back in the same scale later.
                                        const sampleR = colorAttr.getX(0);
                                        const sampleG = colorAttr.getY(0);
                                        const sampleB = colorAttr.getZ(0);
                                        const colorsAreRaw255 = (sampleR > 1.5 || sampleG > 1.5 || sampleB > 1.5);
                                        node.userData.colorsAreRaw255 = colorsAreRaw255;
                                        log(`🔍 Color scale detected: ${{colorsAreRaw255 ? 'raw 0-255' : 'normalized 0-1'}}`, 'ok');
                                        
                                        for (let i = 0; i < n; i++) {{
                                            const r = colorAttr.getX(i);
                                            const g = colorAttr.getY(i);
                                            const b = colorAttr.getZ(i);
                                            proxy[i] = jetInverse(r, g, b);
                                        }}
                                        node.userData.attentionProxy = proxy;
                                        node.userData.originalColors = colorAttr.array.slice();
                                    }} else {{
                                        log('⚠️ No vertex colors found, using fallback red', 'wait');
                                        node.material = new THREE.MeshPhongMaterial({{
                                            color: 0xff6666,
                                            emissive: 0x330000,
                                            shininess: 40,
                                            side: THREE.DoubleSide
                                        }});
                                    }}
                                    node.userData.label = 'tumor';
                                }}
                            }});
                            group.add(tumorModel);
                            window._tumorModel = tumorModel;
                            log('✅ Tumor mesh loaded', 'ok');
                            onMeshLoaded();
                        }},
                        (progress) => {{
                            const pct = Math.round((progress.loaded / progress.total) * 100);
                            log(`📥 Tumor loading: ${{pct}}%`, 'wait');
                        }},
                        (error) => {{
                            log('❌ Tumor load error: ' + error.message, 'err');
                            showError('Tumor mesh load failed: ' + error.message);
                        }}
                    );
                    
                    // Edema mesh (yellow, semi-transparent)
                    if (hasEdema) {{
                        const edemaUrl = 'data:model/gltf-binary;base64,{edema_gltf_b64 or ""}';
                        log('📦 Loading edema mesh...', 'wait');
                        
                        loader.load(edemaUrl,
                            (gltf) => {{
                                edemaModel = gltf.scene;
                                applyNormalsFix(edemaModel, 'edema');
                                edemaModel.traverse((node) => {{
                                    if (node.isMesh) {{
                                        node.material = new THREE.MeshPhongMaterial({{
                                            color: 0xffee44,
                                            emissive: 0x332200,
                                            shininess: 20,
                                            side: THREE.DoubleSide,
                                            transparent: true,
                                            opacity: 0.35,
                                            depthWrite: false
                                        }});
                                        node.userData.label = 'edema';
                                    }}
                                }});
                                group.add(edemaModel);
                                window._edemaModel = edemaModel;
                                log('✅ Edema mesh loaded', 'ok');
                                onMeshLoaded();
                            }},
                            (progress) => {{
                                const pct = Math.round((progress.loaded / progress.total) * 100);
                                log(`📥 Edema loading: ${{pct}}%`, 'wait');
                            }},
                            (error) => {{
                                log('❌ Edema load error: ' + error.message, 'err');
                                // Don't block tumor display if edema fails
                                loadedCount++;
                                if (loadedCount === totalToLoad) fitCameraToAll();
                            }}
                        );
                    }}
                    
                    // Wire up visibility checkboxes
                    window.toggleTumor = (visible) => {{
                        if (window._tumorModel) window._tumorModel.visible = visible;
                    }};
                    window.toggleEdema = (visible) => {{
                        if (window._edemaModel) window._edemaModel.visible = visible;
                    }};
                    
                    window.toggleRelativeColormap = (useRelative) => {{
                        if (!window._tumorModel) return;
                        
                        window._tumorModel.traverse((node) => {{
                            if (!node.isMesh || !node.userData.attentionProxy) return;
                            
                            const colorAttr = node.geometry.attributes.color;
                            const proxy = node.userData.attentionProxy;
                            const n = proxy.length;
                            
                            if (useRelative) {{
                                let vmin = Infinity, vmax = -Infinity;
                                for (let i = 0; i < n; i++) {{
                                    if (proxy[i] < vmin) vmin = proxy[i];
                                    if (proxy[i] > vmax) vmax = proxy[i];
                                }}
                                const range = vmax - vmin;
                                
                                for (let i = 0; i < n; i++) {{
                                    const stretched = range > 1e-6 ? (proxy[i] - vmin) / range : 0.5;
                                    let [r, g, b] = jetForward(stretched);
                                    if (node.userData.colorsAreRaw255) {{
                                        r *= 255; g *= 255; b *= 255;
                                    }}
                                    colorAttr.setXYZ(i, r, g, b);
                                }}
                                colorAttr.needsUpdate = true;
                                
                                const label = document.getElementById('colormap-mode-label');
                                if (label) label.textContent = 'Attention Colormap (relative, stretched)';
                                const minLbl = document.getElementById('colormap-min-label');
                                const maxLbl = document.getElementById('colormap-max-label');
                                if (minLbl) minLbl.textContent = `Low (${{vmin.toFixed(3)}})`;
                                if (maxLbl) maxLbl.textContent = `High (${{vmax.toFixed(3)}})`;
                                
                                log(`🎨 Relative colormap: stretched [${{vmin.toFixed(3)}}, ${{vmax.toFixed(3)}}] -> [0,1]`, 'ok');
                            }} else {{
                                colorAttr.array.set(node.userData.originalColors);
                                colorAttr.needsUpdate = true;
                                
                                const label = document.getElementById('colormap-mode-label');
                                if (label) label.textContent = 'Attention Colormap (absolute 0\u20131)';
                                const minLbl = document.getElementById('colormap-min-label');
                                const maxLbl = document.getElementById('colormap-max-label');
                                if (minLbl) minLbl.textContent = 'Low';
                                if (maxLbl) maxLbl.textContent = 'High';
                                
                                log('🎨 Restored absolute (0-1) colormap', 'ok');
                            }}
                        }});
                    }};
                    
                    // Animation
                    function animate() {{
                        requestAnimationFrame(animate);
                        renderer.render(scene, camera);
                    }}
                    animate();
                    log('✅ Animation loop started', 'ok');
                    
                    // Resize
                    window.addEventListener('resize', () => {{
                        camera.aspect = window.innerWidth / window.innerHeight;
                        camera.updateProjectionMatrix();
                        renderer.setSize(window.innerWidth, window.innerHeight);
                    }});
                    
                    window.resetView = () => {{
                        const d = window._defaultCamDist || 50;
                        camera.position.set(d * 0.7, d * 0.7, d);
                        camera.lookAt(0, 0, 0);
                        group.rotation.set(0, 0, 0);
                    }};
                    
                }} catch (e) {{
                    log('❌ Fatal error: ' + e.message, 'err');
                    showError('Error: ' + e.message);
                }}
            }}
        </script>
    </body>
    </html>
    """
    
    st.components.v1.html(html_template, height=height + 50)
