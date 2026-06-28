"""Build the live SL demo HTML by injecting route data and new components."""
import json

# ── Read route data ──
with open('route_data.json', 'r') as f:
    route = json.load(f)

# Compact format for JS: [lon, lat, z, yaw, s]
compact = []
for pt in route:
    compact.append([round(pt[0],7), round(pt[1],7), round(pt[2],1), round(pt[3],3), round(pt[4],1)])
ROUTE_JS = json.dumps(compact, separators=(',',':'))

# ── New CSS for live demo ──
LIVE_CSS = """
/* ===== Live Demo - Status Bar ===== */
.live-status-bar {
  display: flex; gap: 0.5rem; flex-wrap: wrap;
  margin-bottom: 1.2rem;
}
.live-status-item {
  display: flex; align-items: center; gap: 0.45rem;
  background: var(--surface); border: 1px solid var(--border2);
  border-radius: 6px; padding: 0.35rem 0.7rem; font-size: 0.68rem;
  letter-spacing: 0.04em; color: var(--text2);
}
.live-status-item .sdot {
  width: 6px; height: 6px; border-radius: 50%; flex-shrink: 0;
}
.live-status-item .sdot.ok { background: var(--green); box-shadow: 0 0 6px var(--green); }
.live-status-item .sdot.warn { background: var(--orange); box-shadow: 0 0 6px var(--orange); animation: dotPulse 1s ease-in-out infinite; }
.live-status-item .sdot.err { background: var(--red); box-shadow: 0 0 6px var(--red); }

/* ===== Live Demo - Main Layout ===== */
.live-main { display: grid; grid-template-columns: 1.5fr 1fr; gap: 1rem; margin-bottom: 1rem; }

/* ===== Live Canvas ===== */
.live-canvas-wrap {
  background: var(--surface); border: 1px solid var(--border2);
  border-radius: 12px; padding: 0.6rem; position: relative; overflow: hidden;
}
.live-canvas-wrap canvas { width: 100%; height: 360px; display: block; border-radius: 8px; }
.live-canvas-overlay {
  position: absolute; top: 0.8rem; right: 0.8rem;
  font-size: 0.6rem; color: var(--text3); letter-spacing: 0.06em;
  background: rgba(8,8,26,0.8); padding: 0.25rem 0.5rem; border-radius: 4px;
  pointer-events: none;
}

/* ===== Sensor Data Panels ===== */
.sensor-panels { display: flex; flex-direction: column; gap: 0.55rem; }
.sensor-panel {
  background: var(--surface); border: 1px solid var(--border2);
  border-radius: 8px; padding: 0.65rem 0.8rem;
  transition: all 0.3s; position: relative; overflow: hidden;
}
.sensor-panel::after {
  content: ''; position: absolute; top: 0; right: 0; width: 40px; height: 100%;
  background: linear-gradient(90deg, transparent, rgba(0,255,255,0.02));
  pointer-events: none;
}
.sensor-panel .sp-header {
  display: flex; align-items: center; gap: 0.4rem; margin-bottom: 0.35rem;
}
.sensor-panel .sp-header .sp-icon { font-size: 0.8rem; }
.sensor-panel .sp-header .sp-name { font-size: 0.62rem; color: var(--text3); letter-spacing: 0.06em; }
.sensor-panel .sp-header .sp-topic { font-size: 0.58rem; color: var(--text3); opacity: 0.5; margin-left: auto; }
.sensor-panel .sp-values { display: grid; grid-template-columns: 1fr 1fr; gap: 0.15rem 0.5rem; }
.sensor-panel .sp-values .spv { font-size: 0.7rem; color: var(--text); }
.sensor-panel .sp-values .spv .spv-label { font-size: 0.58rem; color: var(--text3); display: block; }
.sensor-panel .sp-values .spv.highlight { color: var(--cyan); font-weight: 600; }
.sensor-panel.flash { border-color: rgba(0,255,255,0.5); box-shadow: 0 0 12px rgba(0,255,255,0.08); }

/* ===== Playback Controls ===== */
.playback-bar {
  display: flex; align-items: center; gap: 0.6rem;
  background: var(--surface); border: 1px solid var(--border2);
  border-radius: 10px; padding: 0.5rem 1rem;
  flex-wrap: wrap;
}
.playback-bar button {
  background: rgba(0,255,255,0.06); border: 1px solid rgba(0,200,200,0.2);
  color: var(--cyan); width: 32px; height: 32px; border-radius: 6px;
  cursor: pointer; font-size: 0.75rem; transition: all 0.2s;
  display: flex; align-items: center; justify-content: center;
}
.playback-bar button:hover { background: rgba(0,255,255,0.14); border-color: var(--cyan); box-shadow: 0 0 10px rgba(0,255,255,0.15); }
.playback-bar button:disabled { opacity: 0.2; cursor: not-allowed; }
.playback-bar button.active-speed { background: rgba(0,255,255,0.15); border-color: var(--cyan); color: #fff; }
.playback-bar .pb-progress {
  flex: 1; min-width: 100px; height: 4px; border-radius: 2px;
  background: rgba(255,255,255,0.06); cursor: pointer; position: relative;
  -webkit-appearance: none; appearance: none; outline: none;
}
.playback-bar .pb-progress::-webkit-slider-thumb {
  -webkit-appearance: none; width: 14px; height: 14px; border-radius: 50%;
  background: var(--cyan); cursor: pointer; box-shadow: 0 0 10px rgba(0,255,255,0.5);
}
.playback-bar .pb-time {
  font-size: 0.65rem; color: var(--text3); letter-spacing: 0.04em;
  white-space: nowrap; min-width: 90px; text-align: center;
}
.playback-bar .pb-sep { width: 1px; height: 16px; background: var(--border2); }

/* ===== ROS Terminal ===== */
.ros-term-wrap { margin-top: 0.8rem; }
.ros-term-header {
  display: flex; align-items: center; gap: 0.5rem;
  font-size: 0.65rem; color: var(--text3); letter-spacing: 0.06em;
  margin-bottom: 0.3rem; cursor: pointer; user-select: none;
}
.ros-term-header .toggle { font-size: 0.6rem; transition: transform 0.3s; }
.ros-term-header .toggle.open { transform: rotate(90deg); }
.ros-terminal {
  background: rgba(4,4,18,0.95); border: 1px solid rgba(0,255,255,0.1);
  border-radius: 6px; padding: 0.5rem 0.7rem; height: 110px; overflow-y: auto;
  font-family: 'Cascadia Code', 'Fira Code', 'Consolas', 'Courier New', monospace;
  font-size: 0.62rem; line-height: 1.55; color: rgba(0,255,200,0.7);
  scroll-behavior: smooth;
}
.ros-terminal .log-line { white-space: nowrap; }
.ros-terminal .log-line .ts { color: rgba(255,255,255,0.25); margin-right: 0.5rem; }
.ros-terminal .log-line .node { color: var(--cyan); }
.ros-terminal .log-line .val { color: rgba(255,255,255,0.65); }
.ros-terminal .log-line.warn { color: var(--orange); }
.ros-terminal::-webkit-scrollbar { width: 3px; }
.ros-terminal::-webkit-scrollbar-thumb { background: rgba(0,255,255,0.15); border-radius: 2px; }

/* Responsive */
@media (max-width: 768px) {
  .live-main { grid-template-columns: 1fr; }
  .live-canvas-wrap canvas { height: 240px; }
  .playback-bar { gap: 0.3rem; padding: 0.4rem 0.5rem; }
  .playback-bar button { width: 26px; height: 26px; font-size: 0.65rem; }
  .sensor-panel .sp-values { grid-template-columns: 1fr; }
}
"""

# ── New Screen 2 HTML ──
SCREEN2_HTML = """
    <!-- ================================================ -->
    <!--  SCREEN 2 — LIVE DEMO · 实测回放                  -->
    <!-- ================================================ -->
    <section class="screen" id="trajectory">
    <div class="screen-inner">
      <div class="sec-badge">LIVE DEMO · 实测回放</div>
      <div class="sec-title">塘朗山轨迹<span>实时回放</span></div>
      <div class="sec-desc">基于 ROS 节点真实数据格式，回放 demo_route.csv 中 1388 帧轨迹数据。模拟 nmea_gps_node / bme280_altitude / vins_estimator / route_matcher 四个节点的实时消息输出。</div>

      <!-- Live Status Bar -->
      <div class="live-status-bar" id="liveStatusBar">
        <div class="live-status-item"><span class="sdot ok" id="sdGPS"></span> GPS <span id="stGPS">3D_FIX</span></div>
        <div class="live-status-item"><span class="sdot ok" id="sdBME"></span> BME280 <span id="stBME">OK</span></div>
        <div class="live-status-item"><span class="sdot ok" id="sdVIO"></span> VIO <span id="stVIO">TRACKING</span></div>
        <div class="live-status-item"><span class="sdot ok" id="sdMATCH"></span> MATCH <span id="stMATCH">LOCKED</span></div>
        <div class="live-status-item"><span class="sdot ok"></span> DK-2500 <span style="color:var(--green);">ONLINE</span></div>
      </div>

      <!-- Main: Canvas + Sensor Panels -->
      <div class="live-main">
        <div>
          <div class="live-canvas-wrap" id="liveCanvasWrap">
            <canvas id="liveTrajCanvas"></canvas>
            <div class="live-canvas-overlay" id="canvasOverlay">FRAME 0 / 1388</div>
          </div>
        </div>
        <div class="sensor-panels" id="sensorPanels">
          <div class="sensor-panel" id="spGPS">
            <div class="sp-header"><span class="sp-icon">🛰️</span><span class="sp-name">GPS / NavSatFix</span><span class="sp-topic">/gps</span></div>
            <div class="sp-values">
              <div class="spv highlight"><span class="spv-label">Latitude</span><span id="vLat">22.5300000°</span></div>
              <div class="spv highlight"><span class="spv-label">Longitude</span><span id="vLon">113.9532000°</span></div>
              <div class="spv"><span class="spv-label">Altitude</span><span id="vAlt">120.0 m</span></div>
              <div class="spv"><span class="spv-label">Fix Quality</span><span id="vFix">3D (1)</span></div>
            </div>
          </div>
          <div class="sensor-panel" id="spBME">
            <div class="sp-header"><span class="sp-icon">🌡️</span><span class="sp-name">BME280 Altitude</span><span class="sp-topic">/altitude</span></div>
            <div class="sp-values">
              <div class="spv highlight"><span class="spv-label">Altitude</span><span id="vBmeAlt">120.00 m</span></div>
              <div class="spv"><span class="spv-label">Pressure</span><span id="vBmePress">1013.25 hPa</span></div>
              <div class="spv"><span class="spv-label">Temperature</span><span id="vBmeTemp">28.5 °C</span></div>
              <div class="spv"><span class="spv-label">Humidity</span><span id="vBmeHum">62.3 %</span></div>
            </div>
          </div>
          <div class="sensor-panel" id="spVIO">
            <div class="sp-header"><span class="sp-icon">📐</span><span class="sp-name">VIO Pose (VINS-Fusion)</span><span class="sp-topic">/vins_estimator/imu_propagate</span></div>
            <div class="sp-values">
              <div class="spv"><span class="spv-label">Position X</span><span id="vVioX">0.00 m</span></div>
              <div class="spv"><span class="spv-label">Position Y</span><span id="vVioY">0.00 m</span></div>
              <div class="spv"><span class="spv-label">Position Z</span><span id="vVioZ">0.00 m</span></div>
              <div class="spv highlight"><span class="spv-label">Yaw</span><span id="vVioYaw">0.5764 rad</span></div>
            </div>
          </div>
          <div class="sensor-panel" id="spMatch">
            <div class="sp-header"><span class="sp-icon">🗺️</span><span class="sp-name">Route Matcher</span><span class="sp-topic">/route_match</span></div>
            <div class="sp-values">
              <div class="spv highlight"><span class="spv-label">Segment ID</span><span id="vSegId">0</span></div>
              <div class="spv highlight"><span class="spv-label">Confidence</span><span id="vConf">0.0%</span></div>
              <div class="spv"><span class="spv-label">Distance</span><span id="vDist">0.00 m</span></div>
              <div class="spv"><span class="spv-label">Matched Nodes</span><span id="vNodes">0</span></div>
            </div>
          </div>
        </div>
      </div>

      <!-- Playback Controls -->
      <div class="playback-bar" id="playbackBar">
        <button id="btnReset" title="Reset">⏮</button>
        <button id="btnPlay" title="Play/Pause">▶</button>
        <button id="btnStepB" title="Step Back">◀</button>
        <button id="btnStepF" title="Step Forward">▶</button>
        <span class="pb-sep"></span>
        <button id="btnSpd05" data-speed="0.5">0.5×</button>
        <button id="btnSpd1" data-speed="1" class="active-speed">1×</button>
        <button id="btnSpd2" data-speed="2">2×</button>
        <button id="btnSpd5" data-speed="5">5×</button>
        <span class="pb-sep"></span>
        <input type="range" class="pb-progress" id="pbSlider" min="0" max="1387" value="0" title="Progress">
        <span class="pb-time" id="pbTime">0 / 1388 · 0.0%</span>
      </div>

      <!-- ROS Terminal -->
      <div class="ros-term-wrap">
        <div class="ros-term-header" id="rosTermToggle">
          <span class="toggle open" id="rosToggleIcon">▶</span> ROS Terminal &nbsp;<span style="color:var(--text3);font-size:0.58rem;">(simulated node messages)</span>
          <span style="margin-left:auto;font-size:0.58rem;color:var(--text3);" id="rosLineCount">0 lines</span>
        </div>
        <div class="ros-terminal" id="rosTerminal"></div>
      </div>

      <!-- Legend -->
      <div class="map-legend" style="margin-top:0.8rem;">
        <span><span class="leg-dot vio"></span> VIO 参考轨迹</span>
        <span><span class="leg-dot gps"></span> GPS 采样点</span>
        <span><span class="leg-dot match"></span> 实时定位</span>
        <span style="margin-left:1rem;color:var(--text3);">坐标: 113.95°E 22.53°N · 塘朗山 · 1388帧</span>
      </div>
    </div>
    </section>
"""

# ── New JS for live demo ──
LIVE_JS = f"""
// ===== Live SL Demo - Route Data =====
const ROUTE_DATA = {ROUTE_JS};
const TOTAL_FRAMES = ROUTE_DATA.length;

// ===== Live Demo State =====
const LS = {{
    playing: false,
    speed: 1.0,
    index: 0,
    animId: null,
    lastTs: 0,
    terminalOpen: true,
    logLines: [],
    // Simulated sensor status
    gpsFixQuality: '3D',  // cycles: 3D -> 2D -> 3D -> NO_FIX -> 3D
    gpsDropoutTimer: 0,
    bmeHealth: 1.0,
    vioHealth: 1.0,
    matchHealth: 1.0,
}};
// Always get safe integer index
function idx() {{ return Math.max(0, Math.min(TOTAL_FRAMES - 1, Math.floor(LS.index))); }}

// ===== DOM refs =====
let $$ = {{}};
function cacheDom() {{
    $$.canvas = document.getElementById('liveTrajCanvas');
    $$.ctx = $$.canvas.getContext('2d');
    $$.overlay = document.getElementById('canvasOverlay');
    $$.btnPlay = document.getElementById('btnPlay');
    $$.btnReset = document.getElementById('btnReset');
    $$.btnStepB = document.getElementById('btnStepB');
    $$.btnStepF = document.getElementById('btnStepF');
    $$.slider = document.getElementById('pbSlider');
    $$.pbTime = document.getElementById('pbTime');
    $$.rosTerm = document.getElementById('rosTerminal');
    $$.rosToggle = document.getElementById('rosTermToggle');
    $$.rosIcon = document.getElementById('rosToggleIcon');
    $$.rosCount = document.getElementById('rosLineCount');
    // Sensor panels
    $$.vLat = document.getElementById('vLat');
    $$.vLon = document.getElementById('vLon');
    $$.vAlt = document.getElementById('vAlt');
    $$.vFix = document.getElementById('vFix');
    $$.vBmeAlt = document.getElementById('vBmeAlt');
    $$.vBmePress = document.getElementById('vBmePress');
    $$.vBmeTemp = document.getElementById('vBmeTemp');
    $$.vBmeHum = document.getElementById('vBmeHum');
    $$.vVioX = document.getElementById('vVioX');
    $$.vVioY = document.getElementById('vVioY');
    $$.vVioZ = document.getElementById('vVioZ');
    $$.vVioYaw = document.getElementById('vVioYaw');
    $$.vSegId = document.getElementById('vSegId');
    $$.vConf = document.getElementById('vConf');
    $$.vDist = document.getElementById('vDist');
    $$.vNodes = document.getElementById('vNodes');
    // Status dots
    $$.sdGPS = document.getElementById('sdGPS');
    $$.sdBME = document.getElementById('sdBME');
    $$.sdVIO = document.getElementById('sdVIO');
    $$.sdMATCH = document.getElementById('sdMATCH');
    $$.stGPS = document.getElementById('stGPS');
    $$.stBME = document.getElementById('stBME');
    $$.stVIO = document.getElementById('stVIO');
    $$.stMATCH = document.getElementById('stMATCH');
    // Panels for flash effect
    $$.spGPS = document.getElementById('spGPS');
    $$.spBME = document.getElementById('spBME');
    $$.spVIO = document.getElementById('spVIO');
    $$.spMatch = document.getElementById('spMatch');
}}

// ===== Canvas Setup =====
let cW, cH, dpr;
function resizeCanvas() {{
    if (!$$.canvas) return;
    dpr = devicePixelRatio || 1;
    const r = $$.canvas.parentElement.getBoundingClientRect();
    cW = r.width;
    cH = 360;
    $$.canvas.width = cW * dpr;
    $$.canvas.height = cH * dpr;
    $$.canvas.style.width = cW + 'px';
    $$.canvas.style.height = cH + 'px';
    $$.ctx.setTransform(1, 0, 0, 1, 0, 0);
    $$.ctx.scale(dpr, dpr);
}}

// ===== Coordinate mapping =====
// Data range: lon 113.9532~113.9646, lat 22.53~22.5360
const DATA_BOUNDS = {{
    lonMin: 113.9532, lonMax: 113.9646,
    latMin: 22.53, latMax: 22.5360,
}};
function lonLatToXY(lon, lat) {{
    const pad = 30;
    const x = pad + ((lon - DATA_BOUNDS.lonMin) / (DATA_BOUNDS.lonMax - DATA_BOUNDS.lonMin)) * (cW - pad * 2);
    const y = cH - pad - ((lat - DATA_BOUNDS.latMin) / (DATA_BOUNDS.latMax - DATA_BOUNDS.latMin)) * (cH - pad * 2);
    return {{ x, y }};
}}

// ===== Draw Canvas =====
function drawCanvas() {{
    const ctx = $$.ctx;
    ctx.clearRect(0, 0, cW, cH);

    // Grid
    ctx.strokeStyle = 'rgba(0,255,255,0.03)';
    ctx.lineWidth = 0.5;
    for (let gx = 0; gx < cW; gx += 40) {{ ctx.beginPath(); ctx.moveTo(gx, 0); ctx.lineTo(gx, cH); ctx.stroke(); }}
    for (let gy = 0; gy < cH; gy += 40) {{ ctx.beginPath(); ctx.moveTo(0, gy); ctx.lineTo(cW, gy); ctx.stroke(); }}

    // Full reference path (faint)
    ctx.beginPath();
    let first = lonLatToXY(ROUTE_DATA[0][0], ROUTE_DATA[0][1]);
    ctx.moveTo(first.x, first.y);
    for (let i = 1; i < TOTAL_FRAMES; i++) {{
        const p = lonLatToXY(ROUTE_DATA[i][0], ROUTE_DATA[i][1]);
        ctx.lineTo(p.x, p.y);
    }}
    ctx.strokeStyle = 'rgba(91,141,239,0.2)';
    ctx.lineWidth = 1.5;
    ctx.setLineDash([4, 6]);
    ctx.stroke();
    ctx.setLineDash([]);

    // GPS sample points (every 15th point)
    for (let i = 0; i <= idx(); i += 15) {{
        if (i >= TOTAL_FRAMES) break;
        const p = lonLatToXY(ROUTE_DATA[i][0], ROUTE_DATA[i][1]);
        ctx.beginPath();
        ctx.arc(p.x, p.y, 2.2, 0, Math.PI * 2);
        ctx.fillStyle = 'rgba(255,140,0,0.5)';
        ctx.fill();
    }}

    // Traveled path (bright cyan)
    if (idx() > 0) {{
        ctx.beginPath();
        const p0 = lonLatToXY(ROUTE_DATA[0][0], ROUTE_DATA[0][1]);
        ctx.moveTo(p0.x, p0.y);
        for (let i = 1; i <= idx(); i++) {{
            const p = lonLatToXY(ROUTE_DATA[i][0], ROUTE_DATA[i][1]);
            ctx.lineTo(p.x, p.y);
        }}
        ctx.strokeStyle = 'rgba(0,255,255,0.7)';
        ctx.lineWidth = 2.5;
        ctx.stroke();
        // Glow
        ctx.strokeStyle = 'rgba(0,255,255,0.12)';
        ctx.lineWidth = 8;
        ctx.stroke();
    }}

    // Current position dot
    if (idx() < TOTAL_FRAMES) {{
        const pt = ROUTE_DATA[idx()];
        const cp = lonLatToXY(pt[0], pt[1]);
        const pulse = 1 + 0.35 * Math.sin(performance.now() * 0.005);
        // Outer glow
        ctx.beginPath();
        ctx.arc(cp.x, cp.y, 14 * pulse, 0, Math.PI * 2);
        ctx.fillStyle = 'rgba(0,255,255,' + (0.08 * pulse) + ')';
        ctx.fill();
        // Inner dot
        ctx.beginPath();
        ctx.arc(cp.x, cp.y, 5, 0, Math.PI * 2);
        ctx.fillStyle = 'rgba(0,255,240,0.9)';
        ctx.fill();
        ctx.strokeStyle = 'rgba(255,255,255,0.7)';
        ctx.lineWidth = 1.5;
        ctx.stroke();
    }}

    // Direction arrow at current position (every 3rd frame to not overwhelm)
    if (idx() < TOTAL_FRAMES && idx() % 3 === 0) {{
        const pt = ROUTE_DATA[idx()];
        const cp = lonLatToXY(pt[0], pt[1]);
        const yaw = pt[3];
        const arrowLen = 16;
        const ax = cp.x + Math.cos(yaw) * arrowLen;
        const ay = cp.y - Math.sin(yaw) * arrowLen;
        ctx.beginPath();
        ctx.moveTo(cp.x, cp.y);
        ctx.lineTo(ax, ay);
        ctx.strokeStyle = 'rgba(0,255,200,0.5)';
        ctx.lineWidth = 2;
        ctx.stroke();
        // arrowhead
        const hx1 = ax - Math.cos(yaw + 2.5) * 5;
        const hy1 = ay + Math.sin(yaw + 2.5) * 5;
        const hx2 = ax - Math.cos(yaw - 2.5) * 5;
        const hy2 = ay + Math.sin(yaw - 2.5) * 5;
        ctx.beginPath();
        ctx.moveTo(ax, ay);
        ctx.lineTo(hx1, hy1);
        ctx.lineTo(hx2, hy2);
        ctx.closePath();
        ctx.fillStyle = 'rgba(0,255,200,0.4)';
        ctx.fill();
    }}

    // Overlay text
    $$.overlay.textContent = 'FRAME ' + idx() + ' / ' + TOTAL_FRAMES;
}}

// ===== Update Sensor Panels =====
function updateSensors() {{
    const pt = ROUTE_DATA[idx()];
    const lon = pt[0], lat = pt[1], alt = pt[2], yaw = pt[3], dist = pt[4];

    // Simulate GPS noise and occasional dropout
    const gpsNoise = 0.000002;
    const gpsLat = lat + (Math.random() - 0.5) * gpsNoise * 2;
    const gpsLon = lon + (Math.random() - 0.5) * gpsNoise * 2;
    const gpsAlt = alt + (Math.random() - 0.5) * 3;

    // GPS dropout simulation (5% chance when near index 400-600, like tunnel/forest)
    let fixQuality, fixLabel;
    LS.gpsDropoutTimer--;
    if (LS.gpsDropoutTimer <= 0 && Math.random() < 0.003) {{
        LS.gpsDropoutTimer = 20 + Math.floor(Math.random() * 60);
    }}
    if (LS.gpsDropoutTimer > 0) {{
        fixQuality = 'NO_FIX';
        fixLabel = 'NO_FIX (0)';
    }} else {{
        const r = Math.random();
        fixQuality = r < 0.85 ? '3D' : (r < 0.97 ? '2D' : '3D');
        fixLabel = fixQuality === '3D' ? '3D (1)' : '2D (2)';
    }}

    // Update GPS panel
    $$.vLat.textContent = gpsLat.toFixed(7) + '°';
    $$.vLon.textContent = gpsLon.toFixed(7) + '°';
    $$.vAlt.textContent = gpsAlt.toFixed(1) + ' m';
    $$.vFix.textContent = fixLabel;

    // GPS status
    if (fixQuality === 'NO_FIX') {{
        $$.sdGPS.className = 'sdot err';
        $$.stGPS.textContent = 'NO_FIX';
    }} else if (fixQuality === '2D') {{
        $$.sdGPS.className = 'sdot warn';
        $$.stGPS.textContent = '2D_FIX';
    }} else {{
        $$.sdGPS.className = 'sdot ok';
        $$.stGPS.textContent = '3D_FIX';
    }}

    // BME280 panel (with realistic variations)
    const basePress = 1013.25 - (alt - 120) * 0.12;
    const pressure = basePress + (Math.random() - 0.5) * 0.3;
    const temp = 28.5 - (alt - 120) * 0.006 + (Math.random() - 0.5) * 0.3;
    const humidity = 62.3 + (Math.random() - 0.5) * 2;
    $$.vBmeAlt.textContent = alt.toFixed(2) + ' m';
    $$.vBmePress.textContent = pressure.toFixed(2) + ' hPa';
    $$.vBmeTemp.textContent = temp.toFixed(1) + ' °C';
    $$.vBmeHum.textContent = humidity.toFixed(1) + ' %';

    // VIO panel (local coordinates from CSV x/y/z columns would be ideal, but we only have lon/lat/z/yaw in compact format)
    // Use distance-based local coords
    const vioX = dist * Math.cos(yaw);
    const vioY = dist * Math.sin(yaw);
    const vioZ = alt - 120;
    $$.vVioX.textContent = vioX.toFixed(2) + ' m';
    $$.vVioY.textContent = vioY.toFixed(2) + ' m';
    $$.vVioZ.textContent = vioZ.toFixed(2) + ' m';
    $$.vVioYaw.textContent = yaw.toFixed(4) + ' rad';

    // Route Matcher panel
    const segId = Math.max(0, Math.floor(dist / 25));
    const conf = 0.6 + 0.35 * (1 - Math.exp(-dist / 500)) + (Math.random() - 0.5) * 0.06;
    const confClamped = Math.min(0.99, Math.max(0.4, conf));
    const matchNodes = Math.max(1, Math.floor(dist / 50));
    $$.vSegId.textContent = segId;
    $$.vConf.textContent = (confClamped * 100).toFixed(1) + '%';
    $$.vDist.textContent = dist.toFixed(2) + ' m';
    $$.vNodes.textContent = matchNodes;

    // Status indicators for others
    $$.sdBME.className = 'sdot ok'; $$.stBME.textContent = 'OK';
    $$.sdVIO.className = 'sdot ok'; $$.stVIO.textContent = 'TRACKING';
    if (confClamped > 0.8) {{
        $$.sdMATCH.className = 'sdot ok'; $$.stMATCH.textContent = 'LOCKED';
    }} else if (confClamped > 0.5) {{
        $$.sdMATCH.className = 'sdot warn'; $$.stMATCH.textContent = 'ACQUIRING';
    }} else {{
        $$.sdMATCH.className = 'sdot err'; $$.stMATCH.textContent = 'SEARCHING';
    }}

    // Flash effect on panels (random, 3% chance)
    if (Math.random() < 0.03) {{
        const panels = [$$.spGPS, $$.spBME, $$.spVIO, $$.spMatch];
        const p = panels[Math.floor(Math.random() * panels.length)];
        p.classList.add('flash');
        setTimeout(() => p.classList.remove('flash'), 200);
    }}
}}

// ===== ROS Terminal =====
function formatTime() {{
    const d = new Date();
    return d.toTimeString().slice(0,8) + '.' + String(d.getMilliseconds()).padStart(3,'0');
}}

function addLogLine(node, msg, cls) {{
    const ts = formatTime();
    LS.logLines.push({{ ts, node, msg, cls: cls || '' }});
    if (LS.logLines.length > 500) LS.logLines.shift();

    if (LS.terminalOpen) {{
        const div = document.createElement('div');
        div.className = 'log-line ' + (cls || '');
        div.innerHTML = '<span class="ts">[' + ts + ']</span><span class="node">[' + node + ']</span> <span class="val">' + msg + '</span>';
        $$.rosTerm.appendChild(div);
        // Keep max 200 DOM elements
        while ($$.rosTerm.children.length > 200) $$.rosTerm.removeChild($$.rosTerm.firstChild);
        $$.rosTerm.scrollTop = $$.rosTerm.scrollHeight;
    }}
    $$.rosCount.textContent = LS.logLines.length + ' lines';
}}

function generateRosMessages() {{
    const pt = ROUTE_DATA[idx()];
    const lon = pt[0], lat = pt[1], alt = pt[2], yaw = pt[3], dist = pt[4];
    const segId = Math.max(0, Math.floor(dist / 25));
    const conf = Math.min(0.99, 0.6 + 0.35 * (1 - Math.exp(-dist / 500)));

    // Only log every few frames based on speed
    const logInterval = LS.speed >= 5 ? 8 : (LS.speed >= 2 ? 4 : 1);
    if (LS.index % logInterval !== 0 && LS.index > 0) return;

    addLogLine('nmea_gps_node',
        'NavSatFix published → lat:' + lat.toFixed(7) + ' lon:' + lon.toFixed(7) + ' alt:' + alt.toFixed(1) + 'm cov[0]:' + (0.8 + Math.random() * 0.4).toFixed(2));

    if (LS.index % 2 === 0) {{
        addLogLine('bme280_altitude',
            'altitude=' + alt.toFixed(2) + 'm press=' + (1013.25 - (alt-120)*0.12).toFixed(2) + 'hPa temp=' + (28.5 - (alt-120)*0.006).toFixed(1) + '°C');
    }}

    if (LS.index % 3 === 0) {{
        addLogLine('vins_estimator',
            'VIO pose → x:' + (dist*Math.cos(yaw)).toFixed(2) + ' y:' + (dist*Math.sin(yaw)).toFixed(2) + ' z:' + (alt-120).toFixed(2) + ' yaw:' + yaw.toFixed(4) + ' feature_pts:' + (80 + Math.floor(Math.random()*40)));
    }}

    if (LS.index % 5 === 0) {{
        const matchMsg = conf > 0.8
            ? 'MATCH LOCKED → seg:' + segId + ' conf:' + (conf*100).toFixed(1) + '% dist:' + dist.toFixed(1) + 'm ✓'
            : 'SEARCHING → best seg:' + segId + ' conf:' + (conf*100).toFixed(1) + '%';
        addLogLine('route_matcher', matchMsg, conf < 0.5 ? 'warn' : '');
    }}

    if (LS.index % 20 === 0) {{
        addLogLine('sys_monitor',
            'DK-2500 ● CPU:38°C MEM:' + (3.2 + Math.random()*1.5).toFixed(1) + 'G/' + (7.8 + Math.random()*0.3).toFixed(1) + 'G NPU:idle POWER:' + (8.5 + Math.random()*2).toFixed(1) + 'W');
    }}
}}

// ===== Playback Logic =====
function setSpeed(s) {{
    LS.speed = s;
    document.querySelectorAll('[data-speed]').forEach(b => b.classList.remove('active-speed'));
    const btn = document.querySelector('[data-speed=\"' + s + '\"]');
    if (btn) btn.classList.add('active-speed');
}}

function seekTo(idx) {{
    LS.index = Math.max(0, Math.min(TOTAL_FRAMES - 1, Math.floor(idx)));
    $$.slider.value = LS.index;
    updateAll();
}}

function updateAll() {{
    drawCanvas();
    updateSensors();
    if (LS.playing) generateRosMessages();
    const pct = TOTAL_FRAMES > 1 ? (idx() / (TOTAL_FRAMES - 1) * 100) : 0;
    $$.pbTime.textContent = idx() + ' / ' + TOTAL_FRAMES + ' · ' + pct.toFixed(1) + '%';
    $$.slider.value = idx();
}}

function play() {{
    if (LS.playing) return;
    LS.playing = true;
    $$.btnPlay.textContent = '⏸';
    LS.lastTs = performance.now();
    // Frames per render: at 1x speed, each frame = 1 meter of travel, total 1386m
    // We'll advance roughly 1 data point per 16ms at 1x
    function tick(ts) {{
        if (!LS.playing) return;
        const dt = ts - LS.lastTs;
        LS.lastTs = ts;
        // At 1x: advance ~1 point per 16ms (60fps → covers 1388pts in ~22s)
        // At 2x: ~2 pts per 16ms, etc.
        const advance = (dt / 16) * LS.speed;
        LS.index += advance;
        if (LS.index >= TOTAL_FRAMES - 1) {{
            LS.index = TOTAL_FRAMES - 1;
            pause();
        }}
        LS.index = Math.min(TOTAL_FRAMES - 1, LS.index);
        updateAll();
        LS.animId = requestAnimationFrame(tick);
    }}
    LS.animId = requestAnimationFrame(tick);
}}

function pause() {{
    LS.playing = false;
    $$.btnPlay.textContent = '▶';
    if (LS.animId) {{ cancelAnimationFrame(LS.animId); LS.animId = null; }}
}}

function reset() {{
    pause();
    LS.index = 0;
    LS.gpsDropoutTimer = 0;
    LS.logLines = [];
    $$.rosTerm.innerHTML = '';
    updateAll();
    addLogLine('sys_monitor', '========== DEMO RESET ==========');
    addLogLine('sys_monitor', 'Loading demo_route.csv: ' + TOTAL_FRAMES + ' frames from 塘朗山 (Shenzhen)');
    addLogLine('sys_monitor', 'ROS nodes initialized: nmea_gps_node, bme280_altitude, vins_estimator, route_matcher');
    addLogLine('sys_monitor', 'DK-2500 platform ready. Press PLAY to start replay.');
}}

// ===== Event Handlers =====
function setupEvents() {{
    $$.btnPlay.addEventListener('click', () => LS.playing ? pause() : play());
    $$.btnReset.addEventListener('click', reset);
    $$.btnStepF.addEventListener('click', () => seekTo(LS.index + 1));
    $$.btnStepB.addEventListener('click', () => seekTo(LS.index - 1));

    document.querySelectorAll('[data-speed]').forEach(b => {{
        b.addEventListener('click', () => setSpeed(parseFloat(b.dataset.speed)));
    }});

    $$.slider.addEventListener('input', () => seekTo(parseInt($$.slider.value)));

    // ROS terminal toggle
    $$.rosToggle.addEventListener('click', () => {{
        LS.terminalOpen = !LS.terminalOpen;
        $$.rosIcon.classList.toggle('open', LS.terminalOpen);
        $$.rosTerm.style.display = LS.terminalOpen ? 'block' : 'none';
        if (LS.terminalOpen) {{
            // Rebuild from logLines
            $$.rosTerm.innerHTML = '';
            LS.logLines.slice(-150).forEach(l => {{
                const div = document.createElement('div');
                div.className = 'log-line ' + (l.cls || '');
                div.innerHTML = '<span class="ts">[' + l.ts + ']</span><span class="node">[' + l.node + ']</span> <span class="val">' + l.msg + '</span>';
                $$.rosTerm.appendChild(div);
            }});
            $$.rosTerm.scrollTop = $$.rosTerm.scrollHeight;
        }}
    }});

    // Keyboard controls
    document.addEventListener('keydown', e => {{
        if (e.target.tagName === 'INPUT') return;
        if (e.code === 'Space') {{ e.preventDefault(); LS.playing ? pause() : play(); }}
        if (e.code === 'ArrowRight') seekTo(LS.index + (e.shiftKey ? 10 : 1));
        if (e.code === 'ArrowLeft') seekTo(LS.index - (e.shiftKey ? 10 : 1));
        if (e.code === 'KeyR') reset();
        if (e.code === 'Digit1') setSpeed(0.5);
        if (e.code === 'Digit2') setSpeed(1);
        if (e.code === 'Digit3') setSpeed(2);
        if (e.code === 'Digit4') setSpeed(5);
    }});

    window.addEventListener('resize', () => {{ resizeCanvas(); drawCanvas(); }});
}}

// ===== Init =====
function initLiveDemo() {{
    cacheDom();
    resizeCanvas();
    setupEvents();
    reset();
    // Initial draw
    updateAll();
    // Continuous redraw for pulse effect even when paused
    function continuousDraw() {{
        drawCanvas();
        requestAnimationFrame(continuousDraw);
    }}
    continuousDraw();
}}

// Start when DOM ready
if (document.readyState === 'loading') {{
    document.addEventListener('DOMContentLoaded', initLiveDemo);
}} else {{
    initLiveDemo();
}}
"""

# ── Read current HTML ──
with open('index.html', 'r', encoding='utf-8') as f:
    html = f.read()

# ── Inject CSS ──
# Add before "/* ===== Responsive ===== */"
html = html.replace(
    '/* ===== Responsive ===== */',
    LIVE_CSS + '\n/* ===== Responsive ===== */'
)

# ── Replace Screen 2 ──
# Find the Screen 2 section and replace it
old_s2_start = html.find('<!-- ================================================ -->\n<!--  SCREEN 2')
old_s2_end = html.find('<!-- ================================================ -->\n<!--  SCREEN 3')
if old_s2_start != -1 and old_s2_end != -1:
    html = html[:old_s2_start] + SCREEN2_HTML + '\n' + html[old_s2_end:]

# ── Replace old trajectory canvas JS ──
# Replace the old trajectory canvas IIFE with new live JS
old_traj_start = html.find("// ===== Trajectory canvas =====")
old_traj_end = html.find("// ===== GeoCoT stepper =====")
if old_traj_start != -1 and old_traj_end != -1:
    html = html[:old_traj_start] + LIVE_JS + '\n\n' + html[old_traj_end:]

# ── Write ──
with open('index.html', 'w', encoding='utf-8') as f:
    f.write(html)

print('Done! Built index.html with live SL demo.')
print(f'Route data: {len(compact)} points embedded')
print(f'HTML size: {len(html)} bytes ({len(html)/1024:.1f} KB)')
