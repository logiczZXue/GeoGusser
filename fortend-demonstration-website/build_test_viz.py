"""Build the real ROS test visualization Screen 2."""
import json

# ── Read route data ──
with open('route_data.json', 'r') as f:
    route = json.load(f)
compact = []
for pt in route:
    compact.append([round(pt[0],7), round(pt[1],7), round(pt[2],1), round(pt[3],3), round(pt[4],1)])
ROUTE_JS = json.dumps(compact, separators=(',',':'))

# ── Read current HTML ──
with open('index.html', 'r', encoding='utf-8') as f:
    html = f.read()

# ═══════════════════════════════════════════════════════
# PART 1: Clean up duplicated CSS blocks
# ═══════════════════════════════════════════════════════
old_css_marker = "/* ===== Live Demo - Status Bar ===== */"
count = html.count(old_css_marker)
if count > 1:
    parts = html.split(old_css_marker)
    cleaned = [parts[0]]
    for i in range(1, len(parts)):
        block = parts[i]
        resp_idx = block.find("/* ===== Responsive ===== */")
        if resp_idx != -1:
            if i == 1:
                cleaned.append(old_css_marker + block[:resp_idx])
            cleaned.append(block[resp_idx:])
        else:
            cleaned.append(old_css_marker + block)
    html = ''.join(cleaned)

# ═══════════════════════════════════════════════════════
# PART 2: Replace old Live Demo CSS with new Test Viz CSS
# ═══════════════════════════════════════════════════════
old_css_start = html.find("/* ===== Live Demo - Status Bar ===== */")
old_css_end = html.find("/* ===== Responsive ===== */")

NEW_CSS = r"""/* ===== Test Viz - Node Status Bar ===== */
.node-status-bar {
  display: flex; gap: 0.4rem; flex-wrap: wrap; margin-bottom: 0.8rem;
}
.node-status-chip {
  display: flex; align-items: center; gap: 0.35rem;
  background: var(--surface); border: 1px solid var(--border2);
  border-radius: 5px; padding: 0.25rem 0.55rem; font-size: 0.62rem;
  letter-spacing: 0.03em; color: var(--text2); transition: all 0.3s;
}
.node-status-chip .ndot {
  width: 5px; height: 5px; border-radius: 50%; flex-shrink: 0;
}
.node-status-chip .ndot.ok { background: var(--green); box-shadow: 0 0 5px var(--green); }
.node-status-chip .ndot.warn { background: var(--orange); box-shadow: 0 0 5px var(--orange); animation: dotPulse 1s ease-in-out infinite; }
.node-status-chip .ndot.err { background: var(--red); box-shadow: 0 0 5px var(--red); }
.node-status-chip .ndot.master { background: #ff00ff; box-shadow: 0 0 6px #ff00ff; animation: dotPulse 1.5s ease-in-out infinite; }
.node-status-chip .nlabel { color: var(--text3); font-size: 0.58rem; }

/* ===== Test Viz - 3 Column Layout ===== */
.test-viz-grid {
  display: grid; grid-template-columns: 0.9fr 1.2fr 0.9fr; gap: 0.7rem; margin-bottom: 0.6rem;
}

/* ===== Camera Feed Panel ===== */
.cam-panel {
  background: var(--surface); border: 1px solid var(--border2);
  border-radius: 10px; overflow: hidden;
}
.cam-panel-header {
  display: flex; align-items: center; gap: 0.4rem;
  padding: 0.35rem 0.6rem; font-size: 0.6rem; color: var(--text3);
  letter-spacing: 0.04em; border-bottom: 1px solid var(--border2);
  background: rgba(0,0,0,0.3);
}
.cam-panel-header .cam-topic { color: var(--cyan); margin-left: auto; font-size: 0.55rem; }
.cam-panel canvas { width: 100%; height: 200px; display: block; }
.cam-panel-footer {
  display: flex; gap: 1rem; padding: 0.3rem 0.6rem; font-size: 0.58rem;
  color: var(--text3); border-top: 1px solid var(--border2);
  background: rgba(0,0,0,0.3); letter-spacing: 0.03em;
}

/* ===== Trajectory Map Panel ===== */
.traj-map-panel {
  background: var(--surface); border: 1px solid var(--border2);
  border-radius: 10px; overflow: hidden; display: flex; flex-direction: column;
}
.traj-map-panel canvas { width: 100%; height: 320px; display: block; flex: 1; }
.traj-map-legend {
  display: flex; gap: 0.8rem; justify-content: center; padding: 0.3rem;
  font-size: 0.58rem; color: var(--text3); letter-spacing: 0.03em;
  border-top: 1px solid var(--border2); background: rgba(0,0,0,0.3);
}
.traj-map-legend span { display: flex; align-items: center; gap: 0.25rem; }
.leg-dot2 { width: 7px; height: 7px; border-radius: 50%; flex-shrink: 0; }
.leg-dot2.gps-raw { background: var(--orange); }
.leg-dot2.vio-est { background: #5b8def; }
.leg-dot2.match-rt { background: var(--cyan); box-shadow: 0 0 4px var(--cyan); }

/* ===== Right Panels Stack ===== */
.right-panels { display: flex; flex-direction: column; gap: 0.5rem; }
.info-panel {
  background: var(--surface); border: 1px solid var(--border2);
  border-radius: 8px; padding: 0.5rem 0.65rem;
}
.info-panel .ip-header {
  display: flex; align-items: center; gap: 0.35rem;
  font-size: 0.6rem; color: var(--text3); letter-spacing: 0.05em;
  margin-bottom: 0.35rem; padding-bottom: 0.3rem; border-bottom: 1px solid var(--border2);
}
.info-panel .ip-header .ip-icon { font-size: 0.75rem; }
.info-panel .ip-header .ip-topic { margin-left: auto; color: var(--cyan); font-size: 0.55rem; }
.info-panel .ip-row { display: flex; justify-content: space-between; font-size: 0.65rem; padding: 0.1rem 0; }
.info-panel .ip-row .ip-label { color: var(--text3); }
.info-panel .ip-row .ip-val { color: var(--text); }
.info-panel .ip-row .ip-val.cyan { color: var(--cyan); }
.info-panel .ip-row .ip-val.green { color: var(--green); }
.info-panel .ip-row .ip-val.orange { color: var(--orange); }

/* Particle mini canvas */
.particle-mini-canvas { width: 100%; height: 50px; display: block; margin-top: 0.25rem; border-radius: 4px; background: rgba(4,4,18,0.5); }

/* ===== Terminal Multiplexer ===== */
.term-mux { margin-top: 0.5rem; }
.term-tabs {
  display: flex; gap: 0; border-bottom: 1px solid var(--border);
}
.term-tab {
  padding: 0.3rem 0.7rem; font-size: 0.58rem; letter-spacing: 0.03em;
  color: var(--text3); cursor: pointer; background: rgba(8,8,20,0.6);
  border: 1px solid transparent; border-bottom: none; border-radius: 6px 6px 0 0;
  transition: all 0.2s; white-space: nowrap;
}
.term-tab:hover { color: var(--text2); background: rgba(12,12,36,0.8); }
.term-tab.active { color: var(--cyan); background: rgba(0,255,255,0.04); border-color: rgba(0,255,255,0.15); }
.term-tab .tt-dot { width: 4px; height: 4px; border-radius: 50%; display: inline-block; margin-right: 0.3rem; }
.term-tab .tt-dot.live { background: var(--green); }
.term-body {
  background: rgba(4,4,18,0.96); border: 1px solid rgba(0,255,255,0.08);
  border-top: none; border-radius: 0 0 6px 6px; padding: 0.4rem 0.6rem;
  height: 85px; overflow-y: auto; font-family: 'Cascadia Code', 'Fira Code', 'Consolas', monospace;
  font-size: 0.58rem; line-height: 1.5; color: rgba(0,255,200,0.6);
}
.term-body .t-line { white-space: pre; }
.term-body .t-line .ts { color: rgba(255,255,255,0.2); }
.term-body .t-line .kw { color: var(--cyan); }
.term-body .t-line .warn { color: var(--orange); }
.term-body::-webkit-scrollbar { width: 3px; }
.term-body::-webkit-scrollbar-thumb { background: rgba(0,255,255,0.1); border-radius: 2px; }

/* Bag progress */
.bag-progress-bar {
  height: 3px; background: rgba(255,255,255,0.05); border-radius: 2px;
  margin-top: 0.35rem; overflow: hidden;
}
.bag-progress-fill {
  height: 100%; background: var(--cyan); border-radius: 2px;
  transition: width 0.1s linear; box-shadow: 0 0 8px rgba(0,255,255,0.3);
}

/* Responsive */
@media (max-width: 768px) {
  .test-viz-grid { grid-template-columns: 1fr; }
  .cam-panel canvas { height: 140px; }
  .traj-map-panel canvas { height: 220px; }
  .term-tab { padding: 0.25rem 0.4rem; font-size: 0.52rem; }
}
"""

if old_css_start != -1 and old_css_end != -1:
    html = html[:old_css_start] + NEW_CSS + "\n" + html[old_css_end:]

# ═══════════════════════════════════════════════════════
# PART 3: Replace Screen 2 HTML
# ═══════════════════════════════════════════════════════
s2_start_marker = '<!--  SCREEN 2 — LIVE DEMO · 实测回放                  -->'
s3_start_marker = '<!--  SCREEN 3 — VISION AI · 视觉推理                  -->'
s2_start = html.find(s2_start_marker)
s3_start = html.find(s3_start_marker)

NEW_SCREEN2 = r"""<!-- ================================================ -->
    <!--  SCREEN 2 — LIVE TEST · 实机测试可视化             -->
    <!-- ================================================ -->
    <section class="screen" id="trajectory">
    <div class="screen-inner">
      <div class="sec-badge">LIVE TEST · 实机测试</div>
      <div class="sec-title">户外实测<span>流程监控</span></div>
      <div class="sec-desc">模拟完整 ROS 实机测试流程：基于 outdoor_1.bag（塘朗山实测数据），回放 GNSS/IMU/相机数据，VINS-Fusion 实时 VIO 估计，路线匹配器粒子滤波定位。包含全部 5 个终端的真实指令模拟。</div>

      <!-- Node Status Bar -->
      <div class="node-status-bar" id="nodeStatusBar">
        <div class="node-status-chip"><span class="ndot master" id="ndMaster"></span> roscore<span class="nlabel">MASTER</span></div>
        <div class="node-status-chip"><span class="ndot ok" id="ndVins"></span> vins_node<span class="nlabel">VINS-Fusion</span></div>
        <div class="node-status-chip"><span class="ndot ok" id="ndMatch"></span> route_matcher<span class="nlabel">/route_match</span></div>
        <div class="node-status-chip"><span class="ndot ok" id="ndImg"></span> image_view<span class="nlabel">/cam0/image_raw</span></div>
        <div class="node-status-chip"><span class="ndot ok" id="ndBag"></span> rosbag play<span class="nlabel">outdoor_1.bag</span></div>
      </div>

      <!-- 3-Column Main -->
      <div class="test-viz-grid">
        <!-- Left: Camera Feed -->
        <div>
          <div class="cam-panel">
            <div class="cam-panel-header">
              <span>📷 CAM0</span>
              <span class="cam-topic">/cam0/image_raw</span>
            </div>
            <canvas id="camFeedCanvas"></canvas>
            <div class="cam-panel-footer">
              <span id="camFrame">FRAME: 0</span>
              <span>RAW · BAYER</span>
              <span id="camRes">640x480</span>
              <span id="camFps">~30 FPS</span>
            </div>
          </div>
        </div>

        <!-- Center: Trajectory Map -->
        <div>
          <div class="traj-map-panel">
            <canvas id="trajMapCanvas"></canvas>
            <div class="traj-map-legend">
              <span><span class="leg-dot2 gps-raw"></span> GPS Raw (/gps)</span>
              <span><span class="leg-dot2 vio-est"></span> VIO Estimated</span>
              <span><span class="leg-dot2 match-rt"></span> Route Matched</span>
              <span style="margin-left:0.8rem;">FRAME <span id="trajFrameNum" style="color:var(--cyan);">0</span>/1388</span>
            </div>
          </div>
        </div>

        <!-- Right: Data Panels -->
        <div class="right-panels">
          <!-- Bag Playback -->
          <div class="info-panel" id="bagPanel">
            <div class="ip-header"><span class="ip-icon">📦</span>Rosbag Playback<span class="ip-topic">/gnss0:=/gps</span></div>
            <div class="ip-row"><span class="ip-label">File</span><span class="ip-val cyan">outdoor_1.bag</span></div>
            <div class="ip-row"><span class="ip-label">Duration</span><span class="ip-val" id="bagDur">23:08</span></div>
            <div class="ip-row"><span class="ip-label">Speed</span><span class="ip-val orange" id="bagSpeed">-r 1.0</span></div>
            <div class="ip-row"><span class="ip-label">Clock</span><span class="ip-val green">use_sim_time=true</span></div>
            <div class="bag-progress-bar"><div class="bag-progress-fill" id="bagProgressFill" style="width:0%"></div></div>
            <div style="font-size:0.58rem;color:var(--text3);margin-top:0.2rem;text-align:right;" id="bagTimeDisplay">00:00 / 23:08</div>
          </div>

          <!-- Route Matcher -->
          <div class="info-panel" id="matcherPanel">
            <div class="ip-header"><span class="ip-icon">🎯</span>Route Matcher<span class="ip-topic">--num_particles:=1500</span></div>
            <div class="ip-row"><span class="ip-label">Particles</span><span class="ip-val cyan">1500</span></div>
            <div class="ip-row"><span class="ip-label">Converged</span><span class="ip-val" id="matcherConverged">0 / 1500</span></div>
            <div class="ip-row"><span class="ip-label">Effective</span><span class="ip-val green" id="matcherEffective">900 / 1500</span></div>
            <div class="ip-row"><span class="ip-label">Segment</span><span class="ip-val cyan" id="matcherSeg">--</span></div>
            <div class="ip-row"><span class="ip-label">Confidence</span><span class="ip-val orange" id="matcherConf">0.0%</span></div>
            <canvas class="particle-mini-canvas" id="particleMiniCanvas"></canvas>
          </div>

          <!-- Sensor Data Stream -->
          <div class="info-panel">
            <div class="ip-header"><span class="ip-icon">📡</span>Topic Monitor</div>
            <div class="ip-row"><span class="ip-label">/gps (NavSatFix)</span><span class="ip-val green" id="topicGps">10.0 Hz</span></div>
            <div class="ip-row"><span class="ip-label">/cam0/image_raw</span><span class="ip-val green" id="topicCam">30.0 Hz</span></div>
            <div class="ip-row"><span class="ip-label">/vins_estimator/imu_propagate</span><span class="ip-val green" id="topicVio">200.0 Hz</span></div>
            <div class="ip-row"><span class="ip-label">/route_match</span><span class="ip-val" id="topicMatch" style="color:var(--cyan);">1.0 Hz</span></div>
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
        <button data-speed="0.5">0.5x</button>
        <button data-speed="1" class="active-speed">1x</button>
        <button data-speed="2">2x</button>
        <button data-speed="5">5x</button>
        <span class="pb-sep"></span>
        <input type="range" class="pb-progress" id="pbSlider" min="0" max="1387" value="0" title="Progress">
        <span class="pb-time" id="pbTime">0 / 1388 · 0.0%</span>
      </div>

      <!-- Terminal Multiplexer -->
      <div class="term-mux">
        <div class="term-tabs" id="termTabs">
          <div class="term-tab active" data-term="0"><span class="tt-dot live"></span>T1: roscore</div>
          <div class="term-tab" data-term="1"><span class="tt-dot live"></span>T2: vins_node</div>
          <div class="term-tab" data-term="2"><span class="tt-dot live"></span>T3: route_matcher</div>
          <div class="term-tab" data-term="3"><span class="tt-dot live"></span>T4: image_view</div>
          <div class="term-tab" data-term="4"><span class="tt-dot live"></span>T5: rosbag play</div>
        </div>
        <div class="term-body" id="termBody"></div>
      </div>
    </div>
    </section>
"""

if s2_start != -1 and s3_start != -1:
    s2_end = html.rfind('</section>', s2_start, s3_start)
    if s2_end != -1:
        html = html[:s2_start] + NEW_SCREEN2 + '\n' + html[s3_start:]

# ═══════════════════════════════════════════════════════
# PART 4: Replace JS - using placeholder to avoid % formatting conflicts
# ═══════════════════════════════════════════════════════
js_start_marker = '// ===== Live SL Demo - Route Data ====='
js_end_marker = '// ===== GeoCoT stepper ====='
js_start = html.find(js_start_marker)
js_end = html.find(js_end_marker)

# Build the JS with route data inserted via replace
NEW_JS = r"""// ===== Live SL Demo - Route Data =====
const ROUTE_DATA = __ROUTE_PLACEHOLDER__;
const TOTAL_FRAMES = ROUTE_DATA.length;

// ===== State =====
const LS = {
    playing: false, speed: 1.0, index: 0, animId: null, lastTs: 0,
    activeTerm: 0,
    termLines: [ [], [], [], [], [] ],
    particles: [],
    PARTICLE_COUNT: 1500,
};

function idx() { return Math.max(0, Math.min(TOTAL_FRAMES - 1, Math.floor(LS.index))); }

// ===== Terminal Content =====
function initTerminals() {
    LS.termLines[0] = [
        '... logging to /home/kai/.ros/log/9f08c.../master.log',
        'auto-starting new master',
        'process[master]: started with pid [12841]',
        'ROS_MASTER_URI=http://dk2500:11311/',
        'started core service [/rosout]',
        'parameter [/use_sim_time]: true',
    ];
    LS.termLines[1] = [
        'VINS-Fusion v2.1.0 | Mono + IMU mode',
        'Config: /home/kai/AI/VINS-Fusion/config/hiking/hiking_mono_imu_config.yaml',
        'IMU topic: /imu0 | Image topic: /cam0/image_raw',
        'Extrinsic: T_ic = [0.014, 0.032, -0.008, 0.0, 0.0, 0.0]',
        'Feature tracker: 150 max corners, 30 min distance',
        'Estimator initialized. Waiting for IMU measurements...',
        'Init: acc_std=0.012 gyr_std=0.0008 bias_acc=[0.001, -0.003, 0.006]',
    ];
    LS.termLines[2] = [
        'Route Matcher Node v0.9',
        'Route table: /home/kai/AI/AI_Data/Outdoor-1/route_table.csv',
        'Particles: 1500 | Auto baro offset: ENABLED',
        'Subscribed to: /gps, /altitude, /vins_estimator/imu_propagate',
        'Publishing to: /route_match (MatchResult)',
        'GPX route loaded: 47 segments, 12.8 km total',
        'Altitude offset calibrated: +0.35m (baro->GPS)',
        'Particle filter initialized: 1500 particles, uniform distribution',
        'Waiting for GPS fix and VIO initialization...',
    ];
    LS.termLines[3] = [
        'image_view v1.16.0',
        'Subscribing to /cam0/image_raw (raw, BayerRG8)',
        'Transport: raw (no compression)',
        'Image size: 640x480 @ ~30 fps',
        'Window: /cam0/image_raw (image_view)',
        '[WARN] image_transport: raw transport selected, high bandwidth usage',
        'Display window opened [640x480]',
    ];
    LS.termLines[4] = [
        'Bag: /home/kai/AI/AI_Data/Outdoor-1/outdoor_1.bag',
        'Topics: /gnss0 /imu0 /cam0/image_raw /gps (remapped)',
        'Duration: 23:08.4 (1388.2s) | Size: 4.2 GB',
        'Start: 2025-03-15T09:32:18.000000',
        'Remapping /gnss0 -> /gps (NavSatFix)',
        'Clock: sim_time enabled | Rate: 1.0x',
        'Waiting for roscore... connected.',
        'Playing bag. Press [SPACE] to pause.',
    ];
}

// ===== Init particles =====
function initParticles() {
    LS.particles = [];
    const pt = ROUTE_DATA[0];
    for (let i = 0; i < LS.PARTICLE_COUNT; i++) {
        LS.particles.push({
            lon: pt[0] + (Math.random() - 0.5) * 0.008,
            lat: pt[1] + (Math.random() - 0.5) * 0.008,
        });
    }
}

// ===== DOM refs =====
let $$ = {};
function cacheDom() {
    $$.camCanvas = document.getElementById('camFeedCanvas');
    $$.camCtx = $$.camCanvas.getContext('2d');
    $$.camFrame = document.getElementById('camFrame');
    $$.camRes = document.getElementById('camRes');
    $$.camFps = document.getElementById('camFps');
    $$.trajCanvas = document.getElementById('trajMapCanvas');
    $$.trajCtx = $$.trajCanvas.getContext('2d');
    $$.trajFrameNum = document.getElementById('trajFrameNum');
    $$.btnPlay = document.getElementById('btnPlay');
    $$.btnReset = document.getElementById('btnReset');
    $$.btnStepB = document.getElementById('btnStepB');
    $$.btnStepF = document.getElementById('btnStepF');
    $$.slider = document.getElementById('pbSlider');
    $$.pbTime = document.getElementById('pbTime');
    $$.bagProgressFill = document.getElementById('bagProgressFill');
    $$.bagTimeDisplay = document.getElementById('bagTimeDisplay');
    $$.bagSpeed = document.getElementById('bagSpeed');
    $$.matcherConverged = document.getElementById('matcherConverged');
    $$.matcherEffective = document.getElementById('matcherEffective');
    $$.matcherSeg = document.getElementById('matcherSeg');
    $$.matcherConf = document.getElementById('matcherConf');
    $$.particleMini = document.getElementById('particleMiniCanvas');
    $$.particleMiniCtx = $$.particleMini.getContext('2d');
    $$.termBody = document.getElementById('termBody');
    $$.ndVins = document.getElementById('ndVins');
    $$.ndMatch = document.getElementById('ndMatch');
    $$.ndImg = document.getElementById('ndImg');
    $$.ndBag = document.getElementById('ndBag');
    $$.topicGps = document.getElementById('topicGps');
    $$.topicCam = document.getElementById('topicCam');
    $$.topicVio = document.getElementById('topicVio');
    $$.topicMatch = document.getElementById('topicMatch');
}

// ===== Coordinate mapping =====
const DATA_BOUNDS = { lonMin: 113.9532, lonMax: 113.9646, latMin: 22.53, latMax: 22.5360 };
let tW, tH, dpr;

function resizeCanvases() {
    if (!$$.trajCanvas) return;
    dpr = devicePixelRatio || 1;
    const tr = $$.trajCanvas.parentElement.getBoundingClientRect();
    tW = tr.width;
    $$.trajCanvas.width = tW * dpr;
    $$.trajCanvas.height = 320 * dpr;
    $$.trajCanvas.style.width = tW + 'px';
    $$.trajCanvas.style.height = '320px';

    const cr = $$.camCanvas.parentElement.getBoundingClientRect();
    $$.camCanvas.width = cr.width * dpr;
    $$.camCanvas.height = 200 * dpr;
    $$.camCanvas.style.width = cr.width + 'px';
    $$.camCanvas.style.height = '200px';

    const pr = $$.particleMini.parentElement.getBoundingClientRect();
    $$.particleMini.width = pr.width * dpr;
    $$.particleMini.height = 50 * dpr;
    $$.particleMini.style.width = pr.width + 'px';
    $$.particleMini.style.height = '50px';
}

function lonLatToXY(lon, lat) {
    const pad = 35;
    const x = pad + ((lon - DATA_BOUNDS.lonMin) / (DATA_BOUNDS.lonMax - DATA_BOUNDS.lonMin)) * (tW - pad * 2);
    const y = 320 - pad - ((lat - DATA_BOUNDS.latMin) / (DATA_BOUNDS.latMax - DATA_BOUNDS.latMin)) * (320 - pad * 2);
    return { x, y };
}

// ===== Draw Camera Feed (simulated trail scene) =====
function drawCameraFeed() {
    const ctx = $$.camCtx;
    const cw = $$.camCanvas.width / dpr;
    const ch = 200;
    ctx.setTransform(1,0,0,1,0,0);
    ctx.scale(dpr, dpr);
    ctx.clearRect(0, 0, cw, ch);
    ctx.fillStyle = '#050510';
    ctx.fillRect(0, 0, cw, ch);

    // Sky gradient
    const skyGrad = ctx.createLinearGradient(0, 0, 0, ch * 0.55);
    skyGrad.addColorStop(0, '#1a3a5c');
    skyGrad.addColorStop(0.5, '#3a6a8c');
    skyGrad.addColorStop(1, '#6a9aac');
    ctx.fillStyle = skyGrad;
    ctx.fillRect(0, 0, cw, ch * 0.55);

    // Mountains
    const pt = ROUTE_DATA[idx()];
    const altNorm = (pt[2] - 120) / 80;
    ctx.fillStyle = '#1a2a1a';
    ctx.beginPath();
    ctx.moveTo(0, ch);
    const baseY = ch * 0.42;
    for (let x = 0; x < cw; x += 3) {
        const a = x / cw;
        const h = 25 + 45 * Math.sin(a * Math.PI * 1.7) * Math.sin(a * 5.3) + 20 * Math.cos(a * 2.1) + altNorm * 8;
        ctx.lineTo(x, baseY + h);
    }
    ctx.lineTo(cw, ch);
    ctx.closePath();
    ctx.fill();

    // Trail
    ctx.strokeStyle = 'rgba(200,160,100,0.3)';
    ctx.lineWidth = 1.5;
    ctx.setLineDash([2, 6]);
    ctx.beginPath();
    ctx.moveTo(0, baseY + 20);
    ctx.quadraticCurveTo(cw * 0.3, baseY - 5, cw * 0.5, baseY + 10);
    ctx.quadraticCurveTo(cw * 0.7, baseY + 25, cw, baseY + 8);
    ctx.stroke();
    ctx.setLineDash([]);

    // Trees
    ctx.strokeStyle = 'rgba(30,60,20,0.6)';
    ctx.lineWidth = 1;
    for (let i = 0; i < 12; i++) {
        const tx = cw * 0.05 + i * cw / 12 + (Math.sin(i * 4.1) * 10);
        const ty = baseY + 5 + Math.sin(i * 1.7) * 15;
        ctx.beginPath();
        ctx.moveTo(tx, ty); ctx.lineTo(tx - 5, ty - 18);
        ctx.moveTo(tx, ty); ctx.lineTo(tx + 5, ty - 18);
        ctx.stroke();
    }

    // REC indicator
    ctx.font = '10px monospace';
    ctx.fillStyle = 'rgba(255,0,64,0.8)';
    ctx.fillText('● REC', 8, 18);
    ctx.fillStyle = 'rgba(255,255,255,0.4)';
    ctx.fillText('BAG: outdoor_1.bag', 8, 32);

    // Crosshair
    ctx.strokeStyle = 'rgba(255,255,255,0.15)';
    ctx.lineWidth = 0.5;
    ctx.beginPath(); ctx.moveTo(cw/2, 0); ctx.lineTo(cw/2, ch); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(0, ch/2); ctx.lineTo(cw, ch/2); ctx.stroke();
    ctx.beginPath(); ctx.arc(cw/2, ch/2, 20, 0, Math.PI*2); ctx.stroke();

    // Frame counter
    ctx.fillStyle = 'rgba(255,255,255,0.5)';
    ctx.font = '9px monospace';
    ctx.fillText('FRAME: ' + idx(), cw - 90, 18);

    $$.camFrame.textContent = 'FRAME: ' + idx();
    $$.camFps.textContent = '~' + Math.round(25 + Math.random() * 10) + ' FPS';
}

// ===== Draw Trajectory Map (multi-track) =====
function drawTrajMap() {
    const ctx = $$.trajCtx;
    ctx.setTransform(1,0,0,1,0,0);
    ctx.scale(dpr, dpr);
    ctx.clearRect(0, 0, tW, 320);

    // Grid
    ctx.strokeStyle = 'rgba(0,255,255,0.02)';
    ctx.lineWidth = 0.4;
    for (let gx = 0; gx < tW; gx += 40) { ctx.beginPath(); ctx.moveTo(gx, 0); ctx.lineTo(gx, 320); ctx.stroke(); }
    for (let gy = 0; gy < 320; gy += 40) { ctx.beginPath(); ctx.moveTo(0, gy); ctx.lineTo(tW, gy); ctx.stroke(); }

    const ci = idx();

    // Reference path (very faint)
    ctx.beginPath();
    let f = lonLatToXY(ROUTE_DATA[0][0], ROUTE_DATA[0][1]);
    ctx.moveTo(f.x, f.y);
    for (let i = 1; i < TOTAL_FRAMES; i += 2) {
        const p = lonLatToXY(ROUTE_DATA[i][0], ROUTE_DATA[i][1]);
        ctx.lineTo(p.x, p.y);
    }
    ctx.strokeStyle = 'rgba(255,255,255,0.03)';
    ctx.lineWidth = 1;
    ctx.setLineDash([2, 8]);
    ctx.stroke();
    ctx.setLineDash([]);

    // GPS raw dots (every 10th, with noise)
    for (let i = 0; i <= ci; i += 10) {
        const p = lonLatToXY(ROUTE_DATA[i][0] + (Math.random()-0.5)*0.00004,
                             ROUTE_DATA[i][1] + (Math.random()-0.5)*0.00004);
        ctx.beginPath();
        ctx.arc(p.x, p.y, 1.8, 0, Math.PI * 2);
        ctx.fillStyle = 'rgba(255,140,0,0.5)';
        ctx.fill();
    }

    // GPS raw path (orange dashed)
    if (ci > 0) {
        ctx.beginPath();
        let g0 = lonLatToXY(ROUTE_DATA[0][0], ROUTE_DATA[0][1]);
        ctx.moveTo(g0.x, g0.y);
        for (let i = 5; i <= ci; i += 5) {
            const p = lonLatToXY(ROUTE_DATA[i][0] + (Math.random()-0.5)*0.00003,
                                 ROUTE_DATA[i][1] + (Math.random()-0.5)*0.00003);
            ctx.lineTo(p.x, p.y);
        }
        ctx.strokeStyle = 'rgba(255,140,0,0.3)';
        ctx.lineWidth = 1.2;
        ctx.setLineDash([3, 6]);
        ctx.stroke();
        ctx.setLineDash([]);
    }

    // VIO estimated track (blue, slightly offset from GPS)
    if (ci > 0) {
        ctx.beginPath();
        let v0 = lonLatToXY(ROUTE_DATA[0][0] + 0.00002, ROUTE_DATA[0][1] + 0.00001);
        ctx.moveTo(v0.x, v0.y);
        for (let i = 1; i <= ci; i++) {
            const drift = i * 0.0000003;
            const p = lonLatToXY(ROUTE_DATA[i][0] + 0.00002 + drift,
                                 ROUTE_DATA[i][1] + 0.00001 + drift * 0.5);
            ctx.lineTo(p.x, p.y);
        }
        ctx.strokeStyle = 'rgba(91,141,239,0.5)';
        ctx.lineWidth = 2;
        ctx.stroke();
    }

    // Matched route (bright cyan)
    if (ci > 0) {
        ctx.beginPath();
        let m0 = lonLatToXY(ROUTE_DATA[0][0], ROUTE_DATA[0][1]);
        ctx.moveTo(m0.x, m0.y);
        for (let i = 1; i <= ci; i++) {
            const p = lonLatToXY(ROUTE_DATA[i][0], ROUTE_DATA[i][1]);
            ctx.lineTo(p.x, p.y);
        }
        ctx.strokeStyle = 'rgba(0,255,255,0.7)';
        ctx.lineWidth = 2.5;
        ctx.stroke();
        ctx.strokeStyle = 'rgba(0,255,255,0.1)';
        ctx.lineWidth = 7;
        ctx.stroke();
    }

    // Particle swarm near current position
    if (ci > 0 && ci < TOTAL_FRAMES) {
        const cp = lonLatToXY(ROUTE_DATA[ci][0], ROUTE_DATA[ci][1]);
        const progress = ci / TOTAL_FRAMES;
        const spread = Math.max(2, 40 - progress * 35);
        for (let p = 0; p < 80; p++) {
            const seed = p * 137.5;
            const a = seed % 7.283;
            const r = spread * (0.3 + 0.7 * Math.abs(Math.sin(seed * 0.37)));
            const px = cp.x + Math.cos(a) * r;
            const py = cp.y + Math.sin(a) * r;
            ctx.beginPath();
            ctx.arc(px, py, 0.6, 0, Math.PI * 2);
            ctx.fillStyle = 'rgba(0,255,255,' + (0.15 + 0.2 * Math.sin(seed)) + ')';
            ctx.fill();
        }
    }

    // Current position pulse dot
    if (ci < TOTAL_FRAMES) {
        const cp = lonLatToXY(ROUTE_DATA[ci][0], ROUTE_DATA[ci][1]);
        const pulse = 1 + 0.35 * Math.sin(performance.now() * 0.005);
        ctx.beginPath();
        ctx.arc(cp.x, cp.y, 14 * pulse, 0, Math.PI * 2);
        ctx.fillStyle = 'rgba(0,255,255,' + (0.06 * pulse) + ')';
        ctx.fill();
        ctx.beginPath();
        ctx.arc(cp.x, cp.y, 5, 0, Math.PI * 2);
        ctx.fillStyle = 'rgba(0,255,240,0.9)';
        ctx.fill();
        ctx.strokeStyle = 'rgba(255,255,255,0.8)';
        ctx.lineWidth = 1.5;
        ctx.stroke();
    }

    $$.trajFrameNum.textContent = ci;
}

// ===== Draw Particle Mini Canvas =====
function drawParticleMini() {
    const ctx = $$.particleMiniCtx;
    const pw = $$.particleMini.width / dpr;
    const ph = 50;
    ctx.setTransform(1,0,0,1,0,0);
    ctx.scale(dpr, dpr);
    ctx.clearRect(0, 0, pw, ph);
    ctx.fillStyle = 'rgba(4,4,18,0.5)';
    ctx.fillRect(0, 0, pw, ph);

    const ci = idx();
    if (ci >= TOTAL_FRAMES) return;
    const progress = ci / TOTAL_FRAMES;
    const spread = Math.max(2, 50 - progress * 48);

    for (let p = 0; p < 150; p++) {
        const seed = p * 197.3;
        const a = seed % 7.283;
        const r = spread * (0.2 + 0.8 * Math.abs(Math.sin(seed * 0.43)));
        const px = pw/2 + Math.cos(a) * r;
        const py = ph/2 + Math.sin(a) * r;
        ctx.beginPath();
        ctx.arc(px, py, 0.5, 0, Math.PI * 2);
        ctx.fillStyle = 'rgba(0,255,255,' + (0.2 + 0.3 * Math.sin(seed * 0.17 + progress * 10)) + ')';
        ctx.fill();
    }
    ctx.beginPath();
    ctx.arc(pw/2, ph/2, 2, 0, Math.PI * 2);
    ctx.fillStyle = 'rgba(0,255,255,0.8)';
    ctx.fill();
}

// ===== Update Data Panels =====
function updatePanels() {
    const ci = idx();
    const pt = ROUTE_DATA[ci];
    const dist = pt[4];
    const progress = ci / TOTAL_FRAMES;

    // Bag panel
    const elapsedSec = Math.floor(ci * 1.0);
    const min = Math.floor(elapsedSec / 60);
    const sec = elapsedSec % 60;
    $$.bagTimeDisplay.textContent = String(min).padStart(2,'0') + ':' + String(sec).padStart(2,'0') + ' / 23:08';
    $$.bagProgressFill.style.width = (progress * 100).toFixed(1) + '%';
    $$.bagSpeed.textContent = '-r ' + LS.speed.toFixed(1);

    // Matcher panel
    const converged = Math.floor(progress * 1500);
    const effective = Math.floor(converged * (0.5 + 0.5 * Math.min(1, progress * 2)));
    const segId = Math.max(1, Math.floor(dist / 25));
    const conf = Math.min(0.98, progress * 0.95 + (Math.random() - 0.5) * 0.04);
    $$.matcherConverged.textContent = converged + ' / 1500';
    $$.matcherEffective.textContent = effective + ' / 1500';
    $$.matcherSeg.textContent = 'Seg ' + segId;
    $$.matcherConf.textContent = (conf * 100).toFixed(1) + '%';

    // Topic rates
    $$.topicGps.textContent = (9.8 + Math.random() * 0.4).toFixed(1) + ' Hz';
    $$.topicCam.textContent = (29.5 + Math.random() * 1).toFixed(1) + ' Hz';
    $$.topicVio.textContent = (198 + Math.random() * 4).toFixed(1) + ' Hz';
    $$.topicMatch.textContent = (0.9 + Math.random() * 0.2).toFixed(1) + ' Hz';
}

// ===== Terminal Multiplexer =====
function renderTerminal() {
    const lines = LS.termLines[LS.activeTerm];
    if (!$$.termBody) return;
    let html = '';
    for (const l of lines) {
        html += '<div class="t-line"><span class="ts">$</span> ' + escapeHtml(l) + '</div>';
    }
    $$.termBody.innerHTML = html;
    $$.termBody.scrollTop = $$.termBody.scrollHeight;
}

function escapeHtml(s) {
    return s.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
}

function addTermLine(termIdx, line) {
    LS.termLines[termIdx].push(line);
    if (LS.termLines[termIdx].length > 200) LS.termLines[termIdx].shift();
    if (termIdx === LS.activeTerm) renderTerminal();
}

function switchTerm(idx) {
    LS.activeTerm = idx;
    document.querySelectorAll('.term-tab').forEach((t, i) => t.classList.toggle('active', i === idx));
    renderTerminal();
}

// ===== Simulated terminal output =====
function generateTermOutput() {
    const ci = idx();
    const pt = ROUTE_DATA[ci];
    const dist = pt[4];
    const segId = Math.max(1, Math.floor(dist / 25));
    const conf = Math.min(0.98, (ci / TOTAL_FRAMES) * 0.95);

    if (ci % 200 === 0) addTermLine(0, '[INFO] master: registering service [/rosout/get_loggers]');
    if (ci % 300 === 50) addTermLine(0, '[INFO] +PUB [/rosout] /rosout_agg | msgs: ' + (ci + 10));

    if (ci % 30 === 0) addTermLine(1, '[VINS] frame ' + ci + ' | features: ' + (120 + Math.floor(Math.random()*30)) + ' | solver: ' + (6 + Math.random()*4).toFixed(1) + 'ms');
    if (ci % 100 === 10) addTermLine(1, '[VINS] marginalization | oldest: ' + (ci - 20) + ' | newest: ' + ci);
    if (ci === 50) addTermLine(1, '[VINS] init finished. Gravity: ' + (9.78 + Math.random()*0.04).toFixed(2) + ' m/s^2');

    if (ci % 40 === 0) addTermLine(2, '[MATCHER] step ' + ci + ' | best: seg_' + segId + ' (conf=' + (conf*100).toFixed(1) + '%) | alive: ' + Math.floor(700 + conf*800));
    if (ci % 80 === 10) addTermLine(2, '[MATCHER] resampling | effective_N: ' + (700 + Math.floor(conf*800)));
    if (ci === 100) addTermLine(2, '[MATCHER] baro_offset calibrated: +0.35m | alt_match: GOOD');

    if (ci % 60 === 0) addTermLine(3, '[IMAGE] received frame ' + ci + ' | 640x480 BayerRG8 | ' + (25 + Math.floor(Math.random()*10)) + ' fps');
    if (ci % 180 === 30) addTermLine(3, '[IMAGE] display update | latency: ' + (15 + Math.random()*10).toFixed(1) + 'ms');

    if (ci % 100 === 0) {
        const d = Math.floor(ci / 60), s = ci % 60;
        addTermLine(4, '[BAG] ' + String(d).padStart(2,'0') + ':' + String(s).padStart(2,'0') + ' / 23:08 | ' + ((ci/TOTAL_FRAMES)*100).toFixed(1) + '% | speed ' + LS.speed.toFixed(1) + 'x');
    }
    if (ci === 0) {
        addTermLine(4, '[BAG] Remapping /gnss0 -> /gps');
        addTermLine(4, '[BAG] Waiting for roscore... OK');
        addTermLine(4, '[BAG] Clock set: use_sim_time=true');
    }
}

// ===== Playback =====
function setSpeed(s) {
    LS.speed = s;
    document.querySelectorAll('#playbackBar [data-speed]').forEach(b => b.classList.remove('active-speed'));
    const btn = document.querySelector('#playbackBar [data-speed="' + s + '"]');
    if (btn) btn.classList.add('active-speed');
}

function seekTo(newIdx) {
    LS.index = Math.max(0, Math.min(TOTAL_FRAMES - 1, Math.floor(newIdx)));
    $$.slider.value = idx();
    updateAll();
}

function updateAll() {
    drawCameraFeed();
    drawTrajMap();
    drawParticleMini();
    updatePanels();
    const pct = TOTAL_FRAMES > 1 ? (idx() / (TOTAL_FRAMES - 1) * 100) : 0;
    $$.pbTime.textContent = idx() + ' / ' + TOTAL_FRAMES + ' · ' + pct.toFixed(1) + '%';
    $$.slider.value = idx();
}

function play() {
    if (LS.playing) return;
    LS.playing = true;
    $$.btnPlay.textContent = '⏸';
    LS.lastTs = performance.now();
    function tick(ts) {
        if (!LS.playing) return;
        const dt = ts - LS.lastTs;
        LS.lastTs = ts;
        const advance = (dt / 16) * LS.speed;
        LS.index += advance;
        if (LS.index >= TOTAL_FRAMES - 1) { LS.index = TOTAL_FRAMES - 1; pause(); }
        LS.index = Math.min(TOTAL_FRAMES - 1, LS.index);
        updateAll();
        if (LS.playing && idx() % 3 === 0) generateTermOutput();
        LS.animId = requestAnimationFrame(tick);
    }
    LS.animId = requestAnimationFrame(tick);
}

function pause() {
    LS.playing = false;
    $$.btnPlay.textContent = '▶';
    if (LS.animId) { cancelAnimationFrame(LS.animId); LS.animId = null; }
}

function reset() {
    pause();
    LS.index = 0;
    LS.activeTerm = 0;
    LS.termLines = [ [], [], [], [], [] ];
    initTerminals();
    initParticles();
    switchTerm(0);
    updateAll();
    renderTerminal();
}

// ===== Event Handlers =====
function setupEvents() {
    $$.btnPlay.addEventListener('click', () => LS.playing ? pause() : play());
    $$.btnReset.addEventListener('click', reset);
    $$.btnStepF.addEventListener('click', () => seekTo(idx() + 1));
    $$.btnStepB.addEventListener('click', () => seekTo(idx() - 1));

    document.querySelectorAll('#playbackBar [data-speed]').forEach(b => {
        b.addEventListener('click', () => setSpeed(parseFloat(b.dataset.speed)));
    });

    $$.slider.addEventListener('input', () => seekTo(parseInt($$.slider.value)));

    document.querySelectorAll('.term-tab').forEach(t => {
        t.addEventListener('click', () => switchTerm(parseInt(t.dataset.term)));
    });

    document.addEventListener('keydown', e => {
        if (e.target.tagName === 'INPUT') return;
        if (e.code === 'Space') { e.preventDefault(); LS.playing ? pause() : play(); }
        if (e.code === 'ArrowRight') seekTo(idx() + (e.shiftKey ? 10 : 1));
        if (e.code === 'ArrowLeft') seekTo(idx() - (e.shiftKey ? 10 : 1));
        if (e.code === 'KeyR') reset();
        if (e.code === 'Digit1') { setSpeed(0.5); switchTerm(0); }
        if (e.code === 'Digit2') { setSpeed(1); switchTerm(1); }
        if (e.code === 'Digit3') { setSpeed(2); switchTerm(2); }
        if (e.code === 'Digit4') { setSpeed(5); switchTerm(3); }
        if (e.code === 'Digit5') switchTerm(4);
    });

    window.addEventListener('resize', () => { resizeCanvases(); updateAll(); });
}

// ===== Init =====
function initTestViz() {
    cacheDom();
    initTerminals();
    initParticles();
    resizeCanvases();
    setupEvents();
    renderTerminal();
    updateAll();
    function continuousDraw() {
        drawCameraFeed();
        drawTrajMap();
        drawParticleMini();
        requestAnimationFrame(continuousDraw);
    }
    continuousDraw();
}

if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initTestViz);
} else {
    initTestViz();
}
""".replace('__ROUTE_PLACEHOLDER__', ROUTE_JS)

if js_start != -1 and js_end != -1:
    html = html[:js_start] + NEW_JS + '\n\n' + html[js_end:]

# ═══════════════════════════════════════════════════════
# Write output
# ═══════════════════════════════════════════════════════
with open('index.html', 'w', encoding='utf-8') as f:
    f.write(html)

print('Done! Built real ROS test visualization.')
print('Route data: {} points embedded'.format(len(compact)))
print('HTML size: {} bytes ({:.1f} KB)'.format(len(html), len(html)/1024))
