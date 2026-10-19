// 3D viewer for the edge drone (infinite sky with weather).
//
// WebSocket messages from server_edge.py:
//   "state"     : drone pose, 28 sensor rays, goal, wind, battery, KPIs, telemetry counters
//   "obstacles" : axis-aligned boxes [x0,y0,z0,x1,y1,z1] for the chunks around the drone
//   "targets"   : vehicles, people, trees and hot rocks around the drone
//   "camera"    : the drone's RGB / thermal / height views with the detector's boxes (PNG)
//
// World coords (server): x, y horizontal, z up, yaw CCW from +x.
// Scene mapping:  world (x, y, z)  ->  three (x, z, -y).   Yaw about world z == rotation.y in three.

import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

const $ = (id) => document.getElementById(id);
const T = (x, y, z) => new THREE.Vector3(x, z, -y);

// ── renderer / scene ─────────────────────────────────────────────────────────────
const canvas = $('view');
const renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
const scene = new THREE.Scene();
const HORIZON = new THREE.Color(0xc4e4ff);
scene.fog = new THREE.Fog(HORIZON, 500, 2600);
const camera = new THREE.PerspectiveCamera(65, 1, 1, 40000);
camera.position.set(-80, 60, 0);
const controls = new OrbitControls(camera, canvas);
controls.enableDamping = true;
controls.enabled = false;

scene.add(new THREE.HemisphereLight(0xdff0ff, 0x4a5a3a, 0.95));
const sun = new THREE.DirectionalLight(0xfff2d8, 1.6);
sun.position.set(600, 1200, 400);
scene.add(sun);

// sky dome that follows the camera
const sky = new THREE.Mesh(
  new THREE.SphereGeometry(20000, 32, 16),
  new THREE.ShaderMaterial({
    side: THREE.BackSide, depthWrite: false, fog: false,
    uniforms: { top: { value: new THREE.Color(0x0a3d91) }, bottom: { value: HORIZON } },
    vertexShader: 'varying vec3 vP; void main(){ vP = position; gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0); }',
    fragmentShader: 'uniform vec3 top; uniform vec3 bottom; varying vec3 vP; void main(){ float h = clamp(normalize(vP).y, 0.0, 1.0); gl_FragColor = vec4(mix(bottom, top, pow(h, 0.55)), 1.0); }',
  }));
scene.add(sky);

// ground: a big grid plane that is re-centred under the drone in whole tiles, so it looks endless
const TILE = 100, PLANE = 30000;
const gridCanvas = document.createElement('canvas'); gridCanvas.width = gridCanvas.height = 256;
{ const g = gridCanvas.getContext('2d'); g.fillStyle = '#2f4a37'; g.fillRect(0, 0, 256, 256);
  g.strokeStyle = 'rgba(190,230,200,.55)'; g.lineWidth = 3; g.strokeRect(0, 0, 256, 256);
  g.strokeStyle = 'rgba(190,230,200,.18)'; g.lineWidth = 1; g.beginPath(); g.moveTo(128, 0); g.lineTo(128, 256); g.moveTo(0, 128); g.lineTo(256, 128); g.stroke(); }
const gridTex = new THREE.CanvasTexture(gridCanvas);
gridTex.wrapS = gridTex.wrapT = THREE.RepeatWrapping; gridTex.repeat.set(PLANE / TILE, PLANE / TILE);
gridTex.anisotropy = 8;
const ground = new THREE.Mesh(new THREE.PlaneGeometry(PLANE, PLANE), new THREE.MeshLambertMaterial({ map: gridTex }));
ground.rotation.x = -Math.PI / 2;
scene.add(ground);

// clouds: a fixed set of puffs that wrap around the drone, so the sky never runs out
const CLOUD_N = 70, CLOUD_BOX = 5000;
const clouds = new THREE.Group();
const cloudMat = new THREE.MeshLambertMaterial({ color: 0xffffff, transparent: true, opacity: 0.8, fog: false });
const cloudBase = [];
{ let s = 12345; const rnd = () => (s = (s * 1664525 + 1013904223) >>> 0) / 4294967296;
  for (let i = 0; i < CLOUD_N; i++) {
    const puff = new THREE.Group();
    for (let k = 0; k < 4; k++) {
      const m = new THREE.Mesh(new THREE.SphereGeometry(80 + rnd() * 120, 10, 8), cloudMat);
      m.position.set((rnd() - 0.5) * 320, (rnd() - 0.5) * 60, (rnd() - 0.5) * 240); m.scale.y = 0.45; puff.add(m);
    }
    cloudBase.push([rnd() * CLOUD_BOX, 1100 + rnd() * 1300, rnd() * CLOUD_BOX]); clouds.add(puff);
  } }
scene.add(clouds);

// ── obstacles ─────────────────────────────────────────────────────────────────
// Towers rising from the ground use the SAME building style as the self-driving-car city: dark facade with
// lit windows (same window texture) and the same grey-blue palette. Floating blocks stay bright violet/magenta.
function windowTexture() {
  const c = document.createElement('canvas'); c.width = 64; c.height = 128; const g = c.getContext('2d');
  g.fillStyle = '#20242e'; g.fillRect(0, 0, c.width, c.height);
  const cols = 4, rows = 8, pad = 5;
  const cw = (c.width - pad * (cols + 1)) / cols, ch = (c.height - pad * (rows + 1)) / rows;
  for (let r = 0; r < rows; r++) for (let col = 0; col < cols; col++) {
    g.fillStyle = Math.random() < 0.5 ? '#ffe6a8' : '#10131a';
    g.fillRect(pad + col * (cw + pad), pad + r * (ch + pad), cw, ch);
  }
  const t = new THREE.CanvasTexture(c);
  t.colorSpace = THREE.SRGBColorSpace; t.wrapS = t.wrapT = THREE.RepeatWrapping; t.repeat.set(1, 2);
  return t;
}

const winTex = windowTexture();
const buildingMat = new THREE.MeshStandardMaterial({
  map: winTex, emissiveMap: winTex, emissive: 0xffcf7a, emissiveIntensity: 0.0,
  roughness: 0.7, metalness: 0.1, color: 0xffffff,
});
const BUILDING_PALETTE = [0x8a93a6, 0x9fa6b6, 0x70788c, 0xb6bccb, 0x646b7d];
const MAX_BOXES = 800;
const unitBox = new THREE.BoxGeometry(1, 1, 1);
const towerMesh = new THREE.InstancedMesh(unitBox, buildingMat, MAX_BOXES);
towerMesh.count = 0; towerMesh.frustumCulled = false; scene.add(towerMesh);

// ── floating obstacles: real birds (3 species) and small patrol drones, not boxes ──────────────────────────
// Real-world scale (a bird body is well under 1 sim unit = 1 m), positioned at the hazard box's centre. Reused
// via a position-keyed pool (obstacleByKey) instead of rebuilt every call, and animated (wing flap / rotor spin)
// once per frame in frame() below.
const _m = new THREE.Matrix4(), _c = new THREE.Color(), _oq = new THREE.Quaternion();

const BIRD_SCALE = 16, DRONE_SCALE = 14;                                         // real-world meters -> this world's sim units
const BIRD_SPECIES = [
  { name: 'crow', body: 0x1c1c1c, wing: 0x2a2a2a, beak: 0x333333, flapHz: 4.2, span: 0.85 },
  { name: 'gull', body: 0xf0f0f0, wing: 0xb9b9b9, beak: 0xe8a23c, flapHz: 3.0, span: 1.1 },
  { name: 'hawk', body: 0x5b3a22, wing: 0x7a5230, beak: 0x2b2b2b, flapHz: 2.2, span: 1.3 },
];

function makeBird(spec) {
  const g = new THREE.Group();
  const bodyMat = new THREE.MeshStandardMaterial({ color: spec.body, roughness: 0.85, metalness: 0.02 });
  const wingMat = new THREE.MeshStandardMaterial({ color: spec.wing, roughness: 0.8, metalness: 0.02, side: THREE.DoubleSide });
  const beakMat = new THREE.MeshStandardMaterial({ color: spec.beak, roughness: 0.6, metalness: 0.1 });
  const bodyLen = spec.span * 0.42;

  const body = new THREE.Mesh(new THREE.CapsuleGeometry(bodyLen * 0.22, bodyLen * 0.55, 4, 8), bodyMat);
  body.rotation.z = Math.PI / 2; g.add(body);
  const head = new THREE.Mesh(new THREE.SphereGeometry(bodyLen * 0.2, 8, 6), bodyMat);
  head.position.set(bodyLen * 0.62, bodyLen * 0.05, 0); g.add(head);
  const beak = new THREE.Mesh(new THREE.ConeGeometry(bodyLen * 0.06, bodyLen * 0.22, 6), beakMat);
  beak.rotation.z = -Math.PI / 2; beak.position.set(bodyLen * 0.86, bodyLen * 0.03, 0); g.add(beak);
  const tail = new THREE.Mesh(new THREE.ConeGeometry(bodyLen * 0.22, bodyLen * 0.5, 4), wingMat);
  tail.rotation.z = Math.PI / 2; tail.position.set(-bodyLen * 0.68, 0, 0); g.add(tail);

  const wingLen = spec.span * 0.5;
  const wingGeo = new THREE.BoxGeometry(wingLen, wingLen * 0.03, wingLen * 0.34);
  wingGeo.translate(0, 0, wingLen * 0.15);                                        // root edge near the pivot, tip further out
  const pivotL = new THREE.Group(); pivotL.position.set(0, bodyLen * 0.05, bodyLen * 0.14);
  const wingL = new THREE.Mesh(wingGeo, wingMat); pivotL.add(wingL); g.add(pivotL);
  const pivotR = new THREE.Group(); pivotR.position.set(0, bodyLen * 0.05, -bodyLen * 0.14); pivotR.rotation.x = Math.PI;
  const wingR = new THREE.Mesh(wingGeo, wingMat); pivotR.add(wingR); g.add(pivotR);

  g.scale.setScalar(BIRD_SCALE);                                                  // real-world-meter dimensions above -> this world's sim units (player drone body is 9 units long)
  return { group: g, wingL: pivotL, wingR: pivotR, flapHz: spec.flapHz };
}

const NPC_DRONE_COLORS = [0xd94b2b, 0xe8e8e8, 0x3a6fb0, 0x2e2e2e, 0xf2c744, 0x3fae5a, 0x8a4bd9, 0xff7a1a];

function makeNpcDrone(color) {
  const g = new THREE.Group();
  const bodyMat = new THREE.MeshStandardMaterial({ color, metalness: 0.4, roughness: 0.4 });
  const dark = new THREE.MeshStandardMaterial({ color: 0x14181f, metalness: 0.3, roughness: 0.5 });
  g.add(new THREE.Mesh(new THREE.BoxGeometry(0.42, 0.13, 0.3), bodyMat));
  const pod = new THREE.Mesh(new THREE.SphereGeometry(0.09, 10, 6), dark); pod.position.set(0.2, -0.03, 0); g.add(pod);
  const rotors = [];
  for (const [ax, az] of [[0.32, 0.32], [0.32, -0.32], [-0.32, 0.32], [-0.32, -0.32]]) {
    const arm = new THREE.Mesh(new THREE.BoxGeometry(Math.hypot(ax, az) * 1.1, 0.035, 0.045), dark);
    arm.position.set(ax / 2, 0.01, az / 2); arm.rotation.y = -Math.atan2(az, ax); g.add(arm);
    const rotor = new THREE.Mesh(new THREE.CylinderGeometry(0.17, 0.17, 0.012, 16),
      new THREE.MeshBasicMaterial({ color: 0xd8e8ff, transparent: true, opacity: 0.45 }));
    rotor.position.set(ax, 0.07, az); g.add(rotor); rotors.push(rotor);
  }
  const led = new THREE.Mesh(new THREE.SphereGeometry(0.035, 6, 6), new THREE.MeshBasicMaterial({ color: 0xff3030 }));
  led.position.set(0.18, 0.05, 0); g.add(led);
  g.scale.setScalar(DRONE_SCALE);
  return { group: g, rotors };
}

function obsHash01(x, y, z) {                                                     // stable per-position hash in [0, 1) -- not the same helper as the vehicle/tree hash01(x, y) below, needs 3 axes
  return Math.abs(Math.sin(x * 12.9898 + y * 78.233 + z * 37.719) * 43758.5453) % 1;
}

const obstacleByKey = new Map();                                                 // position-keyed pool: created once, reused, hidden when out of view

const hazardMarkerGeo = new THREE.TorusGeometry(1, 0.12, 8, 28);                  // unit ring, scaled to 30 below -- same size as the goalRing (GOAL_RADIUS defaults to 30)
const hazardMarkerMat = new THREE.MeshBasicMaterial({ color: 0xff3b3b, transparent: true, opacity: 0.75 });
const hazardMarkers = new Map();                                                 // one marker per hazard SPOT (not per flyer): a stack of 3 rings on a pole, at the fixed anchor

function makeHazardMarker() {
  const g = new THREE.Group();
  const rings = [];
  for (const dy of [-6, 6]) {                                                    // "multiple rings in a single pole" -- stacked at the anchor, sized to the flyer's own wander range, not the goal ring's much larger radius
    const r = new THREE.Mesh(hazardMarkerGeo, hazardMarkerMat);
    r.scale.setScalar(14); r.rotation.x = Math.PI / 2; r.position.y = dy;
    g.add(r); rings.push(r);
  }
  scene.add(g);
  return { group: g, rings, phase: Math.random() * Math.PI * 2 };
}

function keyFor(cx, cy, cz) { return Math.round(cx) + '_' + Math.round(cy) + '_' + Math.round(cz); }

function makeMember(kind, mh) {                                                  // mh: a hash in [0,1), distinct per swarm member
  const built = kind === 3
    ? makeNpcDrone(NPC_DRONE_COLORS[Math.floor(mh * NPC_DRONE_COLORS.length) % NPC_DRONE_COLORS.length])
    : makeBird(BIRD_SPECIES[kind]);
  scene.add(built.group);
  return {
    ...built,
    flapPhase: mh * 20,
    heading: mh * Math.PI * 2,                                                  // current facing/travel direction -- randomly steered each frame, not a fixed path
    localX: 0, localZ: 0,                                                        // running offset from the flock's anchor point
    speed: (kind === 3 ? 16 : 9) * (0.7 + mh * 0.6),                             // sim units / second
    turnRate: (kind === 3 ? 0.5 : 1.3) * (0.7 + mh * 0.6),                       // radians/s of random heading drift -- drones turn steadier than birds
    leash: (kind === 3 ? 40 : 26) + mh * 20,                                     // how far it's allowed to wander before it's steered/snapped back
    bobAmp: 2 + mh * 5,
    bobSpeed: 0.5 + mh * 0.7,
    bobPhase: mh * Math.PI * 2,
  };
}

function poolEntry(key, kind, hv) {
  let e = obstacleByKey.get(key);
  if (e && e.kind === kind) return e;
  if (e) for (const m of e.members) scene.remove(m.group);
  // most spots are a single flyer; some become a flock of birds or a small drone patrol, picked stably from the hash
  const n = kind === 3 ? (hv < 0.28 ? 2 + (Math.floor(hv * 977) % 2) : 1)         // ~28% of drone spots: 2-3 drones
                       : (hv < 0.35 ? 3 + (Math.floor(hv * 977) % 4) : 1);        // ~35% of bird spots: 3-6 birds
  const members = [];
  for (let m = 0; m < n; m++) members.push(makeMember(kind, (hv + m * 0.1373) % 1));
  const entry = { kind, key, members, visible: true };
  obstacleByKey.set(key, entry);
  return entry;
}

function setObstacles(boxes) {
  let nt = 0;
  const used = new Set();
  const usedMarkers = new Set();
  for (let i = 0; i < boxes.length; i++) {
    const [x0, y0, z0, x1, y1, z1] = boxes[i];
    const w = x1 - x0, d = y1 - y0, h = z1 - z0, cx = (x0 + x1) / 2, cy = (y0 + y1) / 2;
    if (z0 <= 0.5) {                                                              // tower = city-style building
      if (nt >= MAX_BOXES) continue;
      _m.compose(T(cx, cy, (z0 + z1) / 2), _oq.identity(), new THREE.Vector3(w, h, d));
      towerMesh.setMatrixAt(nt, _m); towerMesh.setColorAt(nt, _c.set(BUILDING_PALETTE[nt % BUILDING_PALETTE.length])); nt++;
    } else {                                                                      // floating block -> BOTH a bird flock AND a drone patrol, always, not a random pick between them
      const hv = obsHash01(x0, y0, z0);
      const anchor = T(cx, cy, z0 + h * 0.35);                                  // biased toward the bottom of the box, closer to where the player actually flies
      const birdKey = keyFor(cx, cy, z0) + '_bird';
      used.add(birdKey);
      const birdEntry = poolEntry(birdKey, Math.floor(hv * 3), hv);             // 0/1/2 bird species
      birdEntry.anchor = anchor; birdEntry.visible = true;

      const droneKey = keyFor(cx, cy, z0) + '_drone';
      used.add(droneKey);
      const droneEntry = poolEntry(droneKey, 3, (hv + 0.5) % 1);
      droneEntry.anchor = anchor; droneEntry.visible = true;

      const markerKey = keyFor(cx, cy, z0);
      usedMarkers.add(markerKey);
      let mk = hazardMarkers.get(markerKey);
      if (!mk) { mk = makeHazardMarker(); hazardMarkers.set(markerKey, mk); }
      mk.group.position.copy(anchor); mk.group.visible = true;
    }
  }
  for (const e of obstacleByKey.values()) if (!used.has(e.key)) e.visible = false;
  for (const [k, mk] of hazardMarkers) if (!usedMarkers.has(k)) mk.group.visible = false;
  towerMesh.count = nt; towerMesh.instanceMatrix.needsUpdate = true; if (towerMesh.instanceColor) towerMesh.instanceColor.needsUpdate = true;
}

function animateObstacles(tSec, dt) {
  for (const e of obstacleByKey.values()) {
    for (const m of e.members) {
      if (!e.visible) { m.group.visible = false; continue; }
      m.group.visible = true;
      m.heading += (Math.random() - 0.5) * m.turnRate * dt * 2;                 // free wander: a random walk of heading, not any fixed circle/pattern
      const dist = Math.hypot(m.localX, m.localZ);
      if (dist > m.leash) {                                                     // drifted too far from its flock -- steer back toward the anchor
        const toHome = Math.atan2(-m.localZ, -m.localX);
        m.heading += (((toHome - m.heading + Math.PI * 3) % (Math.PI * 2)) - Math.PI) * 0.06;
        if (dist > m.leash * 1.8) { m.localX = 0; m.localZ = 0; }               // respawn: if it somehow got lost, snap it straight back to the anchor
      }
      m.localX += Math.cos(m.heading) * m.speed * dt;
      m.localZ += Math.sin(m.heading) * m.speed * dt;
      const oy = Math.sin(tSec * m.bobSpeed + m.bobPhase) * m.bobAmp;
      m.group.position.set(e.anchor.x + m.localX, e.anchor.y + oy, e.anchor.z + m.localZ);
      m.group.rotation.y = -m.heading;                                          // face the direction it's actually travelling
      if (m.rotors) { for (const r of m.rotors) r.rotation.y += 1.3; }
      else { const a = 0.55 * Math.sin(tSec * m.flapHz * Math.PI * 2 + m.flapPhase); m.wingL.rotation.x = a; m.wingR.rotation.x = a; }
    }
  }
  for (const mk of hazardMarkers.values()) {                                     // one marker per hazard spot (not per flyer): a stack of 3 red rings, same size as the goal ring, at the fixed anchor
    if (!mk.group.visible) continue;
    const pulse = 1 + 0.15 * Math.sin(tSec * 1.8 + mk.phase);
    for (const r of mk.rings) { r.scale.setScalar(14 * pulse); r.rotation.z += 0.015; }
  }
}

// ── the drone ──────────────────────────────────────────────────────────────────
const drone = new THREE.Group();          // world position + yaw
const tilt = new THREE.Group();           // pitch / roll
drone.add(tilt); scene.add(drone);
const bodyMat = new THREE.MeshStandardMaterial({ color: 0x2f7bff, metalness: 0.5, roughness: 0.35 });
const dark = new THREE.MeshStandardMaterial({ color: 0x14181f, metalness: 0.4, roughness: 0.5 });
tilt.add(new THREE.Mesh(new THREE.BoxGeometry(9, 3, 6), bodyMat));
const pod = new THREE.Mesh(new THREE.SphereGeometry(2.2, 12, 8), dark); pod.position.set(5, -0.6, 0); tilt.add(pod);   // camera pod (nose)
const rotors = [];
for (const [ax, az] of [[7, 7], [7, -7], [-7, 7], [-7, -7]]) {
  const arm = new THREE.Mesh(new THREE.BoxGeometry(Math.hypot(ax, az) * 1.05, 0.9, 1), dark);
  arm.position.set(ax / 2, 0.2, az / 2); arm.rotation.y = -Math.atan2(az, ax); tilt.add(arm);
  const rotor = new THREE.Mesh(new THREE.CylinderGeometry(3.8, 3.8, 0.25, 20),
    new THREE.MeshBasicMaterial({ color: 0xbfd8ff, transparent: true, opacity: 0.45 }));
  rotor.position.set(ax, 1.4, az); tilt.add(rotor); rotors.push(rotor);
}
const ledF = new THREE.Mesh(new THREE.SphereGeometry(0.9, 8, 6), new THREE.MeshBasicMaterial({ color: 0x47ff8c })); ledF.position.set(4.6, 1.8, 0); tilt.add(ledF);
const ledB = new THREE.Mesh(new THREE.SphereGeometry(0.9, 8, 6), new THREE.MeshBasicMaterial({ color: 0xff4d4d })); ledB.position.set(-4.6, 1.8, 0); tilt.add(ledB);
const shadow = new THREE.Mesh(new THREE.CircleGeometry(9, 20), new THREE.MeshBasicMaterial({ color: 0x000000, transparent: true, opacity: 0.35 }));
shadow.rotation.x = -Math.PI / 2; scene.add(shadow);

// ── goal marker ────────────────────────────────────────────────────────────────
const goalGroup = new THREE.Group(); scene.add(goalGroup);
const goalRing = new THREE.Mesh(new THREE.TorusGeometry(1, 0.06, 10, 48), new THREE.MeshBasicMaterial({ color: 0x47ff8c }));
const goalCore = new THREE.Mesh(new THREE.SphereGeometry(5, 14, 10), new THREE.MeshBasicMaterial({ color: 0x9dffc4 }));
const goalBeam = new THREE.Mesh(new THREE.CylinderGeometry(1.2, 1.2, 1, 8, 1, true),
  new THREE.MeshBasicMaterial({ color: 0x47ff8c, transparent: true, opacity: 0.25, depthWrite: false }));
goalGroup.add(goalRing, goalCore, goalBeam);

// ── sensors + trail ────────────────────────────────────────────────────────────
const NS = 28;
const sensorLines = new THREE.LineSegments(new THREE.BufferGeometry(), new THREE.LineBasicMaterial({ vertexColors: true }));
sensorLines.geometry.setAttribute('position', new THREE.BufferAttribute(new Float32Array(NS * 6), 3));
sensorLines.geometry.setAttribute('color', new THREE.BufferAttribute(new Float32Array(NS * 6), 3));
sensorLines.frustumCulled = false; scene.add(sensorLines);
const TRAIL_N = 400, trailPos = new Float32Array(TRAIL_N * 3);
const trail = new THREE.Line(new THREE.BufferGeometry(), new THREE.LineBasicMaterial({ color: 0xffd76a, transparent: true, opacity: 0.8 }));
trail.geometry.setAttribute('position', new THREE.BufferAttribute(trailPos, 3)); trail.frustumCulled = false; scene.add(trail);
let trailCount = 0;

// ── state ─────────────────────────────────────────────────────────────────────
let last = null;
const pose = { x: 0, y: 0, z: 80, yaw: 0 }, target = { x: 0, y: 0, z: 80, yaw: 0 };
let prevGoals = 0, camMode = 'chase', roll = 0, pitch = 0, lastYaw = 0;
const camPos = new THREE.Vector3(-80, 60, 0), camLook = new THREE.Vector3();
const _fwd = new THREE.Vector3(), _want = new THREE.Vector3(), _lookT = new THREE.Vector3(), _up22 = new THREE.Vector3(0, 22, 0);      // reused every frame (no new objects to collect)
const lerp = (a, b, t) => a + (b - a) * t;
const angDiff = (a, b) => ((b - a + 540) % 360) - 180;

// ── websocket ─────────────────────────────────────────────────────────────────
const ws = new WebSocket(`${location.protocol === 'https:' ? 'wss' : 'ws'}://${location.host}/ws`);
ws.onmessage = (ev) => {
  const msg = JSON.parse(ev.data);
  if (msg.type === 'state') {
    if (last) { const dx = msg.drone.x - target.x, dy = msg.drone.y - target.y, dz = msg.drone.z - target.z;
      if (dx * dx + dy * dy + dz * dz > 250 * 250) { Object.assign(pose, msg.drone); trailCount = 0; } }   // respawn: snap, restart trail
    last = msg;
    target.x = msg.drone.x; target.y = msg.drone.y; target.z = msg.drone.z; target.yaw = msg.drone.yaw;
    if (msg.goals > prevGoals) goalBurst(); prevGoals = msg.goals;
    hud(msg);
    updateReplayUI(msg.replay);
  } else if (msg.type === 'obstacles') {
    setObstacles(msg.boxes);
  } else if (msg.type === 'replays') {
    fillReplayList(msg.files, msg.error);
  } else if (msg.type === 'targets') {
    setTargets(msg);
  } else if (msg.type === 'camera') {
    showCamera(msg);
  }
};
ws.onopen = () => send({ type: 'control', action: 'list_replays' });
function fillReplayList(files, error) {
  const sel = $('replaySel'), keep = sel.value; sel.innerHTML = '';
  for (const f of files) { const o = document.createElement('option'); o.value = f.name; o.textContent = f.name + '  (' + f.mb + ' MB)'; sel.appendChild(o); }
  if (keep) sel.value = keep;
  if (error) $('replayL').textContent = error;
  else if (!files.length) $('replayL').textContent = 'No recordings yet. Put replay.jsonl files into web3d_drones_edge/logs/ and press Refresh.';
}
let seeking = false;
function updateReplayUI(r) {
  const el = $('replayL'), seek = $('replaySeek');
  if (!r) { if (el.dataset.on) { el.textContent = 'Live'; delete el.dataset.on; } return; }
  el.dataset.on = '1';
  el.textContent = 'REPLAY ' + r.name + ' - step ' + r.pos + ' / ' + r.n + (r.paused ? ' (paused)' : '');
  if (!seeking) seek.value = Math.round(1000 * r.pos / Math.max(1, r.n - 1));
}
const send = (o) => { if (ws.readyState === WebSocket.OPEN) ws.send(JSON.stringify(o)); };
window.ctrl = (action, value) => send({ type: 'control', action, value });

let burst = 0;
function goalBurst() { burst = 1; }

// ── HUD ───────────────────────────────────────────────────────────────────────
// readable names for the drop-downs: the program keeps using the short names (gale_dusty), the page shows "Gale with dusty sensors"
const PRETTY = { calm: 'Calm', breeze: 'Breeze', windy: 'Windy', gale: 'Gale', gale_dusty: 'Gale with dusty sensors' };
const pretty = (t) => PRETTY[t] || (t.charAt(0).toUpperCase() + t.slice(1));
const fill = (id, items, current) => {
  const el = $(id); if (!el || el.dataset.n === String(items.length)) { if (el && el.value !== current) el.value = current; return; }
  el.innerHTML = ''; el.dataset.n = String(items.length);
  for (const it of items) { const o = document.createElement('option'); o.value = it; o.textContent = pretty(it); el.appendChild(o); }
  el.value = current;
};
function hud(s) {
  const w = s.wind, ws = Math.hypot(w[0], w[1]), d = s.drone;
  $('windL').textContent = ws.toFixed(1); $('windDirL').textContent = ws < 0.05 ? 'still' : ['E', 'NE', 'N', 'NW', 'W', 'SW', 'S', 'SE'][Math.round(((Math.atan2(w[1], w[0]) * 180 / Math.PI + 360) % 360) / 45) % 8] + ' (blowing to)';
  $('battL').textContent = s.battery.toFixed(0) + '%';
  $('battL').style.color = s.battery < 20 ? '#ff5a5a' : s.battery < 50 ? '#ffc94d' : '#47ff8c';
  $('goalsL').textContent = s.goals; $('totalGoalsL').textContent = s.totalGoals; $('stepsL').textContent = s.steps;
  $('gpkL').textContent = s.goalsPer1k; $('cpkL').textContent = s.collPer1k; $('totalCollisionsL').textContent = s.totalCollisions;
  $('epgL').textContent = s.energyPerGoal === null ? '-' : s.energyPerGoal;
  $('gsL').textContent = s.groundSpeed; $('altL').textContent = d.z.toFixed(0);
  $('respawnsL').textContent = s.respawns;
  const t = s.telemetry; $('tmL').textContent = t.frames + ' frames, ' + (t.bytes / 1024).toFixed(1) + ' KiB, CRC errors ' + t.crcErrors + ', lost ' + t.lost;
  $('pauseBtn').textContent = s.paused ? 'Resume' : 'Pause';
  fill('weatherSel', s.weathers, s.weather); fill('brainSel', s.brains, s.brain);
  ['day', 'dusk', 'night'].forEach((k) => $('light_' + k).classList.toggle('active', s.light === k));
  $('sensorSel').value = s.sensorSet;
  env.weather = s.weather; env.light = s.light; env.wind = w;
}

// ── collapsible drawers (state remembered per browser; closed by default) ──
document.querySelectorAll('details.drawer').forEach((d) => {
  const key = 'drawer_' + d.dataset.key;
  try { const v = localStorage.getItem(key); d.open = v === null ? d.hasAttribute('open') : v === '1'; } catch (_) {}
  d.addEventListener('toggle', () => { try { localStorage.setItem(key, d.open ? '1' : '0'); } catch (_) {} });
});

// ── controls ─────────────────────────────────────────────────────────────────
window.snapCar = () => { Object.assign(pose, target); };      // used by record_3d.py (frame-exact capture)
$('replayPlay').onclick = () => { const v = $('replaySel').value; if (v) ctrl('replay_start', v); };
$('replaySeek').onpointerdown = () => { seeking = true; };
$('replaySeek').onpointerup = () => { seeking = false; };
$('replaySeek').oninput = () => ctrl('replay_seek', $('replaySeek').value / 1000);
$('pauseBtn').onclick = () => ctrl(last && last.paused ? 'resume' : 'pause');
$('speed').oninput = (e) => { const v = parseFloat(e.target.value); $('speedL').textContent = v.toFixed(1) + 'x'; ctrl('speed', v); };
for (const [id, mode] of [['camChase', 'chase'], ['camOrbit', 'orbit'], ['camTop', 'top']]) {
  $(id).onclick = () => { camMode = mode; ['camChase', 'camOrbit', 'camTop'].forEach(i => $(i).classList.toggle('active', i === id));
    controls.enabled = mode === 'orbit'; };
}

// ── targets on the ground: vehicles, people, trees, hot rocks ─────────────────────
const env = { weather: 'calm', light: 'day', wind: [0, 0, 0] };
const MAXT = 400;
const hash01 = (x, y) => { const v = Math.sin(x * 12.9898 + y * 78.233) * 43758.5453; return v - Math.floor(v); };   // same object, same look, every time
// cars: a body, a dark glass cabin and four wheels. people: a coloured body (capsule) and a head
const CAR_COLORS = [0xd23c3c, 0x2f6fd6, 0xe8c23a, 0xf0f0f0, 0x38a86a, 0x2b2f38, 0xd9772b];
const vehicleMesh = new THREE.InstancedMesh(new THREE.BoxGeometry(10, 2.4, 4.6), new THREE.MeshStandardMaterial({ color: 0xffffff, roughness: 0.4, metalness: 0.3 }), MAXT);
const cabinMesh = new THREE.InstancedMesh(new THREE.BoxGeometry(5.2, 2, 4), new THREE.MeshStandardMaterial({ color: 0x1c2733, roughness: 0.2, metalness: 0.6 }), MAXT);
const wheelGeo = new THREE.CylinderGeometry(1.15, 1.15, 0.9, 12); wheelGeo.rotateX(Math.PI / 2);
const wheelMesh = new THREE.InstancedMesh(wheelGeo, new THREE.MeshStandardMaterial({ color: 0x111111, roughness: 0.9 }), MAXT * 4);
const SHIRTS = [0xe04848, 0x3f7dd6, 0xf0c23a, 0x49b36a, 0xffffff, 0xa45ad6];
const personMesh = new THREE.InstancedMesh(new THREE.CapsuleGeometry(0.8, 2.2, 4, 8), new THREE.MeshStandardMaterial({ color: 0xffffff, roughness: 0.8 }), MAXT);
const headMesh = new THREE.InstancedMesh(new THREE.SphereGeometry(0.75, 10, 8), new THREE.MeshStandardMaterial({ color: 0xe0b08a, roughness: 0.8 }), MAXT);
// trees: pines (a brown trunk with 3 or 4 stacked green cones),
// each with a soft shadow on the ground
const trunkMesh = new THREE.InstancedMesh(new THREE.CylinderGeometry(0.16, 0.3, 1, 7), new THREE.MeshStandardMaterial({ color: 0x5a3f2a, roughness: 0.95 }), MAXT);
const coneMesh = new THREE.InstancedMesh(new THREE.ConeGeometry(1, 1, 9), new THREE.MeshStandardMaterial({ color: 0xffffff, roughness: 0.9 }), MAXT * 4);
const shadowGeo = new THREE.CircleGeometry(1, 16); shadowGeo.rotateX(-Math.PI / 2);
const shadowMesh = new THREE.InstancedMesh(shadowGeo, new THREE.MeshBasicMaterial({ color: 0x000000, transparent: true, opacity: 0.22, depthWrite: false }), MAXT);
const rockMesh = new THREE.InstancedMesh(new THREE.SphereGeometry(1, 8, 6), new THREE.MeshStandardMaterial({ color: 0x8a8a8a, roughness: 1 }), MAXT);
for (const m of [vehicleMesh, cabinMesh, wheelMesh, personMesh, headMesh, trunkMesh, coneMesh, shadowMesh, rockMesh]) { m.count = 0; m.frustumCulled = false; scene.add(m); }
const _q = new THREE.Quaternion(), _e = new THREE.Euler(), _sc = new THREE.Vector3(), _v = new THREE.Vector3();
function setTargets(t) {
  let i = 0, w = 0;
  const put = (mesh, k, base, q, dx, dy, dz, sx, sy, sz) => { _v.set(dx, dy, dz).applyQuaternion(q); mesh.setMatrixAt(k, _m.compose(base.clone().add(_v), q, _sc.set(sx, sy, sz))); };
  for (const [x, y, a] of t.vehicles.slice(0, MAXT)) {
    const base = T(x, y, 0); _q.setFromEuler(_e.set(0, a, 0));
    put(vehicleMesh, i, base, _q, 0, 1.9, 0, 1, 1, 1);
    vehicleMesh.setColorAt(i, _c.set(CAR_COLORS[Math.floor(hash01(x, y) * CAR_COLORS.length)]));
    put(cabinMesh, i, base, _q, -0.6, 3.6, 0, 1, 1, 1);
    for (const [wx, wz] of [[3.1, 2.3], [3.1, -2.3], [-3.1, 2.3], [-3.1, -2.3]]) put(wheelMesh, w++, base, _q, wx, 1.15, wz, 1, 1, 1);
    i++;
  }
  vehicleMesh.count = cabinMesh.count = i; wheelMesh.count = w;
  for (const m of [vehicleMesh, cabinMesh, wheelMesh]) m.instanceMatrix.needsUpdate = true;
  if (vehicleMesh.instanceColor) vehicleMesh.instanceColor.needsUpdate = true; i = 0;
  for (const [x, y] of t.people.slice(0, MAXT)) {
    const base = T(x, y, 0); _q.identity();
    put(personMesh, i, base, _q, 0, 2.3, 0, 1, 1, 1);
    personMesh.setColorAt(i, _c.set(SHIRTS[Math.floor(hash01(y, x) * SHIRTS.length)]));
    put(headMesh, i, base, _q, 0, 4.95, 0, 1, 1, 1);
    i++;
  }
  personMesh.count = headMesh.count = i; personMesh.instanceMatrix.needsUpdate = headMesh.instanceMatrix.needsUpdate = true;
  if (personMesh.instanceColor) personMesh.instanceColor.needsUpdate = true; i = 0;

  // trees: pines. r is the crown radius that the onboard camera image uses. Each tree gets 3 or 4 stacked cones, its own height and shade of green.
  let nt = 0, nc = 0;
  for (const [x, y, r] of t.trees.slice(0, MAXT)) {
    const h = hash01(x, y), h2 = hash01(y, x), h3 = hash01(x + 3, y + 5);
    _q.identity();
    shadowMesh.setMatrixAt(nt, _m.compose(T(x, y, 0.15), _q, _sc.set(r * 1.5, 1, r * 1.5)));
    const trunkH = r * (1.0 + h2 * 0.5), tiers = h < 0.5 ? 3 : 4, tall = 1.0 + h3 * 0.4;
    trunkMesh.setMatrixAt(nt, _m.compose(T(x, y, trunkH / 2), _q, _sc.set(r * 0.8, trunkH, r * 0.8)));
    for (let k = 0; k < tiers; k++) {
      const rad = r * (1.3 - (0.9 / tiers) * k) , hh = r * 1.5 * tall;
      coneMesh.setMatrixAt(nc, _m.compose(T(x, y, trunkH * 0.5 + k * r * 0.85 * tall + hh / 2), _q, _sc.set(rad, hh, rad)));
      coneMesh.setColorAt(nc, _c.setHSL(0.35 + h3 * 0.04, 0.5, 0.15 + 0.05 * k + h2 * 0.03)); nc++;
    }
    nt++;
  }
  trunkMesh.count = shadowMesh.count = nt; coneMesh.count = nc;
  for (const m of [trunkMesh, shadowMesh, coneMesh]) m.instanceMatrix.needsUpdate = true;
  if (coneMesh.instanceColor) coneMesh.instanceColor.needsUpdate = true; i = 0;
  for (const [x, y, r] of t.rocks.slice(0, MAXT)) rockMesh.setMatrixAt(i++, _m.compose(T(x, y, r * 0.5), _q.identity(), _sc.set(r, r * 0.6, r)));
  rockMesh.count = i; rockMesh.instanceMatrix.needsUpdate = true;
}

// ── the drone's camera panels and the detector's result ──────────────────────────
function showCamera(m) {
  $('camImg').src = 'data:image/png;base64,' + m.png;
  for (let i = 0; i < 3; i++) {                                   // found / really there per class, green when they agree, amber when the detector missed or invented one
    $('fnd' + i).textContent = m.found[i]; $('tru' + i).textContent = m.truth[i];
    $('fnd' + i).className = m.found[i] === m.truth[i] ? 'ok' : 'miss';
  }
  const sensors = { fused: 'RGB + thermal + height', rgb: 'RGB only', thermal: 'Thermal only', lidar: 'Height only', 'thermal+lidar': 'Thermal + height' }[last ? last.sensorSet : 'fused'];
  $('camL').textContent = 'Altitude ' + m.alt + '  ·  Light ' + pretty(m.light) + '  ·  Detector uses ' + sensors;
}

// ── wind: streaks that blow through the air around the drone, and an arrow ────────
const STREAKS = 260, RANGE = 260;
const streakPos = new Float32Array(STREAKS * 6), streakBase = new Float32Array(STREAKS * 3);
for (let i = 0; i < STREAKS; i++) { streakBase[i * 3] = (Math.random() - 0.5) * RANGE * 2; streakBase[i * 3 + 1] = (Math.random() - 0.3) * RANGE; streakBase[i * 3 + 2] = (Math.random() - 0.5) * RANGE * 2; }
const streaks = new THREE.LineSegments(new THREE.BufferGeometry(), new THREE.LineBasicMaterial({ color: 0xffffff, transparent: true, opacity: 0.55 }));
streaks.geometry.setAttribute('position', new THREE.BufferAttribute(streakPos, 3)); streaks.frustumCulled = false; scene.add(streaks);
const windArrow = new THREE.ArrowHelper(new THREE.Vector3(1, 0, 0), new THREE.Vector3(), 30, 0x5ad1ff, 8, 5); scene.add(windArrow);
let windPhase = 0;
function updateWind(dt) {
  const wx = env.wind[0], wy = env.wind[1], wz = env.wind[2], sp = Math.hypot(wx, wy);
  windPhase += dt;
  const flow = T(wx, wy, wz).multiplyScalar(60);                       // three-space drift per second
  streaks.visible = sp > 0.05;
  if (streaks.visible) {
    const o = drone.position, len = 0.5 + sp * 0.7;
    for (let i = 0; i < STREAKS; i++) {
      const bx = streakBase[i * 3], by = streakBase[i * 3 + 1], bz = streakBase[i * 3 + 2];
      let x = ((bx + flow.x * windPhase + RANGE * 3) % (RANGE * 2)) - RANGE, z = ((bz + flow.z * windPhase + RANGE * 3) % (RANGE * 2)) - RANGE;
      const y = by + flow.y * windPhase * 0 ;
      streakPos.set([o.x + x, o.y + y, o.z + z, o.x + x + flow.x * 0.05 * len, o.y + y + flow.y * 0.05 * len, o.z + z + flow.z * 0.05 * len], i * 6);
    }
    streaks.geometry.attributes.position.needsUpdate = true;
  }
  windArrow.visible = sp > 0.05;
  if (windArrow.visible) {
    windArrow.position.copy(drone.position).add(_up26);
    windArrow.setDirection(T(wx, wy, wz).normalize()); windArrow.setLength(14 + sp * 22, 8, 5);
  }
}

// ── look: light level and weather change the sky ──────────────────────────────────
const LOOK = { day: [0xc4e4ff, 0x0a3d91, 0.95, 1.6], dusk: [0xf0a070, 0x2b2a6e, 0.55, 0.7], night: [0x0a1226, 0x02040c, 0.22, 0.12] };
const skyTop = sky.material.uniforms.top.value, skyBottom = sky.material.uniforms.bottom.value, hemi = scene.children.find((c) => c.isHemisphereLight);
const _tmp = new THREE.Color(), _greyH = new THREE.Color(0x7d8794), _greyT = new THREE.Color(0x4a5566), _up26 = new THREE.Vector3(0, 26, 0);
function updateLook() {
  const [hor, top, hemiI, sunI] = LOOK[env.light] || LOOK.day;
  const grey = { calm: 0, breeze: 0.05, windy: 0.25, gale: 0.5, gale_dusty: 0.6 }[env.weather] || 0;
  _tmp.set(hor).lerp(_greyH, grey); skyBottom.lerp(_tmp, 0.05); scene.fog.color.copy(skyBottom);
  _tmp.set(top).lerp(_greyT, grey); skyTop.lerp(_tmp, 0.05);
  hemi.intensity = lerp(hemi.intensity, hemiI * (1 - 0.3 * grey), 0.05); sun.intensity = lerp(sun.intensity, sunI * (1 - 0.4 * grey), 0.05);
  scene.fog.far = 2600 - grey * 1500;
}

// ── render loop ──────────────────────────────────────────────────────────────
function resize() { const w = canvas.clientWidth, h = canvas.clientHeight; renderer.setSize(w, h, false); camera.aspect = w / h; camera.updateProjectionMatrix(); }
addEventListener('resize', resize); resize();
window.edgeDebug = { camera, trunkMesh, coneMesh, vehicleMesh, personMesh, rockMesh, freeCam: false };
const clock = new THREE.Clock(); let fpsFrames = 0, fpsT0 = performance.now();

function frame() {
  requestAnimationFrame(frame);
  const dt = Math.min(0.05, clock.getDelta());
  fpsFrames++; const tNow = performance.now();
  if (tNow - fpsT0 >= 500) { $('fpsL').textContent = Math.round(fpsFrames * 1000 / (tNow - fpsT0)); fpsFrames = 0; fpsT0 = tNow; }

  // smooth the drone toward the latest server pose
  pose.x = lerp(pose.x, target.x, 0.25); pose.y = lerp(pose.y, target.y, 0.25); pose.z = lerp(pose.z, target.z, 0.25);
  pose.yaw += angDiff(pose.yaw, target.yaw) * 0.25;
  const yr = pose.yaw * Math.PI / 180;
  drone.position.copy(T(pose.x, pose.y, pose.z)); drone.rotation.y = yr;
  const turn = angDiff(lastYaw, pose.yaw); lastYaw = pose.yaw;
  roll = lerp(roll, THREE.MathUtils.clamp(turn * 0.05, -0.6, 0.6), 0.2);          // bank into turns
  const vz = last ? last.drone.vz : 0;
  pitch = lerp(pitch, THREE.MathUtils.clamp(-vz * 0.06, -0.5, 0.5), 0.2);         // nose up when climbing
  tilt.rotation.set(roll, 0, pitch);
  for (const r of rotors) r.rotation.y += 1.6;
  animateObstacles(performance.now() / 1000, dt);
  bodyMat.color.set(last && last.drone.colliding ? 0xff4d4d : 0x2f7bff);
  shadow.position.set(drone.position.x, 0.5, drone.position.z);
  shadow.scale.setScalar(1 + Math.max(0, pose.z) / 250); shadow.material.opacity = Math.max(0.08, 0.4 - pose.z / 700);

  if (last) {
    // goal
    const g = last.goal, gr = last.goalRadius || 30;
    goalGroup.position.copy(T(g.x, g.y, g.z));
    const pulse = 1 + 0.06 * Math.sin(performance.now() / 250) + burst * 0.5; burst = Math.max(0, burst - dt * 2);
    goalRing.scale.setScalar(gr * pulse); goalRing.rotation.x = Math.PI / 2 + Math.sin(performance.now() / 900) * 0.25; goalRing.rotation.y += 0.01;
    goalBeam.scale.set(1, g.z + 2000, 1); goalBeam.position.y = -g.z / 2 + (g.z + 2000) / 2 - 0; goalBeam.position.y = (2000 - g.z) / 2;

    // sensors (segments from the drone to each ray's end, coloured by proximity)
    sensorLines.visible = $('showSensors').checked;
    if (sensorLines.visible) {
      const pa = sensorLines.geometry.attributes.position, ca = sensorLines.geometry.attributes.color;
      const o = drone.position;
      last.sensors.forEach((s, i) => {
        const e = T(s[0], s[1], s[2]), heat = Math.min(1, s[3] * 2.5);
        pa.setXYZ(i * 2, o.x, o.y, o.z); pa.setXYZ(i * 2 + 1, e.x, e.y, e.z);
        const r = lerp(0.2, 1, heat), gch = lerp(1, 0.2, heat), b = 0.3;
        ca.setXYZ(i * 2, r, gch, b); ca.setXYZ(i * 2 + 1, r, gch, b);
      });
      pa.needsUpdate = true; ca.needsUpdate = true;
    }
    // trail
    trail.visible = $('showTrail').checked;
    if (trail.visible) {
      if (trailCount < TRAIL_N) trailCount++; else trailPos.copyWithin(0, 3);
      const k = trailCount - 1; trailPos[k * 3] = drone.position.x; trailPos[k * 3 + 1] = drone.position.y; trailPos[k * 3 + 2] = drone.position.z;
      trail.geometry.setDrawRange(0, trailCount); trail.geometry.attributes.position.needsUpdate = true;
    }
  }

  // infinite ground + sky + clouds follow the drone
  ground.position.set(Math.round(drone.position.x / TILE) * TILE, -0.05, Math.round(drone.position.z / TILE) * TILE);
  sky.position.copy(camera.position);
  cloudBase.forEach((b, i) => {
    const wx = ((b[0] - drone.position.x) % CLOUD_BOX + CLOUD_BOX) % CLOUD_BOX - CLOUD_BOX / 2;
    const wz = ((b[2] - drone.position.z) % CLOUD_BOX + CLOUD_BOX) % CLOUD_BOX - CLOUD_BOX / 2;
    clouds.children[i].position.set(drone.position.x + wx, b[1], drone.position.z + wz);
  });

  // camera
  const fwd = _fwd.set(Math.cos(yr), 0, -Math.sin(yr));
  if (window.edgeDebug && window.edgeDebug.freeCam) { renderer.render(scene, camera); return; }
  if (camMode === 'chase') {
    const want = _want.copy(drone.position).addScaledVector(fwd, -55).add(_up22);
    camPos.lerp(want, 0.12); camLook.lerp(_lookT.copy(drone.position).addScaledVector(fwd, 30), 0.2);
    camera.position.copy(camPos); camera.lookAt(camLook);
  } else if (camMode === 'top') {
    camera.position.set(drone.position.x, drone.position.y + 520, drone.position.z + 1); camera.lookAt(drone.position);
  } else { controls.target.lerp(drone.position, 0.2); controls.update(); }

  updateWind(dt); updateLook();
  renderer.render(scene, camera);
}
frame();


// ── automatic weather and light ─────────────────────────────────────────────────
const sendAutoWeather = () => ctrl('auto_weather', { on: $('autoW').checked, every: +$('autoWSec').value, mode: $('autoWMode').value });
const sendAutoLight = () => ctrl('auto_light', { on: $('autoL').checked, every: +$('autoLSec').value });
['autoW', 'autoWSec', 'autoWMode'].forEach((id) => { $(id).onchange = sendAutoWeather; });
['autoL', 'autoLSec'].forEach((id) => { $(id).onchange = sendAutoLight; });
setInterval(() => {                                  // the server is the truth (the setting is shared by every open page)
  if (!last || !last.auto) return;
  const a = last.auto, f = document.activeElement;
  if (f !== $('autoW')) $('autoW').checked = a.weather.on;
  if (f !== $('autoWSec')) $('autoWSec').value = a.weather.every;
  if (f !== $('autoWMode')) $('autoWMode').value = a.weather.mode;
  if (f !== $('autoL')) $('autoL').checked = a.light.on;
  if (f !== $('autoLSec')) $('autoLSec').value = a.light.every;
}, 500);


