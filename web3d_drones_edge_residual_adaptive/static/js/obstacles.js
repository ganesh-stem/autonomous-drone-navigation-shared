// Obstacles: city towers, the colour-shifting hazard gates and the roaming birds / patrol drones.

import * as THREE from 'three';
import { T, scene } from './core.js';
import { city, towerRects } from './city.js';
import { drone } from './drone.js';
import { goalRing, goalLook } from './goal.js';
import { beginTowers, addTower, endTowers } from './buildings.js';

// ── obstacles ──────────────────────────────────────────────────
// Towers (buildings.js) come in three real-world facade styles with their own paint colours; floating blocks are birds and patrol drones (below).
// ── floating obstacles: real birds (3 species) and small patrol drones, not boxes ──────────────────────────
// Real-world scale (a bird body is well under 1 sim unit = 1 m), positioned at the hazard box's centre. Reused
// via a position-keyed pool (obstacleByKey) instead of rebuilt every call, and animated (wing flap / rotor spin)
// once per frame in frame() below.
const _m = new THREE.Matrix4(), _oq = new THREE.Quaternion();

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

const hazardMarkers = new Map();                                                 // one marker per hazard SPOT (not per flyer), at the fixed anchor

// Gate silhouettes: each hazard spot gets one of these shapes (picked stably from its position hash), all on a unit circumradius.
const polyPts = (n, rot = Math.PI / 2) => Array.from({ length: n }, (_, i) => { const a = rot + (i / n) * Math.PI * 2; return new THREE.Vector2(Math.cos(a), Math.sin(a)); });
const starPts = (n, inner) => Array.from({ length: n * 2 }, (_, i) => { const a = Math.PI / 2 + (i / (n * 2)) * Math.PI * 2, r = i % 2 ? inner : 1; return new THREE.Vector2(Math.cos(a) * r, Math.sin(a) * r); });
const HAZARD_SHAPES = [
  { name: 'triangle', pts: polyPts(3), band: 0.80 },
  { name: 'diamond',  pts: polyPts(4), band: 0.86 },
  { name: 'pentagon', pts: polyPts(5), band: 0.88 },
  { name: 'hexagon',  pts: polyPts(6), band: 0.9 },
  { name: 'star',     pts: starPts(5, 0.46), band: 0.8 },
  { name: 'shuriken', pts: starPts(4, 0.36), band: 0.78 },
  { name: 'octagon',  pts: polyPts(8, Math.PI / 8), band: 0.9 },
];
const _hazardGeoCache = new Map();                                              // geometries are shared by every gate of the same shape
function hazardGeos(shape) {
  let g = _hazardGeoCache.get(shape.name);
  if (g) return g;
  const outline = (pts, k) => pts.map(p => p.clone().multiplyScalar(k));
  const frame = (outer, inner, depth) => {                                       // flat frame = outer outline with the inner outline cut out
    const sh = new THREE.Shape(outer); sh.holes.push(new THREE.Path(inner.slice().reverse()));
    const geo = new THREE.ExtrudeGeometry(sh, { depth, bevelEnabled: false }); geo.translate(0, 0, -depth / 2); return geo;
  };
  const field = new THREE.Shape(outline(shape.pts, 0.94)); field.holes.push(new THREE.Path(outline(shape.pts, 0.45).reverse()));
  // cumulative perimeter lengths, so the beads can travel along the outline instead of a circle
  const P = shape.pts, seg = [0];
  for (let i = 0; i < P.length; i++) seg.push(seg[i] + P[i].distanceTo(P[(i + 1) % P.length]));
  g = {
    main: frame(outline(P, 1), outline(P, shape.band), 0.07),
    halo: frame(outline(P, 1.1), outline(P, shape.band * 0.92), 0.04),
    inner: frame(outline(P, 0.64), outline(P, 0.64 * shape.band), 0.05),
    field: new THREE.ShapeGeometry(field),
    perim: { P, seg, total: seg[seg.length - 1] },
  };
  _hazardGeoCache.set(shape.name, g);
  return g;
}
function perimeterPoint(perim, u) {                                             // u in [0,1) -> point on the outline
  const d = (u % 1) * perim.total; let i = 0;
  while (i < perim.P.length - 1 && perim.seg[i + 1] < d) i++;
  const t = (d - perim.seg[i]) / (perim.seg[i + 1] - perim.seg[i] || 1);
  return perim.P[i].clone().lerp(perim.P[(i + 1) % perim.P.length], t);
}

function makeHazardMarker(hv) {                                                  // colour-shifting energy gate in one of several shapes -- same size/tilt/pulse as the goal ring, no centre dot, no beam
  const shape = HAZARD_SHAPES[Math.floor(hv * HAZARD_SHAPES.length) % HAZARD_SHAPES.length];
  const geos = hazardGeos(shape);
  const g = new THREE.Group();                                                   // positioned at the anchor and scaled to the goal ring's radius
  const tilt = new THREE.Group();                                                // copies the goal ring's tilt/wobble so the whole gate moves exactly like it
  const spin = new THREE.Group();                                                // slow spin about the gate's own axis
  g.add(tilt); tilt.add(spin);
  const glow = (op, side = THREE.FrontSide) => ({ transparent: true, opacity: op, blending: THREE.AdditiveBlending, depthWrite: false, side });   // FrontSide: an extruded frame has two caps, DoubleSide would add them together and blow out to white
  const mk = (geo, params) => new THREE.Mesh(geo, new THREE.MeshBasicMaterial(params));

  const frame = mk(geos.main, { color: 0xffffff, side: THREE.DoubleSide });
  const halo = mk(geos.halo, glow(0.16));                                        // soft outer glow
  const field = mk(geos.field, glow(0.09, THREE.DoubleSide));                                      // faint energy field between the frame and a hollow centre
  const inner = mk(geos.inner, glow(0.85));                                      // smaller copy of the shape that spins the other way
  spin.add(frame, halo, field); tilt.add(inner);

  const beads = [];                                                              // glowing crystals racing around the outline
  const beadGeo = new THREE.OctahedronGeometry(0.075, 0);
  for (let i = 0; i < 6; i++) { const bd = mk(beadGeo, { color: 0xffffff }); spin.add(bd); beads.push(bd); }

  scene.add(g);
  return { group: g, tilt, spin, frame, halo, field, inner, beads, perim: geos.perim, shape: shape.name, phase: Math.random() };
}

function animateHazardMarker(mk, tSec) {
  mk.group.scale.setScalar(goalLook.gateRadius);                                          // goal ring's radius * its gentle breathing; deliberately NOT the 'goal reached' burst
  mk.tilt.rotation.copy(goalRing.rotation);                                      // identical tilt, wobble and drift
  const hue = (tSec * 0.07 + mk.phase) % 1;                                      // slow full-spectrum colour cycle, each gate offset from the others
  const col = (dh, l = 0.6) => new THREE.Color().setHSL((hue + dh) % 1, 0.95, l);
  mk.frame.material.color.copy(col(0));
  mk.halo.material.color.copy(col(0.04));
  mk.field.material.color.copy(col(0.12, 0.5));
  mk.inner.material.color.copy(col(0.25, 0.65));
  mk.inner.rotation.z = -tSec * 0.9 + mk.phase * 6.28;
  mk.spin.rotation.z = tSec * 0.25;
  for (let i = 0; i < mk.beads.length; i++) {
    const p = perimeterPoint(mk.perim, tSec * 0.12 + i / mk.beads.length + mk.phase);
    mk.beads[i].position.set(p.x, p.y, 0);
    mk.beads[i].rotation.set(tSec * 2 + i, tSec * 1.5, 0);
    mk.beads[i].material.color.copy(col(0.5 + i / mk.beads.length * 0.5, 0.7));
    mk.beads[i].scale.setScalar(0.8 + 0.5 * Math.sin(tSec * 4 + i * 1.7));
  }
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
    speed: (kind === 3 ? 20 : 14) * (0.7 + mh * 0.6),                             // sim units / second
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

// ── the roaming birds and patrol drones of the simulation ──────────────────────────────────────────────────────────────────
// Live, every bird and every patrol drone is a small obstacle of the simulation that roams on its own path (flyers.py on the server). The server sends where each one is
// (15 messages a second) and it is drawn here, moving between messages with its velocity. The hazard gates are unchanged: they mark the floating blocks, which stay where they are.
// Without those messages (a replay, or --static-obstacles) the flyers wander at random around the gates, as they always did.
const liveFlyers = new Map();                                                    // id (spot + member) -> { group, ..., pos, vel, at }
let liveUntil = 0;                                                               // the server's flyers count as live until this time (seconds)
const isLive = () => performance.now() / 1000 < liveUntil;

function setFlyers(msg) {
  const now = performance.now() / 1000;
  liveUntil = now + 1.2;
  for (const [x0, y0, z0, member, kind, mh, w, d, h, cx, cy, cz, vx, vy, vz] of msg.f) {
    const key = keyFor(x0, y0, z0) + '_' + member;
    let f = liveFlyers.get(key);
    if (!f) {
      const built = kind === 3 ? makeNpcDrone(NPC_DRONE_COLORS[Math.floor(mh * NPC_DRONE_COLORS.length) % NPC_DRONE_COLORS.length]) : makeBird(BIRD_SPECIES[kind]);
      scene.add(built.group);
      f = { ...built, flapPhase: mh * 20, heading: 0, pos: new THREE.Vector3(), vel: new THREE.Vector3(), fresh: true };
      liveFlyers.set(key, f);
    }
    f.pos.copy(T(cx, cy, cz)); f.vel.set(vx, vz, -vy);                            // world (x, y, z) -> three (x, z, -y), also for the velocity
    f.at = now;
  }
}

function animateLiveFlyers(tSec, dt) {
  const now = performance.now() / 1000;
  for (const [key, f] of liveFlyers) {
    if (now - f.at > 1.2) { scene.remove(f.group); liveFlyers.delete(key); continue; }   // left the area around the drone
    const e = Math.min(0.25, Math.max(0, now - f.at));
    f.group.visible = true;
    f.group.position.copy(f.pos).addScaledVector(f.vel, e);
    const target = Math.atan2(f.vel.z, f.vel.x);                                  // face the way it is travelling
    if (f.fresh) { f.heading = target; f.fresh = false; }
    f.heading += _wrap(target - f.heading) * Math.min(1, 6 * dt);
    f.group.rotation.y = -f.heading;
    if (f.rotors) { for (const r of f.rotors) r.rotation.y += 1.3; }
    else { const a = 0.55 * Math.sin(tSec * f.flapHz * Math.PI * 2 + f.flapPhase); f.wingL.rotation.x = a; f.wingR.rotation.x = a; }
  }
}

function setObstacles(boxes) {
  towerRects.length = 0; beginTowers();
  const used = new Set();
  const usedMarkers = new Set();
  for (let i = 0; i < boxes.length; i++) {
    const [x0, y0, z0, x1, y1, z1] = boxes[i];
    const w = x1 - x0, d = y1 - y0, h = z1 - z0, cx = (x0 + x1) / 2, cy = (y0 + y1) / 2;
    if (z0 <= 0.5) {                                                              // tower = city-style building
      towerRects.push([x0, y0, x1, y1]);                                          // the city scenery keeps clear of towers
      _m.compose(T(cx, cy, (z0 + z1) / 2), _oq.identity(), new THREE.Vector3(w, h, d));
      addTower(_m, h, obsHash01(x0, y0, z0));
    } else {                                                                      // floating block -> BOTH a bird flock AND a drone patrol, always, not a random pick between them
      const hv = obsHash01(x0, y0, z0);
      const anchor = T(cx, cy, z0 + h * 0.35);                                  // biased toward the bottom of the box, closer to where the player actually flies
      if (!isLive()) {                                                          // no live flyers from the server: random wanderers around the gate, as before
        const birdKey = keyFor(cx, cy, z0) + '_bird';
        used.add(birdKey);
        const birdEntry = poolEntry(birdKey, Math.floor(hv * 3), hv);           // 0/1/2 bird species
        birdEntry.anchor = anchor; birdEntry.visible = true;

        const droneKey = keyFor(cx, cy, z0) + '_drone';
        used.add(droneKey);
        const droneEntry = poolEntry(droneKey, 3, (hv + 0.5) % 1);
        droneEntry.anchor = anchor; droneEntry.visible = true;
      }

      const markerKey = keyFor(cx, cy, z0);
      usedMarkers.add(markerKey);
      let mk = hazardMarkers.get(markerKey);
      if (!mk) { mk = makeHazardMarker(hv); hazardMarkers.set(markerKey, mk); }
      mk.group.position.copy(anchor); mk.group.visible = true;
    }
  }
  for (const e of obstacleByKey.values()) if (!used.has(e.key)) e.visible = false;
  for (const [k, mk] of hazardMarkers) if (!usedMarkers.has(k)) mk.group.visible = false;
  endTowers();
  city.bx = NaN;                                                                 // tower footprints changed -> rebuild the scenery on the next frame
}

const ROAM_RADIUS = 1100;                                                       // flyers roam the whole area around the player (view range is ~4 chunks = 1000 units) and turn back / get recycled beyond this
const _wrap = (a) => ((a + Math.PI * 3) % (Math.PI * 2)) - Math.PI;

function animateObstacles(tSec, dt) {
  const dx0 = drone.position.x, dz0 = drone.position.z;
  const live = isLive();
  if (live || liveFlyers.size) animateLiveFlyers(tSec, dt);
  for (const [key, e] of obstacleByKey) {
    if (live) { for (const m of e.members) scene.remove(m.group); obstacleByKey.delete(key); continue; }   // the server's flyers replace the random ones
    // flyers roam freely in world space instead of orbiting their gate. A spot that scrolled out of view keeps its flyers
    // flying (no pop-out); they are only removed once they have wandered beyond the roam radius.
    let cx = 0, cz = 0, n = 0;                                                  // flock centroid, for loose cohesion
    for (const m of e.members) if (m.wx !== undefined) { cx += m.wx; cz += m.wz; n++; }
    if (n) { cx /= n; cz /= n; }
    let anyNear = false;
    for (const m of e.members) {
      if (m.wx === undefined) {                                                 // first frame: appear around the gate, then roam off on their own
        const r = Math.random() * m.leash, a = Math.random() * Math.PI * 2;
        m.wx = e.anchor.x + Math.cos(a) * r; m.wz = e.anchor.z + Math.sin(a) * r; m.wy = e.anchor.y;
        m.curve = 0; m.vy = 0;
      }
      m.curve += (-m.curve * 0.6 + (Math.random() - 0.5) * 4 * m.turnRate) * dt;  // slowly-changing turn rate that relaxes back to 0 -> smooth wandering paths, never a steady circle
      m.curve = Math.max(-0.7, Math.min(0.7, m.curve));
      m.heading += m.curve * dt;
      const dd = Math.hypot(m.wx - dx0, m.wz - dz0);
      if (dd > ROAM_RADIUS && e.visible) m.heading += _wrap(Math.atan2(dz0 - m.wz, dx0 - m.wx) - m.heading) * Math.min(1, 1.5 * dt);   // reached the edge of the roam area: turn back toward the player's region
      else if (n > 1 && dd <= ROAM_RADIUS) {                                    // loose flocking: drift toward the flock's centre only when straggling far behind
        const fd = Math.hypot(cx - m.wx, cz - m.wz);
        if (fd > 90) m.heading += _wrap(Math.atan2(cz - m.wz, cx - m.wx) - m.heading) * Math.min(1, 0.8 * dt);
      }
      m.wx += Math.cos(m.heading) * m.speed * dt;
      m.wz += Math.sin(m.heading) * m.speed * dt;
      m.vy += ((Math.random() - 0.5) * 6 - m.vy * 0.8) * dt;                    // gentle altitude wander, kept inside the flight band
      m.wy = Math.max(15, Math.min(300, m.wy + m.vy * dt * 4));
      const oy = Math.sin(tSec * m.bobSpeed + m.bobPhase) * m.bobAmp;
      m.group.visible = true;
      m.group.position.set(m.wx, m.wy + oy, m.wz);
      m.group.rotation.y = -m.heading;                                          // face the direction it's actually travelling
      if (dd < ROAM_RADIUS) anyNear = true;
      if (m.rotors) { for (const r of m.rotors) r.rotation.y += 1.3; }
      else { const a = 0.55 * Math.sin(tSec * m.flapHz * Math.PI * 2 + m.flapPhase); m.wingL.rotation.x = a; m.wingR.rotation.x = a; }
    }
    if (!e.visible && !anyNear) {                                               // orphaned spot whose flyers have all left the area: clean up
      for (const m of e.members) scene.remove(m.group);
      obstacleByKey.delete(key);
    }
  }
  for (const mk of hazardMarkers.values()) {                                     // animate each gate: colour cycle, spinning arcs, orbiting beads, goal-ring tilt/size
    if (mk.group.visible) animateHazardMarker(mk, tSec);
  }
}

export { setObstacles, setFlyers, animateObstacles };
