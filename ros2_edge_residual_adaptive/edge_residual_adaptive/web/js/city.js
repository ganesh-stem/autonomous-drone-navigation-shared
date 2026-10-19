// City ground: the road lattice texture plus per-block scenery (parks, fountains, gardens, playgrounds, courts ...), all visual only.
// A block is one sim chunk (250 x 250). Also answers "is this spot free ground?" for trees / rocks / people (city.blocked).

import * as THREE from 'three';
import { scene } from './core.js';

// ── city ground: real roads + parks, fountains, gardens, sports fields, playgrounds ... ─────────────────────────────
// Purely visual and deterministic per block (no sim change). A block is one sim chunk (250 x 250); roads run along block borders,
// and towers are always generated inside a single chunk, so they never sit on a road. Per-block scenery is drawn with
// instanced meshes and rebuilt only when the drone enters a new block or the tower list changes; towers also veto scenery.
const BLOCK = 250, ROAD_HALF = 11, WALK = 6;
const TILE = BLOCK, PLANE = 30000;                                               // PLANE is a multiple of BLOCK, and the plane is re-centred in whole blocks, so the road texture stays aligned with the chunk borders

function cityGroundTexture() {                                                   // one block: lawn in the middle; road, kerb, sidewalk, lane lines and zebra crossings on all four borders
  const S = 2048, u = S / BLOCK, R = ROAD_HALF * u, W = (ROAD_HALF + WALK) * u;
  const c = document.createElement('canvas'); c.width = c.height = S; const g = c.getContext('2d');
  let seed = 7; const rnd = () => (seed = (seed * 1664525 + 1013904223) >>> 0) / 4294967296;
  g.fillStyle = '#41763c'; g.fillRect(0, 0, S, S);
  for (let i = 0; i < 700; i++) {                                                // soft lawn variation
    g.fillStyle = `rgba(${40 + rnd() * 40 | 0},${95 + rnd() * 60 | 0},${45 + rnd() * 30 | 0},.10)`;
    g.beginPath(); g.arc(rnd() * S, rnd() * S, 20 + rnd() * 70, 0, 6.283); g.fill();
  }
  const bands = (w) => [[0, 0, w, S], [S - w, 0, w, S], [0, 0, S, w], [0, S - w, S, w]];
  g.fillStyle = '#b9b7ad'; for (const b of bands(W)) g.fillRect(...b);          // sidewalks
  g.fillStyle = '#2c2e33'; for (const b of bands(R)) g.fillRect(...b);           // asphalt
  g.save(); g.beginPath(); for (const b of bands(R)) g.rect(...b); g.clip();      // asphalt speckle
  for (let i = 0; i < 9000; i++) { const v = 36 + rnd() * 26 | 0; g.fillStyle = `rgba(${v},${v},${v + 4},.5)`; g.fillRect(rnd() * S, rnd() * S, 2 + rnd() * 3, 2 + rnd() * 3); }
  g.restore();
  g.fillStyle = '#8d8b83'; const k = 3, L = S - 2 * R;                           // kerbs (not across the intersections)
  g.fillRect(R, R, k, L); g.fillRect(S - R - k, R, k, L); g.fillRect(R, R, L, k); g.fillRect(R, S - R - k, L, k);
  g.fillStyle = '#e9c43c';                                                       // double yellow centre line on the block border (each tile draws its half)
  g.fillRect(2, R, 4, L); g.fillRect(S - 6, R, 4, L); g.fillRect(R, 2, L, 4); g.fillRect(R, S - 6, L, 4);
  g.fillStyle = '#e8e8e8'; const e = 9.4 * u;                                    // white edge lines
  g.fillRect(e, R, 3, L); g.fillRect(S - e - 3, R, 3, L); g.fillRect(R, e, L, 3); g.fillRect(R, S - e - 3, L, 3);
  const corner = () => {                                                         // zebra crossings on the two arms leaving one intersection corner
    for (let y = 0.5 * u; y < R - 0.5 * u; y += 2.1 * u) g.fillRect(R + 2 * u, y, 4 * u, 1.2 * u);
    for (let x = 0.5 * u; x < R - 0.5 * u; x += 2.1 * u) g.fillRect(x, R + 2 * u, 1.2 * u, 4 * u);
  };
  for (const [tx, ty, sx, sy] of [[0, 0, 1, 1], [S, 0, -1, 1], [0, S, 1, -1], [S, S, -1, -1]]) { g.save(); g.translate(tx, ty); g.scale(sx, sy); corner(); g.restore(); }
  const t = new THREE.CanvasTexture(c);
  t.colorSpace = THREE.SRGBColorSpace; t.wrapS = t.wrapT = THREE.RepeatWrapping; t.anisotropy = 8;
  return t;
}
const gridTex = cityGroundTexture(); gridTex.repeat.set(PLANE / TILE, PLANE / TILE);
const ground = new THREE.Mesh(new THREE.PlaneGeometry(PLANE, PLANE), new THREE.MeshLambertMaterial({ map: gridTex }));
ground.rotation.x = -Math.PI / 2;
scene.add(ground);

// Which kind of block (0 = plain lawn, 1..7 = the scenery themes): 1 + (2*bx + 3*by) mod 7 differs between every block and its 8 neighbours (the steps 2, 3, 2+3 and
// 2-3 are never a multiple of 7), so the same scenery never sits next to itself and the kinds are spread evenly. About one block in five is a plain lawn instead.
function themeFor(bx, by) {
  const u = ((Math.imul(bx, 73856093) ^ Math.imul(by, 19349663)) >>> 0) % 1000 / 1000;           // stable pseudo-random number per block
  if (u < 0.2) return 0;
  return 1 + (((2 * bx + 3 * by) % 7) + 7) % 7;
}
const THEME_R = [0, 85, 100, 105, 108, 100, 93, 105];                            // how far each theme's scenery reaches from the block centre (towers inside it force a plain lawn)
// Park layout, in metres from the block centre: an oval loop path with two angled entrances from the sidewalk, a gazebo plaza on the north side of the loop and a natural pond
// (three overlapping ellipses [x, y, rx, ry]) inside it. The drawing and the keep-out rules (inFootprint) both use these, so trees never stand on a path or in the water.
const PARK_PATH = Array.from({ length: 49 }, (_, i) => { const t = i / 48 * Math.PI * 2; return [72 * Math.cos(t), 56 * Math.sin(t)]; });         // closed: the last point is the first
const PARK_ENTRIES = [[[-30.4, -50.7], [-40, -78], [-50, -104]], [[69.5, 14.5], [88, 21], [106, 27]]];
const PARK_PONDS = [[-14, 4, 26, 16], [6, -6, 15, 10], [-34, -2, 12, 9]];
const PARK_PLAZA = [0, 56], PARK_PLAZA_R = 11;
const distToPolyline = (pts, x, y) => {
  let m = 1e9;
  for (let i = 0; i < pts.length - 1; i++) {
    const [ax, ay] = pts[i], [bx, by] = pts[i + 1], vx = bx - ax, vy = by - ay;
    const t = Math.max(0, Math.min(1, ((x - ax) * vx + (y - ay) * vy) / (vx * vx + vy * vy)));
    m = Math.min(m, Math.hypot(x - ax - t * vx, y - ay - t * vy));
  }
  return m;
};
const towerRects = [];                                                           // world-space footprints [x0, y0, x1, y1] of the towers, filled by setObstacles()

const city = (() => {
  const UP = new THREE.Vector3(0, 1, 0);
  const _M = new THREE.Matrix4(), _Q = new THREE.Quaternion(), _P = new THREE.Vector3(), _S = new THREE.Vector3(), _C = new THREE.Color();
  const mulberry = (a) => () => { a |= 0; a = a + 0x6D2B79F5 | 0; let t = Math.imul(a ^ a >>> 15, 1 | a); t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t; return ((t ^ t >>> 14) >>> 0) / 4294967296; };
  const inst = (geo, mat, max) => { const m = new THREE.InstancedMesh(geo, mat, max); m.count = 0; m.frustumCulled = false; m.userData.max = max; scene.add(m); return m; };
  const flatMat = (off) => new THREE.MeshLambertMaterial({ color: 0xffffff, polygonOffset: true, polygonOffsetFactor: -off, polygonOffsetUnits: -off });
  const flat = (g) => g.rotateX(-Math.PI / 2);
  const solid = (g) => g.translate(0, 0.5, 0);                                    // base at y = 0, unit height

  const circ = flat(new THREE.CircleGeometry(1, 44)), quad = flat(new THREE.PlaneGeometry(1, 1)), ring = flat(new THREE.RingGeometry(0.9, 1, 56));
  const L1c = inst(circ, flatMat(2), 800), L2c = inst(circ, flatMat(4), 3000), L3c = inst(circ, flatMat(6), 800);
  const L1q = inst(quad, flatMat(2), 2500), L2q = inst(quad, flatMat(4), 4000), L3q = inst(quad, flatMat(6), 5000);
  const L1r = inst(ring, flatMat(2), 400), L2r = inst(ring, flatMat(4), 600);
  const waterMat = new THREE.MeshLambertMaterial({ color: 0xffffff, polygonOffset: true, polygonOffsetFactor: -5, polygonOffsetUnits: -5 });
  const waterFlat = inst(circ, waterMat, 400);                                    // pond and fountain water surfaces: shimmer is animated per frame
  const cyl = inst(solid(new THREE.CylinderGeometry(1, 1, 1, 18)), new THREE.MeshLambertMaterial({ color: 0xffffff }), 4000);
  const box = inst(solid(new THREE.BoxGeometry(1, 1, 1)), new THREE.MeshLambertMaterial({ color: 0xffffff }), 7000);
  const ball = inst(new THREE.SphereGeometry(1, 8, 6), new THREE.MeshLambertMaterial({ color: 0xffffff }), 60000);
  const bulb = inst(new THREE.SphereGeometry(1, 8, 6), new THREE.MeshBasicMaterial({ color: 0xffffff }), 2500);
  const spray = inst(new THREE.SphereGeometry(1, 5, 4), new THREE.MeshBasicMaterial({ color: 0xdff4ff, transparent: true, opacity: 0.85 }), 4000);
  const meshes = [L1c, L2c, L3c, L1q, L2q, L3q, L1r, L2r, waterFlat, cyl, box, ball, bulb];

  const _E = new THREE.Euler();
  function put(mesh, wx, y, wy, sx, sy, sz, rot, color, tilt) {                   // world x/y in, three coords out; tilt (radians, about the local z axis) is optional
    const i = mesh.count; if (i >= mesh.userData.max) return;
    if (tilt) _Q.setFromEuler(_E.set(0, rot || 0, tilt, 'YXZ')); else _Q.setFromAxisAngle(UP, rot || 0);
    _P.set(wx, y, -wy); _S.set(sx, sy, sz);
    _M.compose(_P, _Q, _S); mesh.setMatrixAt(i, _M); mesh.setColorAt(i, _C.set(color)); mesh.count++;
  }
  const waters = [], fountains = [];                                              // shimmering surfaces (index into waterFlat) and fountain spots (for the spray)

  function build(bx, by) {
    for (const m of meshes) m.count = 0;
    waters.length = 0; fountains.length = 0;
    const N = 3;
    for (let i = bx - N; i <= bx + N; i++) for (let j = by - N; j <= by + N; j++) genBlock(i, j);
    for (const m of meshes) { m.instanceMatrix.needsUpdate = true; if (m.instanceColor) m.instanceColor.needsUpdate = true; }
  }

  function genBlock(bx, by) {
    const rnd = mulberry((bx * 73856093) ^ (by * 19349663) ^ 0x9e3779b9);
    const x0 = bx * BLOCK, y0 = by * BLOCK, cx = x0 + BLOCK / 2, cy = y0 + BLOCK / 2;
    const rects = towerRects.filter(r => r[0] < x0 + BLOCK && r[2] > x0 && r[1] < y0 + BLOCK && r[3] > y0);
    const free = (x, y, pad) => !rects.some(r => x > r[0] - pad && x < r[2] + pad && y > r[1] - pad && y < r[3] + pad);
    const hitsDisc = (x, y, rad) => rects.some(r => Math.max(r[0] - x, 0, x - r[2]) ** 2 + Math.max(r[1] - y, 0, y - r[3]) ** 2 < rad * rad);
    const pick = (a) => a[Math.floor(rnd() * a.length) % a.length];

    // ---- street furniture: lamp posts along every sidewalk
    const off = ROAD_HALF + WALK / 2;
    for (let s = 25; s < BLOCK; s += 50) for (const [lx, ly] of [[x0 + s, y0 + off], [x0 + s, y0 + BLOCK - off], [x0 + off, y0 + s], [x0 + BLOCK - off, y0 + s]]) {
      if (!free(lx, ly, 3)) continue;
      put(cyl, lx, 0, ly, 0.35, 9, 0.35, 0, 0x3a3f47); put(bulb, lx, 9.6, ly, 1.1, 1.1, 1.1, 0, 0xffe9a8);
    }
    const bench = (x, y, rot) => { put(box, x, 0.9, y, 4.2, 0.35, 1.5, rot, 0x8a5a32); put(box, x - Math.sin(rot) * 0.7, 1.5, y - Math.cos(rot) * 0.7, 4.2, 1.1, 0.3, rot, 0x7a4c2a); put(box, x, 0, y, 3.8, 0.9, 1.1, rot, 0x4a4f57); };
    const bed = (x, y, w, d) => {                                                // hedge-bordered flower bed: green lawn base, big dense flowers in soft colour clusters (no stripes, no flag colours, nothing in the middle)
      const pal = pick([[0xff6fb5, 0xd946a8, 0xffc2e0], [0xa66bff, 0xd8b4fe, 0xffb3d9], [0xff9f1c, 0xffd23f, 0xff7a59], [0xffe066, 0xfff3b0, 0xffc08a], [0xe63946, 0xff8fa3, 0xff7a59]]);     // one colour family per bed
      put(L2q, x, 0.5, y, w, 0.001, d, 0, 0x3d8c3f);
      for (const [hx, hy, hw, hd] of [[x, y - d / 2, w + 1.4, 1.4], [x, y + d / 2, w + 1.4, 1.4], [x - w / 2, y, 1.4, d], [x + w / 2, y, 1.4, d]]) put(box, hx, 0, hy, hw, 1.7, hd, 0, 0x2f6a35);
      const hash = (u, v) => { const t = Math.sin(u * 12.9898 + v * 78.233) * 43758.5453; return t - Math.floor(t); };     // a stable pseudo-random number from a position (does not touch the block's random stream)
      for (let yy = -d / 2 + 2.6; yy < d / 2 - 1.6; yy += 2.9) for (let xx = -w / 2 + 2.6; xx < w / 2 - 1.6; xx += 2.9) {
        const h = hash(x + xx, y + yy);
        const cluster = Math.floor(hash(x + Math.floor((xx + 40) / 7), y + Math.floor((yy + 40) / 7)) * 3) % 3;   // one colour per 7 x 7 cluster, so the flowers form soft patches, not stripes
        const col = h < 0.2 ? pal[Math.floor(h * 15) % 3] : pal[cluster];
        put(ball, x + xx + (h - 0.5) * 0.8, 1.15 + h * 0.5, y + yy + (hash(yy, xx) - 0.5) * 0.8, 1.45, 1.1, 1.45, 0, col);
      }
    };
    const water = (x, y, rx, ry, color, phase) => { waters.push([waterFlat.count, color, phase]); put(waterFlat, x, 0.3, y, rx, 1, ry, 0, color); };    // the same phase = the same shimmer, so overlapping ellipses of one pond do not flicker
    const fountain = (x, y, s) => {                                              // stone basin, water, column, upper bowl; spray is animated in animate()
      put(cyl, x, 0, y, 15 * s, 1.5 * s, 15 * s, 0, 0xbdb8ac);
      waters.push([waterFlat.count, 0x4fb4e8]); put(waterFlat, x, 1.55 * s, y, 13.6 * s, 1, 13.6 * s, 0, 0x4fb4e8);
      put(cyl, x, 0, y, 1.7 * s, 7 * s, 1.7 * s, 0, 0xd6d1c4); put(cyl, x, 6 * s, y, 6.2 * s, 0.9 * s, 6.2 * s, 0, 0xbdb8ac);
      waters.push([waterFlat.count, 0x62c4f0]); put(waterFlat, x, 6.95 * s, y, 5.3 * s, 1, 5.3 * s, 0, 0x62c4f0);
      put(ball, x, 7.8 * s, y, 1.3 * s, 1.3 * s, 1.3 * s, 0, 0xe9e4d6);
      fountains.push([x, y, s]);
    };
    const bushes = (n, rad, r0 = 40, th = 0) => { for (let i = 0; i < n; i++) { const a = rnd() * 6.283, r = r0 + rnd() * rad, x = cx + Math.cos(a) * r, y = cy + Math.sin(a) * r; if (Math.abs(x - cx) < 106 && Math.abs(y - cy) < 106 && !inFootprint(th, x - cx, y - cy) && free(x, y, 6)) { const sz = 2 + rnd() * 2.2; put(ball, x, sz * 0.8, y, sz * 1.2, sz, sz * 1.2, 0, pick([0x2f7a3a, 0x3a8a40, 0x4a9a3a, 0x7a9a3a])); } } };

    const theme = themeFor(bx, by);
    const R = THEME_R[theme];
    if (theme === 0 || hitsDisc(cx, cy, R)) { bushes(14, 80); if (free(cx, cy, 6)) bench(cx, cy, 0); return; }   // plain lawn (also the fallback when a tower is in the way)

    if (theme === 1) {                                                           // fountain plaza
      put(L1c, cx, 0.3, cy, 82, 1, 82, 0, 0xcfc9bb); put(L2r, cx, 0.3, cy, 82, 1, 82, 0, 0x8e887c); put(L2r, cx, 0.3, cy, 62, 1, 62, 0, 0xa9a398);
      put(L1q, cx, 0.3, cy, 220, 1, 9, 0, 0xcfc9bb); put(L1q, cx, 0.3, cy, 9, 1, 220, 0, 0xcfc9bb);
      fountain(cx, cy, 1.9);
      for (const [sx, sy] of [[-1, -1], [1, -1], [-1, 1], [1, 1]]) bed(cx + sx * 74, cy + sy * 74, 40, 40);
      for (let a = 0; a < 4; a++) { const an = a * 1.5708 + 0.785; bench(cx + Math.cos(an) * 68, cy + Math.sin(an) * 68, -an + 1.5708); }
    } else if (theme === 2) {                                                    // formal garden: 3 x 3 hedge beds, small fountain in the middle, gravel paths
      put(L1q, cx, 0.3, cy, 200, 1, 200, 0, 0xd8cfae);
      for (let i = -1; i <= 1; i++) for (let j = -1; j <= 1; j++) { if (i === 0 && j === 0) fountain(cx, cy, 0.95); else bed(cx + i * 62, cy + j * 62, 48, 48); }
    } else if (theme === 3) {                                                    // park: oval loop path, gazebo plaza, natural pond with a pier and ducks, shade trees, benches, lamps
      put(L1q, cx, 0.3, cy, 205, 1, 205, 0, 0x58a04b);                           // lawn
      const ribbon = (pts, w) => {                                               // a path is a ribbon of overlapping discs along a polyline
        for (let i = 0; i < pts.length - 1; i++) {
          const [ax, ay] = pts[i], [bx, by] = pts[i + 1], n = Math.max(1, Math.ceil(Math.hypot(bx - ax, by - ay) / 1.8));
          for (let k = 0; k < n; k++) put(L2c, cx + ax + (bx - ax) * k / n, 0.33, cy + ay + (by - ay) * k / n, w, 1, w, 0, 0xcdbf99);
        }
      };
      ribbon(PARK_PATH, 2.9); for (const e of PARK_ENTRIES) ribbon(e, 2.6);
      const [gx, gy] = PARK_PLAZA;                                               // paved plaza on the loop with the gazebo on it
      put(L3c, cx + gx, 0.35, cy + gy, PARK_PLAZA_R, 1, PARK_PLAZA_R, 0, 0xd6d0c2);
      put(cyl, cx + gx, 0, cy + gy, 6.4, 0.5, 6.4, 0, 0xe9e4d6);
      for (let k = 0; k < 6; k++) { const a = k * 1.0472; put(cyl, cx + gx + Math.cos(a) * 5.4, 0.5, cy + gy + Math.sin(a) * 5.4, 0.28, 4.6, 0.28, 0, 0xf4f4f4); }
      put(cyl, cx + gx, 5.1, cy + gy, 7.4, 0.55, 7.4, 0, 0x5f6f82); put(cyl, cx + gx, 5.65, cy + gy, 4.2, 0.5, 4.2, 0, 0x6f8094); put(ball, cx + gx, 6.5, cy + gy, 0.9, 0.9, 0.9, 0, 0xe9c43c);
      for (let k = 0; k < 11; k++) { const a = 0.25 + k * 0.264; put(ball, cx + gx + Math.cos(a) * 15.5, 0.7, cy + gy + Math.sin(a) * 15.5, 1, 0.9, 1, 0, [0xff6fb5, 0xffe066, 0xc77dff, 0xff9f1c][k % 4]); }   // flowers on the outer side of the plaza
      for (const [px, py, rx, ry] of PARK_PONDS) put(L2c, cx + px, 0.33, cy + py, rx + 3.4, 1, ry + 3.4, 0, 0xcdbb8a);     // sandy shore
      for (const [px, py, rx, ry] of PARK_PONDS) water(cx + px, cy + py, rx, ry, 0x3f8fc0, 1);
      for (const [dx, dy] of [[-22, 7], [-8, 1], [-30, 1], [4, -6]]) { put(ball, cx + dx, 0.95, cy + dy, 0.95, 0.7, 0.95, 0, 0xf6f6f2); put(ball, cx + dx + 0.8, 1.45, cy + dy, 0.5, 0.5, 0.5, 0, 0xe8b02a); }   // ducks
      put(box, cx - 14, 0.5, cy - 9, 2.6, 0.4, 12, 0, 0x8a6a43);                  // pier from the south shore into the pond
      for (const [ox, oy] of [[-15.2, -14.6], [-12.8, -14.6], [-15.2, -3.4], [-12.8, -3.4]]) put(cyl, cx + ox, 0, cy + oy, 0.2, 0.9, 0.2, 0, 0x6a4a28);
      const along = (i, off, fn) => {                                             // a point beside the loop: i = which path point, off = metres to the side (positive = towards the inside)
        const [ax, ay] = PARK_PATH[i], [bx, by] = PARK_PATH[i + 1], len = Math.hypot(bx - ax, by - ay), tx = (bx - ax) / len, ty = (by - ay) / len;
        fn(cx + ax - ty * off, cy + ay + tx * off, Math.atan2(ty, tx));
      };
      for (const i of [3, 10, 27, 36, 44]) along(i, 4.8, (x, y, r) => bench(x, y, r));
      for (const i of [0, 7, 15, 22, 31, 40]) along(i, -4.6, (x, y) => { put(cyl, x, 0, y, 0.3, 5.6, 0.3, 0, 0x3a3f47); put(bulb, x, 6, y, 1, 1, 1, 0, 0xffe9a8); });
      const tree = (x, y, sc, g1, g2) => {                                        // a broadleaf shade tree
        put(cyl, x, 0, y, 0.5 * sc, 4.4 * sc, 0.5 * sc, 0, 0x6a4a2a);
        put(ball, x, 6.2 * sc, y, 4.2 * sc, 3.5 * sc, 4.2 * sc, 0, g1); put(ball, x + 1.6 * sc, 5.0 * sc, y + 0.8 * sc, 2.9 * sc, 2.5 * sc, 2.9 * sc, 0, g2);
      };
      let placed = 0, tries = 0;
      while (placed < 18 && tries++ < 250) {
        const x = (rnd() * 2 - 1) * 98, y = (rnd() * 2 - 1) * 98;
        if (inFootprint(3, x, y) || !free(cx + x, cy + y, 6)) continue;
        const fall = rnd() < 0.18;                                                // a few trees in autumn colours
        tree(cx + x, cy + y, 0.85 + rnd() * 0.7, fall ? 0xd9822b : pick([0x3a8a40, 0x2f7a3a, 0x4a9a3a]), fall ? 0xc94c3a : pick([0x4aa04a, 0x5aa83f]));
        placed++;
      }
    } else if (theme === 4) {                                                    // tennis club: 3 x 4 hard courts with run-off, lines, nets and floodlights
      put(L1q, cx, 0.3, cy, 176, 1, 148, 0, 0x8f9f94);
      for (let i = -1; i <= 1; i++) for (let j = -1.5; j <= 1.5; j += 1) {
        const px = cx + i * 56, py = cy + j * 34, wl = 0xf4f4f4;
        put(L2q, px, 0.32, py, 54, 1, 32, 0, 0x3f8f5a);                          // green run-off area
        put(L3q, px, 0.34, py, 38, 1, 17.6, 0, 0x2c6eb5);                        // blue playing surface
        put(L3q, px - 19, 0.36, py, 0.4, 1, 17.6, 0, wl); put(L3q, px + 19, 0.36, py, 0.4, 1, 17.6, 0, wl);               // baselines
        put(L3q, px, 0.36, py - 8.8, 38, 1, 0.4, 0, wl); put(L3q, px, 0.36, py + 8.8, 38, 1, 0.4, 0, wl);                 // doubles sidelines
        put(L3q, px, 0.36, py - 6.6, 38, 1, 0.4, 0, wl); put(L3q, px, 0.36, py + 6.6, 38, 1, 0.4, 0, wl);                 // singles sidelines
        put(L3q, px - 10.2, 0.36, py, 0.4, 1, 13.2, 0, wl); put(L3q, px + 10.2, 0.36, py, 0.4, 1, 13.2, 0, wl);           // service lines
        put(L3q, px, 0.36, py, 20.4, 1, 0.4, 0, wl);                                                                      // centre service line
        put(box, px, 0, py, 0.25, 1.05, 17.6, 0, 0x2a2f36); put(box, px, 1.0, py, 0.3, 0.12, 17.6, 0, 0xf4f4f4);          // net with white tape
        put(cyl, px, 0, py - 9.3, 0.18, 1.2, 0.18, 0, 0x2a2f36); put(cyl, px, 0, py + 9.3, 0.18, 1.2, 0.18, 0, 0x2a2f36); // net posts
      }
      for (const [sx, sy] of [[-1, -1], [1, -1], [-1, 1], [1, 1]]) { const px = cx + sx * 92, py = cy + sy * 78; put(cyl, px, 0, py, 0.5, 24, 0.5, 0, 0x4a4f57); put(bulb, px, 24.6, py, 2.4, 2.4, 2.4, 0, 0xfff6c8); }
    } else if (theme === 5) {                                                    // playground + basketball courts: swings, slide, roundabout, seesaw, picnic tables
      put(L1q, cx, 0.3, cy, 196, 1, 186, 0, 0x62a954);                           // lawn
      put(L2q, cx, 0.32, cy - 62, 92, 1, 28, 0, 0x8f9f94);                       // paved area of the two courts
      for (const sx of [-1, 1]) {
        const x = cx + sx * 22, y = cy - 62, wl = 0xf4f4f4;
        put(L3q, x, 0.34, y, 36, 1, 19, 0, 0xd98b3a);                            // court
        for (const kx of [-1, 1]) put(L3q, x + kx * 14, 0.35, y, 8, 1, 5.5, 0, 0xb5532f);   // the painted keys
        put(L3q, x, 0.36, y - 9.3, 36, 1, 0.4, 0, wl); put(L3q, x, 0.36, y + 9.3, 36, 1, 0.4, 0, wl);
        put(L3q, x - 18, 0.36, y, 0.4, 1, 19, 0, wl); put(L3q, x + 18, 0.36, y, 0.4, 1, 19, 0, wl); put(L3q, x, 0.36, y, 0.4, 1, 19, 0, wl);
        for (const kx of [-1, 1]) { put(L3q, x + kx * 10, 0.36, y, 0.4, 1, 5.5, 0, wl);
          put(cyl, x + kx * 19.6, 0, y, 0.25, 4.6, 0.25, 0, 0x4a4f57); put(box, x + kx * 18.8, 3.2, y, 0.25, 2, 3, 0, 0xf4f4f4); put(cyl, x + kx * 17.7, 3.3, y, 0.9, 0.1, 0.9, 0, 0xe8631a); }   // hoops
      }
      for (const [sx, sy] of [[-1, -1], [1, -1], [-1, 1], [1, 1]]) { const px = cx + sx * 47, py = cy - 62 + sy * 14.5; put(cyl, px, 0, py, 0.4, 14, 0.4, 0, 0x4a4f57); put(bulb, px, 14.6, py, 1.8, 1.8, 1.8, 0, 0xfff6c8); }
      put(L2q, cx, 0.32, cy - 27, 7, 1, 44, 0, 0xcdbf99);                        // gravel path from the courts to the playground
      bench(cx - 30, cy - 44, 0); bench(cx + 30, cy - 44, 0);

      put(L2q, cx - 38, 0.32, cy + 28, 96, 1, 66, 0, 0xe8d6a6);                  // sand
      for (const [bx, by, bw, bd] of [[cx - 38, cy - 5, 98, 1.4], [cx - 38, cy + 61, 98, 1.4], [cx - 86, cy + 28, 1.4, 66], [cx + 10, cy + 28, 1.4, 66]]) put(box, bx, 0, by, bw, 0.9, bd, 0, 0x9a7448);   // timber edge
      const sw = cx - 68, sy0 = cy + 30;                                          // swing set: frame, bar, 2 ropes + seat each
      for (const px of [-7, 7]) for (const py of [-3, 3]) put(cyl, sw + px, 0, sy0 + py, 0.25, 5.2, 0.25, 0, 0xd94b4b);
      put(box, sw, 5, sy0 - 3, 15, 0.35, 0.35, 0, 0xd94b4b); put(box, sw, 5, sy0 + 3, 15, 0.35, 0.35, 0, 0xd94b4b); put(box, sw, 5, sy0, 0.35, 0.35, 6, 0, 0xd94b4b);
      for (const px of [-3.2, 3.2]) { put(cyl, px + sw, 1.4, sy0, 0.06, 3.6, 0.06, 0, 0x555a62); put(box, px + sw, 1.2, sy0, 2.2, 0.22, 1, 0, px < 0 ? 0xf2c230 : 0x3f7dd6); }
      const sl = cx - 38, sly = cy + 42;                                          // slide: ladder, platform, tilted ramp
      for (const px of [-1.4, 1.4]) for (const py of [-1.4, 1.4]) put(cyl, sl + px, 0, sly + py, 0.2, 4, 0.2, 0, 0x4a4f57);
      put(box, sl, 3.9, sly, 3.4, 0.3, 3.4, 0, 0x3f7dd6); put(box, sl - 1.9, 0, sly, 0.3, 3.9, 1.6, 0, 0x4a4f57);
      put(box, sl + 5.0, 2.0, sly, 7.4, 0.25, 2.2, 0, 0xd94b4b, -0.56);
      const mr = cx - 8, mry = cy + 14;                                           // roundabout
      put(cyl, mr, 0.5, mry, 4.2, 0.35, 4.2, 0, 0x2fa8b8); put(cyl, mr, 0, mry, 0.3, 3.2, 0.3, 0, 0x4a4f57);
      for (const a of [0, 1.5708, 3.1416, 4.7124]) put(cyl, mr + Math.cos(a) * 3.2, 0.8, mry + Math.sin(a) * 3.2, 0.12, 1.6, 0.12, 0, 0xf2c230);
      const ss = cx - 8, ssy = cy + 48;                                           // seesaw
      put(box, ss, 0, ssy, 1, 1.3, 1.2, 0, 0x4a4f57); put(box, ss, 1.3, ssy, 9, 0.3, 1, 0, 0xf2c230, 0.2);
      for (const px of [-62, -48, -34]) { put(cyl, cx + px, 0, cy + 6, 0.25, 1.1, 0.25, 0, 0x7a7f87); put(ball, cx + px, 1.9, cy + 6, 1.1, 0.9, 1.5, 0, px === -48 ? 0xe8631a : 0x3fae5a); }   // spring riders

      put(L2q, cx + 66, 0.32, cy + 35, 24, 1, 80, 0, 0xcdbf99);                  // picnic area
      for (const py of [8, 35, 62]) {
        put(box, cx + 66, 2.3, cy + py, 5, 0.3, 2.2, 0, 0x9a7448);
        for (const lx of [-1.8, 1.8]) put(box, cx + 66 + lx, 0, cy + py, 0.4, 2.3, 2, 0, 0x7a5a32);
        for (const bz of [-1.7, 1.7]) put(box, cx + 66, 1.2, cy + py + bz, 5, 0.25, 0.8, 0, 0x9a7448);
      }
      put(box, cx - 30, 0, cy + 82, 66, 1.7, 1.4, 0, 0x2f6a35); put(box, cx + 42, 0, cy + 82, 48, 1.7, 1.4, 0, 0x2f6a35);   // hedge along the south side
    } else if (theme === 6) {                                                    // car park: two rows of painted stalls on either side of an aisle, a booth, lamps (the parked cars come from the server: city_world.py)
      put(L1q, cx, 0.3, cy, 176, 1, 56, 0, 0x3a3d42);
      for (const row of [-1, 1]) {
        for (let k = 0; k <= 26; k++) put(L2q, cx - 84.5 + k * 6.5, 0.34, cy + row * 17, 0.35, 1, 12, 0, 0xe8e8e8);        // stall lines (6.5 wide, 12 deep)
        put(L2q, cx, 0.34, cy + row * 23.2, 169, 1, 0.35, 0, 0xe8e8e8);                                                  // back line
      }
      for (let k = -10; k <= 10; k++) put(L2q, cx + k * 8, 0.34, cy, 4, 1, 0.4, 0, 0xe9c43c);                                // dashed centre line of the aisle
      put(box, cx + 80, 0, cy + 31, 5, 3.4, 4, 0, 0xcfd4da); put(box, cx + 80, 3.4, cy + 31, 6, 0.4, 5, 0, 0x5f6f82);       // ticket booth
      for (const lx of [-70, -20, 30]) { put(cyl, cx + lx, 0, cy, 0.35, 8, 0.35, 0, 0x3a3f47); put(bulb, cx + lx, 8.6, cy, 1.1, 1.1, 1.1, 0, 0xffe9a8); }
    } else {                                                                     // community garden: fenced lawn, raised beds with real crop rows, greenhouse, tank, scarecrow, sunflower border
      put(L1q, cx, 0.3, cy, 198, 1, 160, 0, 0x62a954);                           // lawn inside the fence
      put(L2q, cx, 0.32, cy, 194, 1, 11, 0, 0xcdbf99);                           // gravel paths: one across, one between each pair of bed columns, and the entrance
      for (const px of [-29, 29]) put(L2q, cx + px, 0.32, cy - 8, 9, 1, 100, 0, 0xcdbf99);
      put(L2q, cx, 0.32, cy + 66, 11, 1, 28, 0, 0xcdbf99);
      const kinds = [0, 1, 2, 3, 4, 5]; for (let i = kinds.length - 1; i > 0; i--) { const j = Math.floor(rnd() * (i + 1)); [kinds[i], kinds[j]] = [kinds[j], kinds[i]]; }
      const greens = [0x7fcf6a, 0x5fb85a, 0x9bd86a];
      let bi = 0;
      for (const by of [-27, 27]) for (const bx of [-58, 0, 58]) {                // six raised beds, each with its own crop
        const x = cx + bx, y = cy + by, w = 42, d = 30, kind = kinds[bi++];
        put(L2q, x, 0.32, y, w, 1, d, 0, 0x4a3220);                               // soil
        for (const [hx, hy, hw, hd] of [[x, y - d / 2, w + 1.8, 1.8], [x, y + d / 2, w + 1.8, 1.8], [x - w / 2, y, 1.8, d], [x + w / 2, y, 1.8, d]]) put(box, hx, 0, hy, hw, 1.6, hd, 0, 0x9a7448);   // timber frame
        for (const ry of [-9, 0, 9]) for (let rx = -w / 2 + 5; rx <= w / 2 - 4; rx += kind === 2 ? 9 : kind === 1 ? 6.5 : 5) {
          const px = x + rx, py = y + ry;
          if (kind === 0) put(ball, px, 0.9, py, 1.7, 0.9, 1.7, 0, pick(greens));                                          // lettuce
          else if (kind === 1) { put(ball, px, 1.7, py, 2.3, 1.9, 2.3, 0, 0x6fae78); put(ball, px, 2.3, py, 1.3, 1.1, 1.3, 0, 0x93cf92); }   // cabbage
          else if (kind === 2) { put(ball, px, 1.5, py, 2.6, 1.9, 2.6, 0, 0xe8863a); put(ball, px + 1.8, 0.8, py + 1.5, 1, 0.5, 1, 0, 0x3a8a35); }   // pumpkin
          else if (kind === 3) { put(cyl, px, 0, py, 0.14, 3.6, 0.14, 0, 0x7a5a32); put(ball, px, 2.3, py, 1.4, 1.7, 1.4, 0, 0x3f9a3a); put(ball, px + 0.9, 2.6, py, 0.5, 0.5, 0.5, 0, 0xe03a2a); put(ball, px - 0.8, 1.7, py + 0.6, 0.5, 0.5, 0.5, 0, 0xe03a2a); }   // tomato on a stake
          else if (kind === 4) { put(ball, px, 0.9, py, 1.2, 1.1, 1.2, 0, 0x4fb04a); put(ball, px, 0.35, py, 0.5, 0.5, 0.5, 0, 0xf08a24); }   // carrot tops
          else { put(cyl, px, 0, py, 0.12, 2.2, 0.12, 0, 0x3a8a35); put(ball, px, 2.5, py, 1.2, 1.2, 1.2, 0, pick([0xff6fb5, 0xff9f1c, 0xc77dff, 0xffe066, 0xff5a5a])); }   // cut flowers
        }
      }
      for (let sx = -86; sx <= 86; sx += 12) { put(cyl, cx + sx, 0, cy - 72, 0.2, 7, 0.2, 0, 0x3a7a35); put(ball, cx + sx, 7.2, cy - 72, 2, 2, 2, 0, 0xf2c230); put(ball, cx + sx, 7.9, cy - 72, 1.1, 1.1, 1.1, 0, 0x5a3a1a); }   // sunflower border
      put(box, cx - 62, 0, cy + 60, 26, 5, 14, 0, 0xcdeff2); put(box, cx - 62, 5, cy + 60, 26, 2, 6, 0, 0xe3f9fb); put(box, cx - 62, 0, cy + 67.2, 4, 4, 0.3, 0, 0x5a7a80);   // greenhouse
      put(cyl, cx + 26, 0, cy + 60, 3.2, 5, 3.2, 0, 0x6b8fa8); put(cyl, cx + 26, 5, cy + 60, 2.6, 0.5, 2.6, 0, 0x4f6f86);                                                       // water tank
      put(cyl, cx + 62, 0, cy + 60, 0.25, 7, 0.25, 0, 0x7a5a32); put(box, cx + 62, 5, cy + 60, 6, 0.4, 0.4, 0, 0x7a5a32); put(box, cx + 62, 2.6, cy + 60, 2.4, 3, 1.2, 0, 0xc0392b);
      put(ball, cx + 62, 7.5, cy + 60, 1.2, 1.2, 1.2, 0, 0xe8c87a); put(box, cx + 62, 8.2, cy + 60, 3.2, 0.4, 3.2, 0, 0x6a4a28); put(box, cx + 62, 8.5, cy + 60, 1.8, 1, 1.8, 0, 0x6a4a28);   // scarecrow
      for (let fx = -96; fx <= 96; fx += 8) for (const fy of [-79, 79]) if (!(fy > 0 && Math.abs(fx) < 8)) put(box, cx + fx, 0, cy + fy, 0.6, 3, 0.6, 0, 0x8a6a43);   // fence posts (gap = gate)
      for (let fy = -79; fy <= 79; fy += 8) for (const fx of [-96, 96]) put(box, cx + fx, 0, cy + fy, 0.6, 3, 0.6, 0, 0x8a6a43);
      for (const h of [1.0, 2.2]) { put(box, cx, h, cy - 79, 192, 0.3, 0.3, 0, 0x9a7a53); put(box, cx - 52, h, cy + 79, 88, 0.3, 0.3, 0, 0x9a7a53); put(box, cx + 52, h, cy + 79, 88, 0.3, 0.3, 0, 0x9a7a53); put(box, cx - 96, h, cy, 0.3, 0.3, 158, 0, 0x9a7a53); put(box, cx + 96, h, cy, 0.3, 0.3, 158, 0, 0x9a7a53); }
    }
  }

  // ---- ground rules for the sim's own objects (trees, rocks, people, cars): where is the ground NOT free?
  function blockTheme(bx, by) {                                                  // same choice genBlock makes, including the tower fallback
    const th = themeFor(bx, by);
    if (th === 0) return 0;
    const x0 = bx * BLOCK, y0 = by * BLOCK, cx = x0 + BLOCK / 2, cy = y0 + BLOCK / 2, R = THEME_R[th];
    for (const r of towerRects) {
      if (!(r[0] < x0 + BLOCK && r[2] > x0 && r[1] < y0 + BLOCK && r[3] > y0)) continue;
      if (Math.max(r[0] - cx, 0, cx - r[2]) ** 2 + Math.max(r[1] - cy, 0, cy - r[3]) ** 2 < R * R) return 0;
    }
    return th;
  }
  function inFootprint(th, dx, dy) {                                             // is local point (dx, dy) from the block centre covered by theme th's scenery (water, courts, plazas, beds, lots, paths)?
    switch (th) {
      case 1: return Math.abs(dx) < 96 && Math.abs(dy) < 96;                                     // plaza + corner beds
      case 2: return Math.abs(dx) < 102 && Math.abs(dy) < 102;                                   // formal garden
      case 3: return PARK_PONDS.some(([px, py, rx, ry]) => ((dx - px) / (rx + 12)) ** 2 + ((dy - py) / (ry + 12)) ** 2 < 1) || Math.hypot(dx - PARK_PLAZA[0], dy - PARK_PLAZA[1]) < PARK_PLAZA_R + 8 ||   // park: pond, gazebo plaza
                     distToPolyline(PARK_PATH, dx, dy) < 5 || PARK_ENTRIES.some((e) => distToPolyline(e, dx, dy) < 5);                                   // and the paths
      case 4: return Math.abs(dx) < 90 && Math.abs(dy) < 76;                                     // tennis club
      case 5: return Math.abs(dx) < 96 && Math.abs(dy) < 92;                                     // playground + basketball courts
      case 6: return Math.abs(dx) < 90 && Math.abs(dy) < 34;                                     // car park
      case 7: return Math.abs(dx) < 98 && Math.abs(dy) < 80;                                     // community garden
      default: return false;                                                                      // plain lawn
    }
  }
  function blocked(x, y) {                                                       // true where a tree / rock must not stand
    const bx = Math.floor(x / BLOCK), by = Math.floor(y / BLOCK);
    return inFootprint(blockTheme(bx, by), x - (bx + 0.5) * BLOCK, y - (by + 0.5) * BLOCK);
  }

  function animate(tSec) {
    for (const [idx, col, ph] of waters) {                                        // slow shimmer
      const k = 0.88 + 0.12 * Math.sin(tSec * 1.6 + (ph === undefined ? idx : ph) * 1.3);
      waterFlat.setColorAt(idx, _C.set(col).multiplyScalar(k));
    }
    if (waterFlat.instanceColor) waterFlat.instanceColor.needsUpdate = true;
    let n = 0;
    for (const [fx, fy, s] of fountains) {
      for (let i = 0; i < 20; i++) {                                              // jets from the upper bowl's rim outward into the basin
        const a = (i / 20) * 6.283, t = (tSec * 0.7 + (i % 5) * 0.2) % 1, r = (5.6 + 7.2 * t) * s;
        _P.set(fx + Math.cos(a) * r, (6.9 + (1.5 - 6.9) * t + 3.2 * 4 * t * (1 - t)) * s, -(fy + Math.sin(a) * r)); _S.setScalar(0.5 * s);
        _M.compose(_P, _Q.identity(), _S); if (n < 4000) spray.setMatrixAt(n++, _M);
      }
      for (let i = 0; i < 8; i++) {                                               // central jet rising from the finial and falling back into the bowl
        const a = i * 0.785, t = (tSec * 0.9 + i * 0.125) % 1, r = 3.4 * t * s;
        _P.set(fx + Math.cos(a) * r, (8.2 + 7 * 4 * t * (1 - t) - 1.2 * t) * s, -(fy + Math.sin(a) * r)); _S.setScalar(0.55 * s);
        _M.compose(_P, _Q.identity(), _S); if (n < 4000) spray.setMatrixAt(n++, _M);
      }
    }
    spray.count = n; spray.instanceMatrix.needsUpdate = true;
  }
  return { bx: NaN, by: NaN, rebuild(bx, by) { this.bx = bx; this.by = by; build(bx, by); }, animate, blocked };
})();

export { BLOCK, ROAD_HALF, WALK, TILE, PLANE, ground, towerRects, city };
