// Buildings: the city's towers in three real-world facade styles, each with its own paint colours:
//   masonry  (brick, concrete, sandstone, cream, grey)  small punched windows with frames and sills
//   glass    (blue, aqua, steel, deep blue, bronze)       a curtain wall of reflective glass panels with dark mullions
//   ribbon   (white, light grey, pale sand, pale blue)    continuous horizontal window bands between pale spandrels
// Taller towers are more often glass, like downtown skyscrapers. The windows keep one real size (a window cell is CELL units wide and tall) whatever the size of the
// building, and light up warm at dusk and night. Instanced meshes: one draw call per style.

import * as THREE from 'three';
import { scene } from './core.js';

const CELL = 4;                                                                  // size of one window cell in world units (4 columns x 8 rows per texture tile)
const MAX_PER_STYLE = 800;
const PALETTES = [
  [0xcfc6b4, 0xb9b4aa, 0xa9553d, 0x8c4a38, 0xd9c9a3, 0xe6dfcf, 0x9a9690, 0xb87a5a],           // masonry: beige, concrete, brick, dark brick, sandstone, cream, grey, terracotta
  [0xa8cbe6, 0x8fd0c8, 0x9fb3c4, 0x6f93b8, 0xc9b88c, 0xb7c9c9, 0x86a8c9],                     // glass: sky blue, aqua, steel, deep blue, bronze, silver-green, blue
  [0xf2f0ea, 0xcfd2d6, 0xe8dcb8, 0xc2d0dc, 0xdedbd2],                                         // ribbon: white, light grey, pale sand, pale blue, stone
];
// Share of the windows that are lit (0..1). It is a level that moves with the time of day (see updateBuildingLight): people switch the lights on at dusk, most of
// the city is lit in the early night, and then window after window goes dark. Every window has its own random threshold and every building its own habit
// (a few offices are dark all night, some towers stay bright), so no two buildings and no two floors look alike.
const uLit = { value: 0 };

// ── facade textures: 128 x 256 px = 4 columns x 8 rows of 32 px cells; the emissive twin has only the lit windows
function facade(draw) {
  const mk = () => { const c = document.createElement('canvas'); c.width = 128; c.height = 256; return [c, c.getContext('2d')]; };
  const [cm, gm] = mk(), [ce, ge] = mk();
  ge.fillStyle = '#000'; ge.fillRect(0, 0, 128, 256);
  draw(gm, ge);
  const tex = (c, srgb) => { const t = new THREE.CanvasTexture(c); if (srgb) t.colorSpace = THREE.SRGBColorSpace; t.wrapS = t.wrapT = THREE.RepeatWrapping; t.anisotropy = 4; return t; };
  return { map: tex(cm, true), emissive: tex(ce, true) };
}
const cells = (fn) => { for (let r = 0; r < 8; r++) for (let c = 0; c < 4; c++) fn(c * 32, r * 32, r, c); };
const LIT = '#ffffff';                                                           // every window is painted in the emissive twin; the shader picks which ones are lit, and their colour

const masonry = facade((g, e) => {
  g.fillStyle = '#f3efe6'; g.fillRect(0, 0, 128, 256);
  for (let i = 0; i < 1400; i++) { g.fillStyle = `rgba(0,0,0,${Math.random() * 0.05})`; g.fillRect(Math.random() * 128, Math.random() * 256, 2, 2); }      // weathering
  cells((x, y) => {
    g.fillStyle = 'rgba(0,0,0,0.05)'; g.fillRect(x, y, 32, 2);                                                                                             // floor line
    g.fillStyle = '#ddd8ca'; g.fillRect(x + 5, y + 4, 22, 24);                                                                                              // window frame
    g.fillStyle = '#3d556b'; g.fillRect(x + 7, y + 6, 18, 20);                                                                                              // glass
    g.fillStyle = 'rgba(255,255,255,0.18)'; g.fillRect(x + 7, y + 6, 18, 6);                                                                                // sky reflection
    g.fillStyle = '#ddd8ca'; g.fillRect(x + 15, y + 6, 2, 20); g.fillRect(x + 4, y + 27, 24, 3);                                                            // mullion and sill
    e.fillStyle = LIT; e.fillRect(x + 7, y + 6, 18, 20);
  });
});
const glass = facade((g, e) => {
  cells((x, y) => {
    const grad = g.createLinearGradient(x, y, x, y + 32); grad.addColorStop(0, '#e2f0fa'); grad.addColorStop(1, Math.random() < 0.2 ? '#9bc0dc' : '#b9d6ea');
    g.fillStyle = grad; g.fillRect(x, y, 32, 32);
    g.fillStyle = 'rgba(255,255,255,0.3)'; g.fillRect(x + 3, y + 3, 10, 26);                                                                               // glare streak
    g.strokeStyle = '#3c5266'; g.lineWidth = 2; g.strokeRect(x + 1, y + 1, 30, 30);                                                                          // mullions
    g.fillStyle = '#7790a6'; g.fillRect(x, y + 25, 32, 4);                                                                                                  // spandrel panel at floor level
    e.fillStyle = LIT; e.fillRect(x + 3, y + 3, 26, 21);
  });
});
const ribbon = facade((g, e) => {
  g.fillStyle = '#efede6'; g.fillRect(0, 0, 128, 256);
  cells((x, y) => {
    g.fillStyle = '#41586d'; g.fillRect(x, y + 6, 32, 17);                                                                                                  // window band
    g.fillStyle = 'rgba(255,255,255,0.16)'; g.fillRect(x, y + 6, 32, 5);
    g.fillStyle = '#d0ccbf'; g.fillRect(x + 15, y + 6, 2, 17); g.fillRect(x, y + 24, 32, 2);                                                                // mullion and floor edge
    e.fillStyle = LIT; e.fillRect(x + 1, y + 7, 30, 15);
  });
});

function material(tex, roughness, metalness, roofUV) {
  const m = new THREE.MeshStandardMaterial({ map: tex.map, emissiveMap: tex.emissive, emissive: 0xffffff, emissiveIntensity: 1.0, roughness, metalness, color: 0xffffff });
  m.onBeforeCompile = (sh) => {                                                   // scale the texture by the real size of each building so windows keep their size
    sh.uniforms.uLit = uLit;
    sh.vertexShader = sh.vertexShader.replace('void main() {', `varying vec4 vLitInfo;
      void main() {`).replace('#include <uv_vertex>', `#include <uv_vertex>
      #ifdef USE_INSTANCING
        vec3 isc = vec3(length(instanceMatrix[0].xyz), length(instanceMatrix[1].xyz), length(instanceMatrix[2].xyz));
        float hz = abs(normal.x) > 0.5 ? isc.z : isc.x;
        bool isRoof = abs(normal.y) > 0.5;
        vec2 tuv = isRoof ? vec2(${roofUV[0]}, ${roofUV[1]}) : uv * vec2(hz / ${(CELL * 4).toFixed(1)}, isc.y / ${(CELL * 8).toFixed(1)});
        #ifdef USE_MAP
          vMapUv = tuv;
        #endif
        #ifdef USE_EMISSIVEMAP
          vEmissiveMapUv = tuv;
        #endif
        float bh = fract(sin(dot(instanceMatrix[3].xz, vec2(12.9898, 78.233))) * 43758.5453);     // a stable number per building
        vLitInfo = vec4(tuv * vec2(4.0, 8.0), bh, isRoof ? 1.0 : 0.0);
      #endif`);
    sh.fragmentShader = sh.fragmentShader.replace('void main() {', `varying vec4 vLitInfo;
      uniform float uLit;
      float h21(vec2 p) { return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453); }
      void main() {`).replace('#include <emissivemap_fragment>', `#include <emissivemap_fragment>
      {
        vec2 cell = floor(vLitInfo.xy);
        float bh = floor(vLitInfo.z * 1024.0 + 0.5) / 1024.0;                    // interpolation leaves tiny errors: snap to a stable number, or the hashes below would turn them into noise
        cell = floor(vLitInfo.xy + 0.0005);
        float hw = h21(cell + bh * 57.0);                                          // this window's own threshold
        float hf = h21(vec2(cell.y, bh * 91.0));                                   // shared by the whole floor: offices switch off floor by floor
        float th = mix(hw, hf, 0.3);
        float habit = mix(0.6, 1.3, h21(vec2(bh, 3.0)));                           // some buildings are brighter than others, but every building has lit windows
        float lvl = clamp(uLit * habit, 0.0, 1.0);
        float on = (1.0 - smoothstep(lvl - 0.05, lvl, th));
        // far away a window is smaller than a pixel and its random on/off would sparkle as the camera moves: there use the average (the share of lit windows)
        float aa = clamp((max(fwidth(vLitInfo.x), fwidth(vLitInfo.y)) - 0.3) * 3.0, 0.0, 1.0);      // x and y are in window cells: 0.3 cells per pixel = a window about 3 pixels wide
        on = (1.0 - vLitInfo.w) * mix(on, lvl, aa);
        float k = h21(cell * 1.7 + bh * 13.0);                                     // the colour of the light behind the window
        vec3 tint = k < 0.40 ? vec3(1.0, 0.82, 0.52) : k < 0.68 ? vec3(1.0, 0.93, 0.78) : k < 0.86 ? vec3(1.0, 0.68, 0.38) : k < 0.95 ? vec3(0.78, 0.88, 1.0) : vec3(0.9, 0.55, 1.0);
        tint = mix(tint * (0.45 + 0.7 * h21(cell + bh * 7.0)), vec3(0.85, 0.72, 0.5), aa);                  // far away: the average colour
        totalEmissiveRadiance *= on * tint;
      }`);
  };
  return m;
}
const materials = [material(masonry, 0.8, 0.02, [0.02, 0.02]), material(glass, 0.3, 0.12, [0.1, 0.07]), material(ribbon, 0.65, 0.06, [0.02, 0.02])];     // low metalness: with no environment map a metal would just look black
const unitBox = new THREE.BoxGeometry(1, 1, 1);
const meshes = materials.map((m) => { const im = new THREE.InstancedMesh(unitBox, m, MAX_PER_STYLE); im.count = 0; im.frustumCulled = false; scene.add(im); return im; });
const _c = new THREE.Color();

// which style a building gets: taller = more glass. hv is a stable number in [0, 1) from the building's position.
export function styleFor(height, hv) {
  const g = height > 170 ? 0.55 : height > 110 ? 0.38 : 0.2;
  return hv < g ? 1 : hv < g + 0.22 ? 2 : 0;
}
export function beginTowers() { for (const m of meshes) m.count = 0; }
export function addTower(matrix, height, hv) {                                   // matrix: position and size (w, h, d) of the building
  const style = styleFor(height, hv), m = meshes[style];
  if (m.count >= MAX_PER_STYLE) return;
  const pal = PALETTES[style];
  m.setMatrixAt(m.count, matrix); m.setColorAt(m.count, _c.set(pal[Math.floor(((hv * 977) % 1) * pal.length) % pal.length]));
  m.count++;
}
export function endTowers() { for (const m of meshes) { m.instanceMatrix.needsUpdate = true; if (m.instanceColor) m.instanceColor.needsUpdate = true; } }
// The level follows the time of day: dusk after the day switches the lights on, the early night is the brightest, then more and more windows go dark; the dusk that
// follows a night (dawn) keeps going down. dt is taken from the clock, so it works at any frame rate.
let level = 0, lastLight = 'day', cameFromNight = false, lastT = performance.now() / 1000;
const PEAK = 0.3;                                                               // share of the windows lit at the evening peak
const NIGHT_SECONDS = 28;                                                        // time for the lights of the night to go from the evening peak to the late-night low
export const setBuildingLightLevel = (v) => { level = v; };                       // for tests
export function updateBuildingLight(light) {
  const now = performance.now() / 1000, dt = Math.min(0.25, now - lastT); lastT = now;
  if (light !== lastLight) { if (light === 'night') { cameFromNight = true; if (lastLight === 'day') level = PEAK; } else if (light === 'day') cameFromNight = false; lastLight = light; }
  if (light === 'day') level += (0 - level) * Math.min(1, dt * 1.5);
  else if (light === 'dusk' && !cameFromNight) level += (PEAK - level) * Math.min(1, dt * 0.8);                     // evening: lights come on
  else level = Math.max(0.05, level - dt * (PEAK - 0.05) / NIGHT_SECONDS);                                          // night and dawn: window after window goes dark
  uLit.value = level;
}
