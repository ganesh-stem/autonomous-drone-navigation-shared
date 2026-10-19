// UI: HUD read-outs, drop-downs, replay controls, camera panel, drawers and the weather / light automation.

import { $, controls, env } from './core.js';
import { ctrl } from './net.js';
import { state } from './state.js';

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

// ── the drone's camera panels and the detector's result ──────────────────────────
function showCamera(m) {
  $('camImg').src = 'data:image/png;base64,' + m.png;
  for (let i = 0; i < 3; i++) {                                   // found / really there per class, green when they agree, amber when the detector missed or invented one
    $('fnd' + i).textContent = m.found[i]; $('tru' + i).textContent = m.truth[i];
    $('fnd' + i).className = m.found[i] === m.truth[i] ? 'ok' : 'miss';
  }
  const sensors = { fused: 'RGB + thermal + height', rgb: 'RGB only', thermal: 'Thermal only', lidar: 'Height only', 'thermal+lidar': 'Thermal + height' }[state.last ? state.last.sensorSet : 'fused'];
  $('camL').textContent = 'Altitude ' + m.alt + '  ·  Light ' + pretty(m.light) + '  ·  Detector uses ' + sensors;
}

// ── controls ─────────────────────────────────────────────────────────────────
window.snapCar = () => { Object.assign(state.pose, state.target); };      // used by record_3d.py (frame-exact capture)
$('replayPlay').onclick = () => { const v = $('replaySel').value; if (v) ctrl('replay_start', v); };
$('replaySeek').onpointerdown = () => { seeking = true; };
$('replaySeek').onpointerup = () => { seeking = false; };
$('replaySeek').oninput = () => ctrl('replay_seek', $('replaySeek').value / 1000);
$('pauseBtn').onclick = () => ctrl(state.last && state.last.paused ? 'resume' : 'pause');
$('speed').oninput = (e) => { const v = parseFloat(e.target.value); $('speedL').textContent = v.toFixed(1) + 'x'; ctrl('speed', v); };
for (const [id, mode] of [['camChase', 'chase'], ['camOrbit', 'orbit'], ['camTop', 'top']]) {
  $(id).onclick = () => { state.camMode = mode; ['camChase', 'camOrbit', 'camTop'].forEach(i => $(i).classList.toggle('active', i === id));
    controls.enabled = mode === 'orbit'; };
}

// ── automatic weather and light ─────────────────────────────────────────────────
const sendAutoWeather = () => ctrl('auto_weather', { on: $('autoW').checked, every: +$('autoWSec').value, mode: $('autoWMode').value });
const sendAutoLight = () => ctrl('auto_light', { on: $('autoL').checked, every: +$('autoLSec').value });
['autoW', 'autoWSec', 'autoWMode'].forEach((id) => { $(id).onchange = sendAutoWeather; });
['autoL', 'autoLSec'].forEach((id) => { $(id).onchange = sendAutoLight; });
setInterval(() => {                                  // the server is the truth (the setting is shared by every open page)
  if (!state.last || !state.last.auto) return;
  const a = state.last.auto, f = document.activeElement;
  if (f !== $('autoW')) $('autoW').checked = a.weather.on;
  if (f !== $('autoWSec')) $('autoWSec').value = a.weather.every;
  if (f !== $('autoWMode')) $('autoWMode').value = a.weather.mode;
  if (f !== $('autoL')) $('autoL').checked = a.light.on;
  if (f !== $('autoLSec')) $('autoLSec').value = a.light.every;
}, 500);

export { hud, showCamera, fillReplayList, updateReplayUI };
