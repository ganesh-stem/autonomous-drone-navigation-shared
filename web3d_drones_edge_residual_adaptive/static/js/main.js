// Entry point: wires the server messages to the scene, runs the render loop and the camera.

import * as THREE from 'three';
import { $, T, renderer, scene, camera, controls, canvas, sky, lerp, angDiff } from './core.js';
import { ground, TILE, BLOCK, city } from './city.js';
import { updateClouds } from './clouds.js';
import { drone, tilt, bodyMat, rotors, shadow, DRONE_COLOR, resetTrail, updateSensors, updateTrail } from './drone.js';
import { goalBurst, updateGoalMarker } from './goal.js';
import { setObstacles, setFlyers, animateObstacles } from './obstacles.js';
import { setTargets, trunkMesh, coneMesh, personMesh, rockMesh } from './targets.js';
import { updateTraffic, setCars, trafficBody, trafficStats, trafficData } from './traffic.js';
import { updateWind, updateLook } from './environment.js';
import { hud, showCamera, fillReplayList, updateReplayUI } from './ui.js';
import { connect } from './net.js';
import { state } from './state.js';
import { setBuildingLightLevel } from './buildings.js';

// ── server messages ──────────────────────────────────────────────────────────────
//   "state"     : drone pose, 28 sensor rays, goal, wind, battery, KPIs, telemetry counters
//   "obstacles" : axis-aligned boxes [x0,y0,z0,x1,y1,z1] for the chunks around the drone
//   "targets"   : vehicles, people, trees and hot rocks around the drone
//   "camera"    : the drone's RGB / thermal / height views with the detector's boxes (PNG)
//   "replays"   : the list of recorded runs
connect({
  state(msg) {
    const { pose, target } = state;
    if (state.last) {
      const dx = msg.drone.x - target.x, dy = msg.drone.y - target.y, dz = msg.drone.z - target.z;
      if (dx * dx + dy * dy + dz * dz > 250 * 250) { Object.assign(pose, msg.drone); resetTrail(); }   // respawn: snap, restart trail
    }
    state.last = msg;
    target.x = msg.drone.x; target.y = msg.drone.y; target.z = msg.drone.z; target.yaw = msg.drone.yaw;
    if (msg.goals > state.prevGoals) goalBurst();
    state.prevGoals = msg.goals;
    hud(msg);
    updateReplayUI(msg.replay);
  },
  obstacles: (msg) => setObstacles(msg.boxes),
  replays: (msg) => fillReplayList(msg.files, msg.error),
  targets: setTargets,
  cars: setCars,
  flyers: setFlyers,                                                             // the roaming birds and patrol drones, each a real obstacle of the simulation, where it is now
  camera: showCamera,
});

// ── render loop ──────────────────────────────────────────────────────────────────
function resize() { const w = canvas.clientWidth, h = canvas.clientHeight; renderer.setSize(w, h, false); camera.aspect = w / h; camera.updateProjectionMatrix(); }
addEventListener('resize', resize); resize();
window.edgeDebug = { drone, setBuildingLightLevel, camera, trunkMesh, coneMesh, vehicleMesh: trafficBody, personMesh, rockMesh, trafficStats, trafficData, freeCam: false };

let roll = 0, pitch = 0, lastYaw = 0, spin = 0;
const camPos = new THREE.Vector3(-80, 60, 0), camLook = new THREE.Vector3();
const _fwd = new THREE.Vector3(), _want = new THREE.Vector3(), _lookT = new THREE.Vector3(), _up22 = new THREE.Vector3(0, 22, 0);      // reused every frame (no new objects to collect)
const clock = new THREE.Clock(); let fpsFrames = 0, fpsT0 = performance.now();

function frame() {
  requestAnimationFrame(frame);
  const dt = Math.min(0.05, clock.getDelta());
  spin += dt * 70;                                                                // propeller angle: by the clock, not per frame (a fixed step per frame looked frozen when it matched the blade symmetry)
  const { pose, target, last } = state;
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
  rotors.forEach((r, i) => { r.rotation.y = (i % 2 ? -1 : 1) * spin; });            // neighbouring propellers turn in opposite directions
  animateObstacles(performance.now() / 1000, dt);
  bodyMat.color.set(last && last.drone.colliding ? 0xff4d4d : DRONE_COLOR);
  shadow.position.set(drone.position.x, 0.5, drone.position.z);
  shadow.scale.setScalar(1 + Math.max(0, pose.z) / 250); shadow.material.opacity = Math.max(0.08, 0.4 - pose.z / 700);

  if (last) {
    updateGoalMarker(last.goal, last.goalRadius || 30, dt);
    updateSensors($('showSensors').checked, last.sensors);
    updateTrail($('showTrail').checked);
  }

  // infinite ground + sky + clouds follow the drone
  ground.position.set(Math.round(drone.position.x / TILE) * TILE, -0.05, Math.round(drone.position.z / TILE) * TILE);
  { const bx = Math.floor(drone.position.x / BLOCK), by = Math.floor(-drone.position.z / BLOCK); if (bx !== city.bx || by !== city.by) city.rebuild(bx, by); city.animate(performance.now() / 1000); }
  sky.position.copy(camera.position);
  updateClouds(drone.position);
  updateTraffic();                                                               // draws the server's cars (driving + parked) and the traffic lights

  // camera
  const fwd = _fwd.set(Math.cos(yr), 0, -Math.sin(yr));
  if (window.edgeDebug && window.edgeDebug.freeCam) { renderer.render(scene, camera); return; }
  if (state.camMode === 'chase') {
    const want = _want.copy(drone.position).addScaledVector(fwd, -55).add(_up22);
    camPos.lerp(want, 0.12); camLook.lerp(_lookT.copy(drone.position).addScaledVector(fwd, 30), 0.2);
    camera.position.copy(camPos); camera.lookAt(camLook);
  } else if (state.camMode === 'top') {
    camera.position.set(drone.position.x, drone.position.y + 520, drone.position.z + 1); camera.lookAt(drone.position);
  } else { controls.target.lerp(drone.position, 0.2); controls.update(); }

  updateWind(dt); updateLook();
  renderer.render(scene, camera);
}
frame();
