// Core: the Three.js renderer, scene, camera, lights, sky dome and the small helpers every other module uses.
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

// the sim's weather / light / wind as last reported by the server (read by the sky and the wind streaks)
const env = { weather: 'calm', light: 'day', wind: [0, 0, 0] };
const hemi = scene.children.find((c) => c.isHemisphereLight);
const lerp = (a, b, t) => a + (b - a) * t;
const angDiff = (a, b) => ((b - a + 540) % 360) - 180;

export { $, T, canvas, renderer, scene, camera, controls, sun, sky, hemi, HORIZON, env, lerp, angDiff };
