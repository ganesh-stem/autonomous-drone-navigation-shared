// 3D viewer for the edge drone (infinite city with weather). Entry point only: the code lives in ./js/
//   core.js         renderer, scene, camera, lights, sky dome, shared helpers
//   city.js         road lattice ground + parks / fountains / gardens / playgrounds / courts (and the "free ground" rules)
//   clouds.js       wrapping cloud field
//   drone.js        the player's drone (a modern quadcopter), sensor rays, trail
//   cars.js         the detailed car model (three body styles) drawn with instanced meshes
//   traffic.js      cars that drive the roads: lanes, traffic lights, turns, rush hour at dusk / few cars at night
//   goal.js         the goal "stargate" marker
//   obstacles.js    towers, colour-shifting hazard gates, roaming birds and patrol drones
//   targets.js      the sim's cars / people / trees / rocks, kept off roads and scenery
//   environment.js  wind streaks, sky / fog / light for weather and time of day
//   ui.js           HUD, drop-downs, replay controls, camera panel, drawers, weather / light automation
//   net.js          WebSocket + control helpers          state.js   state shared between them
//   main.js         wires messages to the scene, render loop, camera
import './js/main.js';
