// State shared by the network handlers, the UI and the render loop.

export const state = {
  last: null,                                                                    // latest "state" message from the server
  camMode: 'chase',                                                              // 'chase' | 'orbit' | 'top'
  prevGoals: 0,                                                                  // goals count in the previous message (detects a new goal)
  pose: { x: 0, y: 0, z: 80, yaw: 0 },                                           // smoothed drone pose that is rendered
  target: { x: 0, y: 0, z: 80, yaw: 0 },                                         // latest pose from the server that pose eases toward
};
