// Network: the WebSocket to the server and the control helpers (window.ctrl is also used by the page's buttons).

let ws = null;
export const send = (o) => { if (ws && ws.readyState === WebSocket.OPEN) ws.send(JSON.stringify(o)); };
export const ctrl = (action, value) => send({ type: 'control', action, value });
window.ctrl = ctrl;

// handlers: { state(msg), obstacles(msg), replays(msg), targets(msg), camera(msg) }, called by message type
export function connect(handlers) {
  ws = new WebSocket(`${location.protocol === 'https:' ? 'wss' : 'ws'}://${location.host}/ws`);
  ws.onmessage = (ev) => { const msg = JSON.parse(ev.data); const h = handlers[msg.type]; if (h) h(msg); };
  ws.onopen = () => send({ type: 'control', action: 'list_replays' });
}
