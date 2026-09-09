"use strict";
const $ = id => document.getElementById(id);
const video = $("video"), overlay = $("overlay"), ctx = overlay.getContext("2d");
let sid = null, ws = null, pc = null, media = null, fileUrl = null, selected = null;
let running = false, pending = false, objects = [], loop = null, generation = 0;
const capture = document.createElement("canvas");
function message(text) { $("message").textContent = text; }
async function api(path, method = "GET", body = undefined, binary = false) {
  const headers = {Authorization: `Bearer ${$("token").value}`};
  if (!binary && body !== undefined) headers["Content-Type"] = "application/json";
  const response = await fetch(`/api/v1${path}`, {method, headers, body: binary ? body : body === undefined ? undefined : JSON.stringify(body)});
  if (response.status === 204) return null;
  const data = await response.json();
  if (!response.ok) throw new Error(data.error?.message || `HTTP ${response.status}`);
  return data;
}
function requireSession() { if (!sid) throw new Error("Conecta primero un vídeo o una cámara"); }
function safe(fn) { return async (...args) => { try { await fn(...args); } catch (error) { message(error.message); } }; }
async function configure() {
  requireSession();
  const split = value => value.split(",").map(s => s.trim()).filter(Boolean);
  await api(`/sessions/${sid}/config`, "PUT", {detect: split($("classes").value), ignore: split($("ignore").value), threshold: $("threshold").value === "" ? null : Number($("threshold").value), record_labels: split($("record-labels").value)});
  selected = null; objects = []; render();
}
function choose(id) { selected = id; $("selected").textContent = `Objeto seleccionado: ID ${id}`; render(); }
function render() {
  overlay.width = Math.round(overlay.clientWidth * devicePixelRatio);
  overlay.height = Math.round(overlay.clientHeight * devicePixelRatio);
  ctx.clearRect(0, 0, overlay.width, overlay.height);
  const scale = Math.min(overlay.width/(video.videoWidth || 1), overlay.height/(video.videoHeight || 1));
  const w = video.videoWidth*scale, h = video.videoHeight*scale;
  const ox = (overlay.width-w)/2, oy = (overlay.height-h)/2;
  $("objects").replaceChildren();
  for (const o of objects) {
    const [x1,y1,x2,y2] = o.box;
    const x = ox+x1*w, y = oy+y1*h;
    ctx.strokeStyle = o.id === selected ? "#fff1a8" : "#70e4bf";
    ctx.lineWidth = 2*devicePixelRatio;
    ctx.setLineDash(o.predicted ? [5, 4] : []);
    ctx.strokeRect(x, y, (x2-x1)*w, (y2-y1)*h);
    ctx.font = `${12*devicePixelRatio}px system-ui`;
    ctx.fillStyle = ctx.strokeStyle;
    ctx.fillText(`${o.label || o.class} · ${o.id}`, x+3, Math.max(15, y-5));
    const button = document.createElement("button");
    button.textContent = `${o.label || o.class} · ID ${o.id} · ${Math.round(o.confidence*100)}%`;
    button.title = o.description || "";
    button.classList.toggle("active", o.id === selected);
    button.onclick = () => choose(o.id);
    $("objects").append(button);
  }
}
function result(data) {
  if (data.error) { message(data.error.message); return; }
  if (!data.objects) return;
  objects = data.objects;
  if (!objects.some(o => o.id === selected)) selected = null;
  $("status").textContent = `${data.backend === "demo" ? "DEMO · colores sintéticos" : "Modelos activos"} · frame ${data.frame} · ${data.processing_ms} ms`;
  if (data.alerts.length) message(`Referencia similar: ${data.alerts.map(a => `ID ${a.track_id} (${Math.round(a.similarity*100)}%)`).join(", ")}`);
  render();
}
async function stop() {
  generation++; running = false; pending = false;
  clearInterval(loop); loop = null;
  if (ws) ws.close(); ws = null;
  if (pc) pc.close(); pc = null;
  if (media) media.getTracks().forEach(t => t.stop()); media = null;
  video.pause(); video.srcObject = null; video.removeAttribute("src"); video.load();
  if (fileUrl) URL.revokeObjectURL(fileUrl); fileUrl = null;
  const old = sid; sid = null;
  if (old) { try { await api(`/sessions/${old}`, "DELETE"); } catch (e) { message(e.message); } }
  selected = null; objects = []; render(); $("stop").disabled = true;
  $("placeholder").hidden = false; $("status").textContent = "Desconectado";
}
async function start(file) {
  await stop();
  try {
    const info = await api("/status");
    const created = await api("/sessions", "POST"); sid = created.id;
    await configure();
    if (file) { fileUrl = URL.createObjectURL(file); video.src = fileUrl; video.loop = true; }
    else {
      media = await navigator.mediaDevices.getUserMedia({video: {width: {ideal: 960}, height: {ideal: 540}, frameRate: {ideal: 15, max: 30}}, audio: false});
      video.srcObject = media;
    }
    await video.play(); running = true; const epoch = generation;
    $("placeholder").hidden = true; $("stop").disabled = false;
    $("status").textContent = info.backend === "demo" ? "DEMO · usa objetos de color sobre fondo neutro" : "Conectando modelos";
    if ($("transport").value === "webrtc") {
      const stream = media || (video.captureStream ? video.captureStream() : video.mozCaptureStream?.());
      if (!stream) throw new Error("Este navegador no permite transmitir archivos por WebRTC; usa WebSocket JPEG");
      pc = new RTCPeerConnection({iceServers: []});
      const channel = pc.createDataChannel("results", {ordered: false, maxRetransmits: 0});
      channel.onmessage = event => { if (epoch === generation) result(JSON.parse(event.data)); };
      stream.getVideoTracks().forEach(track => pc.addTrack(track, stream));
      pc.onconnectionstatechange = () => { if (pc?.connectionState === "failed") message("La conexión WebRTC ha fallado. Comprueba la red o usa WebSocket JPEG."); };
      await pc.setLocalDescription(await pc.createOffer());
      await new Promise((resolve, reject) => {
        if (pc.iceGatheringState === "complete") { resolve(); return; }
        const timer = setTimeout(() => reject(new Error("Tiempo de negociación ICE agotado")), 10000);
        pc.onicegatheringstatechange = () => { if (pc?.iceGatheringState === "complete") { clearTimeout(timer); resolve(); } };
      });
      await pc.setRemoteDescription(await api(`/sessions/${sid}/offer`, "POST", {sdp: pc.localDescription.sdp, type: "offer"}));
    } else {
      ws = new WebSocket(`${location.protocol === "https:" ? "wss" : "ws"}://${location.host}/api/v1/sessions/${sid}/stream`);
      ws.onopen = () => ws.send($("token").value);
      ws.onclose = () => { if (running && epoch === generation) message("Stream cerrado. Vuelve a conectar."); };
      ws.onmessage = event => {
        if (epoch !== generation) return;
        pending = false; const data = JSON.parse(event.data);
        if (data.ready) loop = setInterval(() => sendFrame(epoch), 100);
        else result(data);
      };
    }
  } catch (error) { await stop(); throw error; }
}
async function sendFrame(epoch) {
  if (!running || pending || !ws || ws.readyState !== WebSocket.OPEN || video.readyState < 2) return;
  pending = true;
  const factor = Math.min(1, 960/video.videoWidth, 540/video.videoHeight);
  capture.width = Math.max(1, Math.round(video.videoWidth*factor)); capture.height = Math.max(1, Math.round(video.videoHeight*factor));
  capture.getContext("2d").drawImage(video, 0, 0, capture.width, capture.height);
  const blob = await new Promise(resolve => capture.toBlob(resolve, "image/jpeg", 0.75));
  if (epoch !== generation) return;
  if (blob && ws?.readyState === WebSocket.OPEN) ws.send(blob); else pending = false;
}
overlay.onclick = event => {
  const rect = overlay.getBoundingClientRect();
  const scale = Math.min(rect.width/video.videoWidth, rect.height/video.videoHeight);
  const w = video.videoWidth*scale, h = video.videoHeight*scale;
  const x = (event.clientX-rect.left-(rect.width-w)/2)/w, y = (event.clientY-rect.top-(rect.height-h)/2)/h;
  const o = objects.find(o => x >= o.box[0] && x <= o.box[2] && y >= o.box[1] && y <= o.box[3]);
  if (o) choose(o.id);
};
$("start").onclick = safe(() => start());
$("file").onchange = safe(event => event.target.files[0] && start(event.target.files[0]));
$("stop").onclick = safe(stop);
$("filters").onclick = safe(configure);
$("save").onclick = safe(async () => {
  requireSession(); if (!selected) throw new Error("Selecciona un objeto visible");
  await api(`/sessions/${sid}/tracks/${selected}/labels`, "POST", {label: $("label").value});
  message("Etiqueta guardada en la memoria persistente"); await refresh();
});
async function upload(input, learn) {
  const file = input.files[0]; if (!file) return;
  if (learn) await api(`/labels?label=${encodeURIComponent($("label").value)}&category=${encodeURIComponent($("category").value)}`, "POST", file, true);
  else { requireSession(); await api(`/sessions/${sid}/reference`, "PUT", file, true); }
  message(learn ? "Imagen registrada" : "Búsqueda visual activada"); input.value = ""; await refresh();
}
$("reference-learn").onchange = safe(event => upload(event.target, true));
$("reference-search").onchange = safe(event => upload(event.target, false));
$("clear-search").onclick = safe(async () => { requireSession(); await api(`/sessions/${sid}/reference`, "DELETE"); message("Búsqueda desactivada"); });
$("send").onclick = safe(async () => { requireSession(); await api(`/sessions/${sid}/commands`, "POST", {text: $("command").value, track_id: selected}); message("Comando aplicado"); });
const Speech = window.SpeechRecognition || window.webkitSpeechRecognition;
$("voice").disabled = !Speech;
$("voice").onclick = safe(() => {
  const speech = new Speech(); speech.lang = "es-ES";
  speech.onresult = event => { $("command").value = event.results[0][0].transcript; message("Revisa el texto y pulsa Enviar"); };
  speech.onerror = event => message(`Dictado no disponible: ${event.error}`); speech.start();
});
async function refresh() {
  const items = await pages("/labels"); $("labels").replaceChildren();
  for (const item of items) {
    const row = document.createElement("div"), remove = document.createElement("button");
    row.textContent = `${item.label} · ${item.category} · ${item.examples} vistas `;
    remove.textContent = "Eliminar";
    remove.onclick = safe(async () => { await api(`/labels/${item.id}`, "DELETE"); await refresh(); });
    row.append(remove); $("labels").append(row);
  }
}
$("refresh").onclick = safe(refresh);
async function pages(path) {
  const items = []; let cursor = 0;
  do {
    const page = await api(`${path}${path.includes("?") ? "&" : "?"}after=${cursor}&limit=200`);
    items.push(...page.items); cursor = page.next_cursor;
  } while (cursor !== null);
  return items;
}
$("history").onclick = safe(async () => {
  const label = $("event-label").value;
  const items = await pages(`/events${label ? `?label=${encodeURIComponent(label)}` : ""}`);
  $("events").replaceChildren();
  for (const event of items.reverse().slice(0, 100)) {
    const row = document.createElement("div");
    row.textContent = `${new Date(event.timestamp*1000).toLocaleString()} · ${event.label || event.category} · ${event.kind}`;
    $("events").append(row);
  }
});
$("clips-refresh").onclick = safe(async () => {
  const data = await api("/clips"); $("clips").replaceChildren();
  for (const clip of data.items) {
    const button = document.createElement("button");
    button.textContent = `${clip.labels.join(", ")} · ${new Date(clip.started*1000).toLocaleString()}`;
    button.onclick = safe(async () => {
      const response = await fetch(`/api/v1/clips/${clip.id}`, {headers: {Authorization: `Bearer ${$("token").value}`}});
      if (!response.ok) throw new Error("No se pudo descargar el clip");
      const url = URL.createObjectURL(await response.blob()), link = document.createElement("a");
      link.href = url; link.download = `${clip.id}.avi`; link.click(); setTimeout(() => URL.revokeObjectURL(url), 10000);
    });
    $("clips").append(button);
  }
});
window.addEventListener("resize", render);
