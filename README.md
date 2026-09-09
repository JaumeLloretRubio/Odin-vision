# Odin Vision

**Beta software under active testing.** The code runs, the automated test suite passes and the recognition
thresholds were calibrated on real video, but the system has not been validated on a physical camera, on an
RTSP feed or in a browser on real hardware. Treat every number below as a laboratory measurement, not as an
operational guarantee. Do not use this for anything safety critical.

Odin Vision is a client/server video system that detects objects, tracks them across frames, writes a short
description of each one and remembers specific visual instances **without retraining any model**. You give an
object a name once and the server stores its embedding vectors in SQLite. Later, in a new session, the same
object is matched back to that name by cosine similarity.

The important distinction: a **category** comes from the detector vocabulary (person, car, drone) and is shared
by every object of that type. A **label** belongs to one visual instance and is taught by you.

## How it works

1. A client sends JPEG frames to the server over HTTP, WebSocket or WebRTC. The browser page bundled with the
   server can capture a webcam or play back a local video file.
2. The server detects objects in the frame and filters them by the categories you asked for.
3. A tracker assigns a stable numeric ID to each object across frames using IoU plus velocity prediction, so an
   object keeps its ID while it moves.
4. Crops of each track are embedded into a vector. The vector is compared against everything stored in memory.
   A match above the similarity threshold, and far enough ahead of the runner up to clear the margin, returns
   the stored label.
5. Optionally a captioning model writes one description per track.
6. Every sighting, new object and label assignment is written to an episodic event log in SQLite.

## Current status

Verified in this repository:

- 40 automated tests pass with no skips when all extras are installed.
- Threshold calibration on free laboratory video with 47 instances, 7166 verification pairs and 849
  identification probes: AUC 0.958, top 1 accuracy 0.894. This produced the defaults of the `models` backend,
  a threshold of 0.91 and a margin of 0.02.
- Simulated live streaming from a video file, including labelling an object mid stream without cutting the
  connection, recognising it again in a brand new session (46 of 47 frames) and recording a clip to disk.
- A real WebRTC negotiation reaching the `connected` state.

Measured throughput on CPU only, with no GPU available: about 1.3 FPS with `describe: false` and about 0.2 FPS
with descriptions on. That is usable for low cadence surveillance or for deferred analysis of recorded footage.
It is not usable for smooth real time video.

Not yet verified: a physical camera, an RTSP source, GPU inference and the browser interface on real hardware.

## Requirements

Python 3.10 to 3.12 is recommended, because the optional dependencies do not all support newer versions yet.
The base install needs only FastAPI, uvicorn, numpy, Pillow and OpenCV headless. The `models` backend adds
PyTorch, Ultralytics and Transformers, which is a multi gigabyte download.

Optional extras:

| Extra | What it adds |
|---|---|
| `models` | YOLO-World, SigLIP and BLIP for the real backend |
| `webrtc` | aiortc, for the WebRTC transport |
| `client` | httpx, for the headless camera client |
| `test` | pytest, httpx and ruff |

## Quick start with the demo backend

The demo backend needs no model weights and no download. From this directory, in PowerShell:

```powershell
python -m venv .venv
.venv/Scripts/python -m pip install -e ".[test,webrtc]"
.venv/Scripts/python -m odin_vision --backend demo
```

Then open <http://127.0.0.1:8000>.

The demo detector finds saturated colour blobs on a neutral background and labels every one of them with the
first category in your configured list. **It does not recognise people or any real object.** It exists so you
can exercise the whole loop of capture, selection, memory and recall without downloading a single model. Put a
red, green or blue object in front of the camera and it will pick it up.

## Using the web interface

The bundled page is in Spanish. The button names below are the literal on screen text.

1. Press **Conectar cámara** to start the webcam, or use **Abrir vídeo** to play a local file instead.
2. Click a box in the video or an entry in the object list to select an object.
3. Type a name in **Etiqueta** and press **Guardar objeto**. The server embeds the stored crops of that track
   and writes them to memory.
4. Press **Detener**, connect again, and the object is still recognised. Memory lives on disk, not in the
   session.
5. To teach from a photo instead of from live video, fill in the label and the category and use
   **Registrar imagen**. The photo must be a crop of the object itself, not a whole scene.
6. **Buscar similares en directo** uploads an image as a temporary reference. Any object similar to it raises
   an alert in the interface and in the frame result payload. **Quitar búsqueda** clears it.
7. **Aplicar filtros** pushes the detection settings, the ignore list, the minimum similarity and the list of
   labels to record as clips.
8. The memory panel lists stored labels, sightings filtered by label and recorded clips.

## Real models backend

```powershell
.venv/Scripts/python -m pip install -e ".[models,webrtc,test]"
$env:ODIN_DATA_DIR = "data/real"
$env:ODIN_DEVICE = "cpu"
.venv/Scripts/python -m odin_vision --backend models
```

Set `ODIN_DEVICE` to `cuda:0` instead if you have a PyTorch build with matching CUDA. The first run downloads
the configured weights. If a dependency, the weights or enough memory are missing, startup fails with an error.
The server never falls back to the demo backend silently.

| Component | Implementation |
|---|---|
| Detection with an editable vocabulary | YOLO-World, `yolov8s-worldv2.pt` |
| Tracking | IoU plus velocity prediction, per stream |
| Embeddings | SigLIP, `google/siglip-base-patch16-224` |
| Descriptions | BLIP, `Salesforce/blip-image-captioning-base`, once per track |
| Instance memory | SQLite plus exact cosine similarity, several views per label |
| Episodic memory | Sightings, new objects, label assignments and an optional location |
| Video transport | JPEG over WebSocket, or WebRTC through aiortc |

Generated descriptions may come out in English because the captioning model is English only. The label is
always exactly the text you typed. Saving a label never changes any model weight.

## Headless camera client

On a capture device that only needs to push frames, install the `client` extra. It needs neither PyTorch nor
any weights, since all inference happens on the server.

```powershell
python -m pip install -e ".[client]"
$env:ODIN_TOKEN = "the-token-configured-on-the-server"
python -m odin_vision.camera --server https://server.example --source 0 --fps 5 --classes person,drone
```

| Flag | Default | Meaning |
|---|---|---|
| `--server` | `http://127.0.0.1:8000` | Server base URL |
| `--source` | `0` | Camera index, file path or video URL the client can open |
| `--fps` | `5` | Maximum frames sent per second |
| `--width` | `960` | Frames are downscaled to this width before encoding |
| `--classes` | `person,car,backpack,drone` | Categories pushed to the session config |
| `--max-frames` | `0` | Stop after N frames, 0 means run until the source ends or Ctrl+C |

`--source` accepts anything OpenCV can open, including an RTSP URL. The client decodes locally and uploads
JPEG over HTTP, so this is **not** a direct H.264 transport. Results are printed as JSON Lines on stdout and
the session ID is printed on stderr. Use that ID to send filters, labels and commands to the API while capture
is running. Ctrl+C releases both the camera and the server session. Real camera capture still needs validation
on your target hardware.

## Configuration

Environment variables read at startup:

| Variable | Default | Purpose |
|---|---|---|
| `ODIN_BACKEND` | `demo` | `demo` or `models` |
| `ODIN_DATA_DIR` | `data` | Directory for the SQLite database and recorded clips |
| `ODIN_TOKEN` | empty | Shared bearer token, required to listen on a network interface |
| `ODIN_DEVICE` | `cpu` | Inference device, for example `cuda:0` |
| `ODIN_DETECTOR` | `yolov8s-worldv2.pt` | YOLO-World weights |
| `ODIN_ENCODER` | `google/siglip-base-patch16-224` | SigLIP model ID or local directory |
| `ODIN_CAPTIONER` | `Salesforce/blip-image-captioning-base` | BLIP model ID or local directory |
| `ODIN_THRESHOLD` | backend default | Minimum similarity to accept a match, 0 to 1 |
| `ODIN_MARGIN` | backend default | Minimum gap over the runner up label, 0 to 1 |

Calibrated defaults are 0.91 threshold and 0.02 margin for `models`, and 0.85 and 0.05 for `demo`. The demo
backend uses colour histograms, whose similarity scale is not comparable to a learned encoder, which is why its
defaults differ.

Command line flags of the server: `--host` (default `127.0.0.1`), `--port` (default `8000`) and `--backend`.
Binding to anything other than loopback is refused unless `ODIN_TOKEN` is set.

Per session settings, sent to the config endpoint or through the filter form:

| Field | Default | Purpose |
|---|---|---|
| `detect` | `person, car, backpack, drone` | Detector vocabulary for this session |
| `ignore` | empty | Categories dropped after detection |
| `threshold`, `margin` | backend defaults | Recognition sensitivity |
| `detector_interval` | `1` | Run the detector once every N frames |
| `recognition_interval` | `15` | Re run memory matching for a track every N frames |
| `describe` | `true` | Generate a description per track, the main cost on CPU |
| `record_labels` | empty | Record a clip whenever one of these labels is visible |

### Deployment notes

For access from another device, set `ODIN_TOKEN` and run with `--host 0.0.0.0`. Put an HTTPS and WSS reverse
proxy in front, because browsers normally require a secure context before granting camera and microphone
access. API requests are rejected if the `Origin` header does not match the `Host` header, so a browser on a
different origin cannot call the API.

The service is **single user**. One shared token, and one shared memory across all of its streams. There are no
accounts and no tenants. Do not run several workers against the same data directory, because stream state,
memory revisions and model serialisation all live inside a single process. Inference is serialised behind one
lock, so a request arriving while another one is being processed gets HTTP 429 rather than piling up in a
queue.

## HTTP API

Every route is prefixed with `/api/v1` and requires `Authorization: Bearer <token>` when a token is configured.
Interactive documentation is served at `/docs`.

| Method and path | Purpose |
|---|---|
| `GET /status` | Backend name, encoder fingerprint, active sessions and frame size limits |
| `POST /sessions` | Create a session, returns its ID and resolved config |
| `DELETE /sessions/{sid}` | Close a session and its WebRTC peer |
| `PUT /sessions/{sid}/config` | Replace the session config, resets the tracker |
| `PUT /sessions/{sid}/location` | Attach a latitude and longitude to the events of this session |
| `POST /sessions/{sid}/frames` | Upload one JPEG or PNG frame as a raw body, returns the frame result |
| `GET /sessions/{sid}/result` | Last frame result without sending a new frame |
| `POST /sessions/{sid}/tracks/{tid}/labels` | Teach a label from the stored crops of a visible track |
| `POST /sessions/{sid}/commands` | Send a text command, see the command section |
| `PUT /sessions/{sid}/reference` | Upload an image as the live similarity search reference |
| `DELETE /sessions/{sid}/reference` | Clear that reference |
| `POST /labels?label=&category=` | Teach a label directly from an uploaded image |
| `GET /labels?after=&limit=` | List stored labels, cursor paginated |
| `DELETE /labels/{label_id}` | Delete a label and all of its examples |
| `GET /events?after=&limit=&label=&since=&kind=` | Read the episodic event log |
| `GET /clips` | List recorded clips with their metadata |
| `GET /clips/{clip_id}` | Download one clip as AVI |
| `WS /sessions/{sid}/stream` | Send the token as the first text message, then binary frames, and receive one JSON result per frame |
| `POST /sessions/{sid}/offer` | WebRTC SDP offer, answered with the SDP answer |

A frame result contains the frame number, a timestamp, the image size, the processing time in milliseconds and
a list of objects. Each object carries its track ID, a box in normalised coordinates, its category, its label if
recognised, its description, the detector confidence, the memory similarity, the similarity against the live
reference and a `predicted` flag that is true when the box comes from the tracker rather than from a fresh
detection. Alerts triggered by the live reference come in a separate list.

Errors are returned as `{"error": {"code": ..., "message": ...}}`. Uploads are capped at 2 MB and roughly 2
megapixels, and JSON bodies at 120 KB.

## Text and voice commands

Recognised patterns, in Spanish, matched by regular expression:

- `Guarda como Falcon` or `Llama al objeto seleccionado como Falcon`. Requires a selected object.
- `Guarda la persona seleccionada como Objetivo A`. This stores a visual appearance. It is not an identity
  claim about a person.
- `Ignora coches` or `No me marques coches`. Adds a category to the ignore list.
- `Solo muéstrame personas y drones`. Replaces the detection vocabulary and clears the ignore list.

Anything else is rejected with an error. There is no free form interpreter behind this, only those patterns.
Dictation uses the browser speech API, which on most browsers sends audio to an external service. The
transcript is shown to you and nothing is sent until you press **Enviar**. There is no local speech
recognition built into the server.

## Data on disk

Everything is written under `ODIN_DATA_DIR`:

- `memory.sqlite3` holds labels, their example vectors and the event log, in WAL mode.
- `clips/` holds recorded clips as AVI with MJPEG, each next to a JSON sidecar carrying the original frame
  timings and the labels that triggered the recording. A clip is capped at 10 seconds and 100 MB.

The database records a fingerprint of the encoder used to create it. Starting the server with a different
encoder against the same directory fails on purpose, because vectors from two different encoders cannot be
compared. Use a separate data directory per encoder.

Data directories, model weights and local logs are ignored by git, so a fresh clone contains the package, the
tests and the helper scripts only. Everything the server needs at runtime is created on first start or
downloaded on first use.

## Tests and checks

```powershell
.venv/Scripts/python -m pytest -q
.venv/Scripts/python -m ruff check odin_vision tests scripts
node --check odin_vision/static/app.js
```

No test downloads a model. The clip and RTP tests use the real libraries and are skipped explicitly if OpenCV
or aiortc is missing. For a full acceptance run, install every extra and confirm that nothing is skipped.

`constraints.txt` pins a known good dependency set if you need reproducible installs.

## Helper scripts

These are development tools, not part of the server. They all need input media that you supply yourself, since
no media is shipped in this repository.

| Script | Purpose |
|---|---|
| `scripts/acceptance_models.py` | Manual end to end acceptance run against the real backend with your own images |
| `scripts/lab_calibration.py` | Derive the similarity threshold and margin from a video, building ground truth from track identity |
| `scripts/lab_stream.py` | Replay a video at its real cadence through the WebSocket to test live labelling and recall in a fresh session |
| `scripts/render_annotated.py` | Render a video with boxes, IDs and labels drawn on top, for visual review |

## Known limits

- Similarity is not a calibrated probability. A score of 0.91 does not mean 91 percent confidence.
- One photo does not guarantee recognition of a person wearing different clothes, at low resolution or partly
  occluded.
- The tracker can swap IDs when similar objects cross each other.
- The calibration set is small, 47 instances across two categories. It is enough to pick sensible defaults, not
  enough to promise operational accuracy at scale.
- All published figures are CPU figures. No GPU was available during testing.
- Tune the threshold and the margin with positive and negative examples from your own camera before trusting
  any result.

## License

[PolyForm Noncommercial License 1.0.0](LICENSE.md). Free to use, modify and share for any noncommercial
purpose. Commercial use requires a separate license from the author.
