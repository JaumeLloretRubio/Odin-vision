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
4. Crops of each observed track are embedded when its category has stored examples or reference search is active.
   The vector is compared against the stored examples of that category.
   A match above the similarity threshold, and far enough ahead of the runner up to clear the margin, returns
   the stored label.
5. Optionally a captioning model writes one description per track.
6. Every sighting, new object and label assignment is written to an episodic event log in SQLite.

## Current status

Verified in this repository:

- 115 automated tests pass with no skips in the validation environment.
- Threshold calibration on free laboratory video with 47 instances, 7166 verification pairs and 849
  identification probes: AUC 0.958, top 1 accuracy 0.894. This produced the defaults of the `models` backend,
  a threshold of 0.91 and a margin of 0.02.
- Simulated live streaming from a video file, including labelling an object mid stream without cutting the
  connection, recognising it again in a brand new session (46 of 47 frames) and recording a clip to disk.
- A real WebRTC negotiation reaching the `connected` state.

Original end to end CPU measurements: about 1.3 FPS with `describe: false` and about 0.2 FPS
with descriptions on. That is usable for low cadence surveillance or for deferred analysis of recorded footage.
It is not usable for smooth real time video.

GPU detection and an exploratory full model flow have also been measured; external network throughput and
full model validation on the declared dependency range remain pending.
Not yet verified: a physical camera, an RTSP source and the browser interface on real hardware.

The 2026-09-14 traffic-video acceptance test passed with real YOLO, SigLIP and BLIP models on GPU:
1,367 streamed frames across live teaching, a new session and a reopened SQLite connection, with no API
errors. Measured throughput was 24.02–26.54 FPS with descriptions enabled. This uses a different video
from the v1 pedestrian streaming test and retains threshold 0.91, margin 0.02 and float32 `highest`.
It used the locally available Transformers 4.44.2, below the supported minimum; it does not close the
supported-runtime validation gap. See [the report and reproduction instructions](validation/traffic-20260914.md).

### Optimization measurements (2026-09-10)

| Operation | Before | After | Scope |
|---|---:|---:|---|
| Demo detection, saturated image | 35.0 ms | 2.16 ms | 320 by 180 pixels, median of 15 runs |
| Memory search, 100 labels | 6.50 ms | 0.143 ms | 16 examples per label, 768 dimensions, warm cache |
| Memory search, 1,000 labels | 75.4 ms | 2.63 ms | Same setup, median of 30 probes |
| Stored example lookup for one label | 42.0 ms | 0.009 ms | SQL lookup of 16 rows among 16,000 examples, median of 20 runs |
| Real person detector, CPU to GPU | 87.4 ms | 9.22 ms | One video frame, 30 runs after warmup, 24 CPU threads versus RTX 5070 Ti Laptop GPU |
| GPU detector including result adapter | 12.06 ms | 9.76 ms | Bulk result transfer, same ten-person frame, 30 runs after warmup |
| Persist 10 frame events | 12.22 ms | 1.25 ms | One transaction per frame, median of 30 batches |
| Persist 100 frame events | 123.24 ms | 1.57 ms | Same setup, retention remains enabled |

The memory cache stores about 49 MB of vectors at 1,000 labels and takes 98 ms to build in this measurement.
Learning, deletion and commits from another SQLite connection invalidate cached data. Search preselection is
vectorized; final contenders use the original dot product to preserve threshold, margin and tie decisions.
Descriptions and crop collection remain active when embeddings can be skipped.
Indexes on example ownership and label category avoid full table scans for selective lookups. Existing databases
receive the indexes on startup. Gallery loading preserves example ID order when resolving ties; full gallery
reads still cost about 70 ms in the indexed SQL benchmark and are amortized by the cache.

The detector returned ten people on both devices; the largest normalized box coordinate difference was
0.000118. These component measurements do not establish end to end throughput or recognition accuracy on GPU.
Bulk extraction transfers coordinates, classes and confidence once per tensor. Its isolated GPU cost fell from
2.53 to 0.37 ms for ten detections (50 runs), with exact output equivalence on the same result tensor.

A GPU run through `Engine.process` handled 100 consecutive video frames at 22.6 FPS including video decoding,
with processing median 35.1 ms and p95 56.6 ms (7–15 objects). This includes detection, tracking, crop collection
and persistent events, with descriptions disabled and an empty gallery; it excludes network transport,
recording and recognition. SigLIP and BLIP remained unloaded.

An exploratory GPU flow through in-process HTTP and WebSocket also exercised SigLIP, BLIP, labelling,
recognition, event storage and recording. After saving four views, the label appeared in all 26 subsequent
frames; median request latency was 74.5 ms, p95 550 ms and throughput 5.04 FPS. A downloaded 3.30 MB clip decoded
to 26 frames, and a fresh Engine/database connection recovered the label while reusing model weights.
This is functional coverage on a short laboratory clip, not an identity accuracy estimate or a network test.
It used Python 3.11, PyTorch 2.7.1+cu128 and locally available Transformers 4.44.2, below the declared minimum
4.48. Repeat on a supported dependency set before treating it as release acceptance. The high p95 remains
an optimization target.

A stage profile of the same unchanged flow measured 33.3 ms median and 231.8 ms p95; this variation is
not a code speedup. Warm caption calls took 139–195 ms on several slow frames, while recording usually
took about 11 ms. Caption token decoding now transfers the whole output tensor to CPU once. An alternating
50-sample comparison on fixed real BLIP output reduced decoding of 10 GPU rows from 0.926 to 0.406 ms and
100 rows from 9.225 to 3.409 ms, with identical text; CPU timings were effectively unchanged. These are
decoding-only measurements on the same exploratory dependency set, not full generation improvements.
The post-change functional rerun passed (26/26 labelled frames, clip download and reopened memory), but
measured 80.0 ms median and 574.0 ms p95. No end-to-end speedup is established; run-to-run variation is
larger than the decoding saving.

Resubmitting the same resolved session configuration now preserves tracking, collected examples,
cached descriptions and the active recording. Configuration changes still reset tracking as before.
HTTP coverage verifies that a repeated configuration produces one continuous clip with consecutive frames.
For a fixed real video frame with 10 detected people, 10 alternating GPU comparisons of configuration
resubmission plus processing measured 948.8 ms before and 40.4 ms after (median), eliminating 10 repeated
caption batches. Object outputs matched except for the deliberately preserved IDs. This benchmark has
an empty gallery and excludes recording and network; it measures redundant filter submissions only.

Tracking groups IoU comparisons by category and evaluates larger groups with NumPy, retaining the
original score/ID/index tie order. A CPU-only synthetic benchmark (30 alternating samples after five
warmups, tracker update only) measured 100 tracks/100 detections of one category at 6.67 to 0.76 ms,
and 1,500 retained tracks/100 detections at 123.69 to 11.39 ms. With 64 categories, the same cases improved
from 0.79 to 0.59 ms and 9.75 to 6.13 ms. Small groups use scalar IoU to avoid matrix overhead.
Boundary, tie, degenerate-box and trajectory tests compare against the previous scalar algorithm;
these timings do not measure detector inference or end-to-end video throughput.
Batching position prediction for 16 or more retained tracks further reduced tracker-update medians:
100 tracks/100 detections of one category went from 0.72 to 0.51 ms; 1,500 tracks/100 detections with
64 categories went from 6.05 to 2.70 ms. These are separate paired measurements against the preceding
IoU optimization, using the same synthetic CPU methodology. Exact clipping and expiry tests cover
empty, small and large track collections.
For crops of at least 32,768 pixels, the sharpness filter uses OpenCV's four-neighbour Laplacian,
discards the border and copies the interior before the existing float32 variance reduction. Smaller
crops retain the previous NumPy calculation. A CPU comparison of 100 alternating samples after five
warmups measured 224×224 crops at 0.339 to 0.219 ms, 960×540 at 5.27 to 3.01 ms, and 1080p at 27.62 to
14.72 ms. The two smaller tested sizes were effectively unchanged. Kernel variances matched exactly;
acceptance tests include the exact variance threshold of 8. These are filter-only synthetic timings.

Recording avoids a redundant PIL copy when the input already matches the clip dimensions. In a
synthetic CPU comparison with 100 alternating samples after five warmups, preparation at 960x540
went from 1.89 to 1.22 ms; including AVI/MJPEG encoding and writes, `Recorder.add` went from 10.86
to 10.41 ms. Decoded output matched exactly across 105 frames at each of three input sizes.
Full recording timings at 320x240 (1.030 to 1.044 ms) and 1080p (46.20 to 47.59 ms) did not show a
benefit; 1080p still uses the same resize path. These results do not establish a general throughput gain.

Event queries bind only the requested filters, allowing an index on `(label, id, timestamp)` to serve
label filtering in pagination order. This replaces the previous `(label, timestamp)` index on startup.
With 10,000 synthetic events and 100 alternating samples, rare-label queries fell from 0.371 to 0.031 ms
and absent-label queries from 0.345 to 0.010 ms. Common-label and unfiltered pages were effectively
unchanged. A common-label query with a late timestamp cutoff regressed from 0.358 to 0.480 ms;
the index prioritizes ID pagination, so this is not a universal query speedup. Separate write tests
measured batches of 100 events at 2.172 to 2.252 ms, and database size at 819,200 to 856,064 bytes.

A subsequent exploratory GPU run replayed a 146-frame sequence ten times through the in-process
HTTP/WebSocket application with recognition, captions, events and recording enabled. All 1,460
requests completed in 244.8 seconds of measured request time (81.4 ms median, 526.0 ms p95).
A learned label appeared in 1,334 responses; this is a presence count, not an identity-accuracy score.
Retention left 16 clips totaling 92,753,138 bytes; all 720 retained frames decoded and matched their
manifest counts. Recognition survived reopening the application and SQLite connection with shared
model weights. Sampled allocated GPU memory rose from 1,448.8 to 1,482.4 MB and then stayed there.
This roughly four-minute replay used Transformers 4.44.2, below the supported minimum, and does not
establish a controlled speedup, peak-memory bound, long-duration stability, or external network/camera acceptance.

A separate demo-backend check used an actual loopback TCP socket with Uvicorn, HTTP and WebSocket
clients. It recognized and recorded 101 frames, retrieved the finalized clip, queried labeled events,
and recovered after an invalid frame. The first 100 WebSocket requests had a 3.82 ms median with
recording enabled on a synthetic 200x120 image. This validates local socket transport only; it does
not measure GPU models, remote networking or a physical camera.

The real-model GPU validation also passed over loopback TCP using Uvicorn and separate HTTP and
WebSocket clients: four learning frames followed by 146 WebSocket frames, all with the learned label
present and object descriptions populated. Four downloaded clips contained all 146 decodable frames;
recognition still worked after restarting the application with shared model weights and reopening SQLite.
This run measured 119.3 ms median and 607.3 ms p95, with sampled allocated GPU memory at 1,448.8 MB.
It used the same exploratory Transformers 4.44.2 environment. These independently timed runs do not
isolate transport overhead or demonstrate a speedup; remote networking and camera validation remain open.

Recognition groups pending objects by category and compares their embeddings in batches of at most
32 queries. Larger batches use matrix multiplication; small workloads retain NumPy's direct reduction.
The existing scalar refinement still resolves candidate scores, thresholds and ties. A synthetic CPU
benchmark of the complete search, with ten alternating samples after three warmups, measured 100
queries against 16,000 examples at 421.5 to 49.9 ms, and ten queries at 34.2 to 4.3 ms. Against 1,600
examples, 100 queries went from 15.8 to 6.1 ms. Results and scores matched the scalar path; tests also
cover threshold boundaries, chunking and dimensions. These are cached-gallery search timings, not
end-to-end GPU gains, and matrix-multiplication performance depends on the installed NumPy/BLAS build.
Repeating the search benchmark in the Python 3.11 model environment measured 100 queries against
16,000 examples at 407.3 to 60.3 ms with identical results. The subsequent real-model loopback check
passed all 146 labeled frames, clip decoding and memory reopening after the batch change. Its gallery
contained only four examples; the independently timed 131.2 ms median does not establish an end-to-end
gain from batching. An Engine test covers mixed categories and reference updates between recognition intervals.

A controlled comparison then alternated individual and batched search within otherwise identical
`Engine.process` runs using shared real GPU model weights, ten objects in a fixed image, a synthetic
16,000-example gallery and active recording. Twenty measured pairs after five warmup pairs reduced
median frame processing from 256.6 to 224.2 ms (12.6%). Object results matched except for track IDs;
both recordings decoded all 26 frames, including initialization and warmups. This includes detection,
embedding, search, tracking and recording, with descriptions already cached. It excludes input decoding,
network transport and cold model loading. The exploratory Transformers 4.44.2 environment and synthetic
gallery limit the claim to this measured workload.

Profiling that warmed workload identified embedding inference as the next major cost. A separate
ten-crop probe using CUDA events and explicit synchronization measured median preprocessing at
24.37 ms, upload at 0.71 ms, GPU feature computation at 125.65 ms, download at 0.20 ms and CPU
normalization at 0.16 ms (ten samples after three warmups). The loaded SigLIP model already used
SDPA with float32 weights. Time attributed to `.cpu()` by a CPU profiler largely included waiting for
GPU computation; these measurements do not justify treating the transfer itself as the bottleneck.

An isolated SigLIP precision experiment alternated float32 `highest` and `high` matrix multiplication
precision on the same ten crops (ten samples after three warmups), measuring GPU feature computation
at 71.34 versus 22.21 ms. Normalized embeddings were not identical: maximum component difference
was 0.000184, minimum same-image cosine was 0.99999946, and maximum cross-image similarity change
against the original embeddings was 0.000206. This experiment restores the original precision and is
not enabled by the application. Recognition decisions near thresholds still need validation before
adoption. [PyTorch's numerical-accuracy guidance](https://docs.pytorch.org/docs/main/notes/numerical_accuracy.html)
describes the precision tradeoff when enabling TF32. These timings isolate model computation and
must not be compared directly with independently timed profiling runs or treated as an application speedup.

A broader TF32 agreement check used 129 person crops from 13 video frames. Each crop was queried
against the original-precision embeddings of the other crops, excluding itself. At threshold 0.91
and margin 0.02, both modes accepted 14 queries and produced the same decision for all 129 queries.
Maximum similarity change was 0.000256. The nearest baseline top score was 0.001856 from the
threshold and the nearest top-two gap was 0.001054 from the margin, so this sample did not exercise
the tightest numerical boundaries. These unlabeled crops measure agreement, not identity accuracy;
TF32 remains an isolated experiment.

A subsequent boundary check found a limit to that agreement: among 1,048,512 two-reference/query
combinations drawn from the same crops, two changed from rejection to acceptance at margin 0.02.
Both changes were reproduced through `Memory.learn`, `Memory.match` and `Memory.match_many` using
the stored embeddings. For example, a query's two similarities moved from approximately
0.965679/0.985677 to 0.965654/0.985679, crossing the margin while remaining above threshold 0.91.
No threshold-only flips occurred across the 16,512 distinct ordered crop pairs. The reference labels
in this check are synthetic, so it demonstrates decision differences rather than identity errors.
TF32 therefore cannot be treated as a drop-in change that preserves all existing recognition decisions.
The optimization priority is to preserve current recognition decisions; TF32 is excluded from the
adopted optimizations on that basis.

An additional SigLIP probe kept float32 `highest` precision and split ten crops into microbatches of
1, 2, 5 or 10. Across ten samples after three warmups, CUDA-event medians were 184.6, 195.5, 160.3
and 185.1 ms respectively. Splitting changed normalized embedding components by up to 0.000000397
relative to the unsplit batch. This remains an isolated experiment: the modest speed difference in
one workload does not establish a general benefit, and exact output equality was not preserved.

RGB PIL inputs processed by the slow `SiglipImageProcessor` now use a 256-value lookup table per
channel for rescaling and normalization. The processor itself generates the table, preserving its
float32 rounding; its usual resize remains in use. Other processors and image modes keep the
original path. A ten-crop NumPy preprocessing probe measured 23.83 to 18.50 ms across 100 samples
after five warmups, with exact input arrays. A paired GPU adapter probe measured 206.10 to 195.90 ms
across 30 samples per path after five warmups, with bit-identical normalized embeddings across all
70 calls. These exploratory measurements used Transformers 4.44.2 and PyTorch 2.7.1+cu128 with
float32 `highest`; supported-version validation and broader workloads remain pending. The regression
suite ran concurrently during part of the adapter probe, so the timings describe this local workload.

A subsequent paired `Engine.process` probe, without concurrent tests, included detection, recognition
against a synthetic 16,000-example gallery and recording of a fixed ten-person frame. Thirty samples
per path after five warmups measured medians of 165.30 ms for the original preprocessing and 162.62 ms
for the lookup. Object outputs matched except for track IDs; both recordings decoded all 36 frames.
The lookup was faster in 17 of 30 pairs. Mean paired savings were 3.55 ms, with a descriptive paired
bootstrap 95% interval of -0.84 to 7.93 ms (20,000 resamples; serial dependence is not modeled).
This run does not establish a stable end-to-end speedup. It excludes network/input decoding, cold
model loading and uncached descriptions, and still uses the experimental Transformers 4.44.2 runtime.

Expanded lookup validation covered 129 crops from 13 frames, in unchanged batches of up to ten.
All normalized embeddings were bit-identical. Against each of the two reference galleries that exposed
TF32 boundary changes, both `Memory.match` and `Memory.match_many` preserved every decision and score
for all 129 queries. The diagnostic initially compared a self-product with a cross-product and observed
a 2.98e-7 rounding difference despite identical vectors. Comparing both query arrays against the same
independent gallery eliminated that difference; self-products also matched each other. This validates
agreement on these inputs, not identity accuracy or universal runtime compatibility.

JPEG inputs already decoded as RGB now reuse the loaded image instead of copying it through another
RGB conversion. The input stream closes explicitly; other modes and PNG/WebP still convert into a
separate image, avoiding retention of their file/decoder state. A paired decode-only benchmark with
50 samples after five warmups measured JPEG at 320×240 from 0.348 to 0.322 ms, at 960×540 from 2.948
to 2.393 ms, and at 1920×1080 from 12.163 to 9.920 ms. Decoded pixels and resized outputs matched
exactly. PNG/WebP timings were effectively unchanged. The synthetic high-entropy image benchmark
raised its byte limit to 20 MB so PNG inputs would fit; application limits are unchanged. These timings
exclude transport and inference. Regression coverage checks JPEG RGB/grayscale/CMYK, PNG RGB/RGBA
and WebP image use after source closure.

HTTP image bodies now retain up to 64 incoming chunks and join them once. A single chunk is reused
directly; more fragmented requests switch to a bytearray to bound the number of retained references.
The byte limit is checked before adding a chunk to either buffer. An isolated body-assembly benchmark
with 200 samples after five warmups measured a 2 MB body in 64 KB chunks at 3.485 to 0.474 ms, with
peak traced allocations falling from 4,160,783 to 2,003,361 bytes. A single 2 MB chunk needed only 968
additional traced bytes; input chunks were preallocated and excluded from these allocation counts.
With 1 KB chunks the same 2 MB body remained effectively unchanged (5.052 to 5.056 ms and roughly
4.16 MB of traced allocations). These are local assembly measurements, excluding sockets, image
decoding, locks and inference. API tests cover empty chunks, single and fragmented bodies, acceptance
at the exact byte limit and rejection one byte above it.

A real loopback HTTP comparison sent a 1,638,529-byte 1080p JPEG through both application variants,
with JPEG decoding and `Engine.process` active but detection disabled. Across 30 alternating samples
after five warmups per sending mode, median request times fell from 17.15 to 15.00 ms when supplied
as one client block, and from 18.29 to 16.23 ms with 64 KB client chunks. Uvicorn delivered these as
13–15 nonempty ASGI chunks, so they exercised the bounded join path. With 1 KB client chunks the
medians were 126.69 and 124.56 ms; server delivery varied across 564–701 chunks, making that timing
more sensitive to scheduling. For the single client block, the paired mean saving was 2.13 ms, with a
descriptive 95% bootstrap interval of 1.90–2.37 ms (20,000 paired resamples, without modeling serial
dependence). Both variants rejected a 2,000,001-byte body with 413 and accepted the next valid frame;
both servers shut down cleanly. This validates local HTTP ingress, not model inference, recording,
remote networks or concurrent-client throughput.

Memory candidate selection now obtains the second-best label score by finding the best example and
then the best example belonging to a different label. This replaces the per-label scatter reduction
and partition, while retaining scalar refinement, the -1 sentinel and existing tie/margin behavior.
Paired cached-search probes (20 samples after five warmups) kept results and scores exact. With
16,000 examples and 100 queries, medians fell from 18.22 to 17.51 ms on Python 3.14/NumPy 2.3.5 and
from 22.46 to 18.11 ms on Python 3.11/NumPy 1.26.4. Smaller workloads were mixed: 1,600 examples and
ten queries changed from 1.339 to 1.376 ms and 1.500 to 1.554 ms respectively. No universal speedup
is claimed. Regression coverage includes multiple leading examples of one label with interleaved
insertion order; the stored 129-crop recognition checks also preserved decisions and scores in both
boundary galleries after this change.

A follow-up isolated ranking from matrix multiplication, using 400 samples after 25 warmups for ten
queries at four gallery sizes. At 1,600 examples, ranking alone improved from 0.0912 to 0.0820 ms on
Python 3.14 and from 0.1336 to 0.0950 ms on Python 3.11. Complete-search medians also decreased,
but paired mean-saving intervals included zero: -0.0390 to 0.0310 ms and -0.00036 to 0.07545 ms
respectively (descriptive 95% bootstrap, 10,000 resamples, serial dependence not modeled). The earlier
small regression was not reproduced, and these data do not justify a gallery-size exception. At
16,000 examples, complete-search medians improved from 3.4760 to 3.3522 ms and 3.4667 to 3.0386 ms.
Results and scores remained exact in every comparison; these probes still exclude inference and I/O.

The matrix-kernel heuristic now uses matrix multiplication for galleries of at least 8,000 examples
or batches with at least 24,000 query/example pairs; smaller batches use `einsum`. The existing
single-query shortcut remains unchanged. Comparing both kernels showed that simply increasing the
pair-count threshold would slow down large galleries with few queries. Final paired full-search
benchmarks covered 30 workloads per environment, with 50 samples after five warmups and exact
results/scores against individual searches. For 1,600 examples and ten queries, medians improved from
1.956 to 1.281 ms on Python 3.14 and from 1.852 to 1.360 ms on Python 3.11. For 1,024 examples and
16 queries, they improved from 1.776 to 1.265 ms and from 2.031 to 1.312 ms. These are local CPU
measurements with 768-dimensional vectors; workloads retaining the same kernel showed timing
variation in both directions, and the heuristic is not a hardware-independent optimum. Boundary tests
also exercise the changed selection with 48- and 768-dimensional vectors.

The combined changes were revalidated with real GPU models over loopback HTTP/WebSocket: four
initial learning frames followed by 146 streamed frames, with captions, recognition, events and
recording active. All 146 streamed responses contained the learned label; both downloaded clips
decoded all 146 recorded frames (18,958,240 bytes total), and a fresh application/SQLite connection
recognized the label after reopening. Four examples were saved and the event query returned 79 rows.
Stream request median/p95 were 42.95/219.12 ms, with 11.89 effective FPS. This was an independent run,
so its timings must not be treated as a paired speedup over earlier validations. It still used the
experimental Transformers 4.44.2 runtime with PyTorch 2.7.1+cu128, a small learned gallery and shared
model weights on reopening. Label presence is not identity accuracy or long-duration stability.

Gallery construction now joins the stored embedding bytes into one mutable buffer and views it as
the final float32 matrix, avoiding a temporary NumPy array for every example. Names, insertion-order
indices and matrix values matched exactly. Rebuilding a 16,000-example, 768-dimensional gallery took
122.71 to 116.23 ms on Python 3.14 and 134.99 to 127.32 ms on Python 3.11 (30 samples after five
warmups). Peak traced allocations fell by about 3.09 MB in both environments, from roughly 106 to
103 MB. Database population and existing benchmark data were excluded; SQLite and OS pages were
warm. An alternative that progressively freed input rows reduced the traced peak to 55.72 MB but
increased reconstruction time from 121.83 to 238.77 ms in its paired probe, so it was not adopted.
The retained approach improves reconstruction modestly without that latency tradeoff; it does not
halve the persistent gallery size or eliminate the temporary copy of stored bytes.

Profiling 20 rebuilds of the 16,000-example gallery attributed 1.758 of 2.496 seconds inside `_gallery`
to SQLite `fetchall`, versus 0.262 seconds to buffer joining. A query-plan probe compared the current
indexed join with an examples-first scan: when all 16,000 examples matched, query/sort medians were
99.90 and 73.94 ms, but for only 16 matching examples they were 0.097 and 45.43 ms. Forcing a scan
globally would therefore be a substantial regression. SQLite documents that `CROSS JOIN` fixes join
order and recommends planner statistics before manual overrides ([query optimizer documentation](https://www.sqlite.org/optoverview.html#manual_control_of_query_plans_using_cross_join)).
An isolated `ANALYZE` probe changed the label-table access plan but retained indexed example lookups;
its independent timings are not a paired speedup measurement. No application query or statistics
maintenance policy was changed from this investigation. The benchmark database's categories were
restored afterward; its planner statistics now reflect the final synthetic population.

An adaptive query experiment included the cost of counting matching/total examples and selected a
scan only for at least 1,024 matches covering 90% of the database. It improved interleaved dense data
(16,000 matches: 94.88 to 73.96 ms), but at 90% coverage with examples grouped by label it regressed
from 58.14 to 63.63 ms. Category proportion alone was therefore insufficient. A second candidate
sorted only IDs/names in a subquery before reading BLOBs by primary key. Its plan used a covering
index for metadata, a temporary metadata sort and ordered primary-key lookups. Across 20 samples
after five warmups, all rows matched exactly: 16,000 interleaved examples improved from 89.38 to
71.78 ms, while grouped examples changed from 67.28 to 68.79 ms. Both experiments covered seven
category sizes in two insertion layouts, with warm database/OS pages. Neither a density-only rule nor
unconditional metadata ordering was adopted because the benefit depends on layout.

Gallery loading now reads an initial sample of 65 rows from the existing indexed query. If that sample
is full and its IDs are out of order, it reruns the query with metadata ordering before reading BLOBs;
otherwise it completes the original cursor. This avoids changing the query for galleries of at most
64 examples and for sampled grouped data. Complete-rebuild probes covered ten workloads per
environment, with 30 samples after five warmups and exact names, indices and matrix values. For
16,000 interleaved examples, Python 3.14 improved from 108.75 to 94.35 ms and Python 3.11 from
115.02 to 100.81 ms. Grouped 16,000-example galleries remained essentially unchanged (78.84 to
78.53 ms and 87.05 to 86.73 ms). The 128-example interleaved case regressed by 0.055 and 0.084 ms
respectively. Sampling is a bounded heuristic, not a guarantee about the remaining rows or a universal
speedup. Scalar-reference tests cover interleaved rebuilds at recognition thresholds and margins.

To reproduce the cached-gallery comparison using the installed application and an isolated temporary
SQLite database, run `python -m scripts.benchmark_memory --labels 1000 --queries 100 --json` from the
repository root. The JSON includes raw alternating samples, exact result/score agreement, effective
parameters and Python/NumPy/BLAS metadata. Half the synthetic probes are known examples; the others
are random. Database creation and initial gallery loading are excluded from the measurements. Use
`--help` for workload controls; the temporary database is removed after the run. No model downloads
or production-memory changes are required.

Candidate ranking now allocates scores only for refined labels, plus two sentinel entries that
preserve the original negative-score and tie behavior. A paired synthetic search benchmark with
20 samples after five warmups measured 100 queries against 16,000 examples at 23.38 to 17.44 ms,
and ten queries at 4.33 to 3.54 ms, with exact results and scores. The one-label workload with 100
queries was slightly slower (0.954 to 1.011 ms). This comparison isolates the ranking change within
the existing batched search and should not be combined with independently measured speedups.
The single-label regression was subsequently addressed with a direct maximum/refinement path that
skips label aggregation and sorting. A paired comparison against the preceding implementation measured
100 queries over 16 examples at 0.927 to 0.837 ms, with exact scores and decisions. Tests cover positive,
zero and negative similarities, exact and adjacent thresholds, and the absence of a competing-label margin.

An isolated experiment on Transformers 4.44.2 reused BLIP image key/value projections during each
generation call. With 10 alternating measured pairs, a 10-image batch went from 741 to 560 ms with identical
tokens on the test images, retaining 425 MB of projection tensors. This experimental patch is not used by
the application. [Upstream BLIP 4.57.6](https://github.com/huggingface/transformers/blob/v4.57.6/src/transformers/models/blip/modeling_blip_text.py)
already implements cross-attention caching; its runtime benefit and memory cost remain to be validated here.

For SigLIP encoders, the unused text tower is released before transferring weights to the target device.
Image extraction still calls `get_image_features` and retains the same memory fingerprint. A probe on three
real images produced exactly equal 768-dimensional embeddings while active PyTorch GPU allocations fell
from 846.4 to 405.1 MB (441.3 MB released). This measures allocated tensors, not total driver memory;
the full checkpoint still loads on CPU initially. Other encoder model types keep their original loading path.
The image-only dependency is confirmed in the upstream implementations for
[4.48.0](https://github.com/huggingface/transformers/blob/v4.48.0/src/transformers/models/siglip/modeling_siglip.py)
and [4.57.6](https://github.com/huggingface/transformers/blob/v4.57.6/src/transformers/models/siglip/modeling_siglip.py);
the runtime probe still uses the exploratory 4.44.2 environment described above.
The full GPU functional rerun passed (26/26 labelled frames, clip download and reopened memory), ending
with 2,057.6 MB allocated. It measured 38.6 ms median and 290.6 ms p95; given the earlier run-to-run
variation, this is functional evidence and a memory reduction, not an established latency speedup.
An extended 150-frame run (four learning frames and 146 WebSocket frames) also passed: both downloaded
clips decoded to all 146 recorded frames and a reopened memory recognized the saved label. Active GPU
allocations sampled after each response stayed at 2,057.6 MB. Median latency was 30.3 ms, p95 208.3 ms,
with 13.97 FPS over 10.45 seconds of measured requests. This short stability sample uses stored video
and in-process ASGI on Transformers 4.44.2; it does not establish long-duration stability, external network
throughput, identity accuracy or a controlled speedup over earlier runs.

The cached YOLO-World text encoder is now excluded temporarily during image prediction, preventing
Ultralytics from copying unused CLIP weights into its image backend. It is restored in a `finally` block
so subsequent vocabulary changes can reuse it, including after a failed prediction. A detector-only
comparison on Ultralytics 8.4.144/PyTorch 2.7.1+cu128 measured 693.0 versus 51.8 MB of allocated GPU memory
after collection, with identical detections for person → person/bus → person and a repeated frame.
The full 150-frame validation passed again: 146 labelled WebSocket frames, three downloaded clips
decoding to 146 frames, and recognition after reopening memory. Final GPU allocation was 1,448.8 MB
(previously 2,057.6 MB). Median latency was 90.5 ms and p95 540.9 ms; no controlled latency improvement
is established. The full-flow dependency and transport limitations above still apply.

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

Set `ODIN_DEVICE` to `cuda:0` instead if you have a PyTorch build with matching CUDA. The detector loads at
startup. SigLIP loads on the first embedding request; BLIP loads on the first requested description. Models
remain cached after loading. First use may download weights and take longer; missing dependencies, weights
or memory cause that operation to fail. A failed load can be retried without restarting the server.
The encoder uses the image processor directly, without loading the unused SigLIP text tokenizer.
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

For an end to end run against a live camera, follow the [live webcam walkthrough](#live-webcam-walkthrough).

## Live webcam walkthrough

This is the shortest path from a cold machine to teaching a label to a real object in front of your webcam,
with the `models` backend. It assumes the `models` extra is installed and the weights are reachable.

**1. Start the server.** Bind it to loopback; the browser only grants camera access on a secure context, which
means `http://127.0.0.1` (or `localhost`) or a real HTTPS origin. A LAN IP over plain HTTP will not work.

```powershell
$env:ODIN_DATA_DIR = "data/live-demo"     # fresh memory for the run
$env:ODIN_DEVICE   = "cuda:0"             # or "cpu"
$env:ODIN_DETECTOR = "data/models/yolov8s-worldv2.pt"   # local weights, optional
python -m odin_vision --host 127.0.0.1 --port 8000 --backend models
```

Check it answers before opening the browser:

```powershell
curl.exe http://127.0.0.1:8000/api/v1/status
```

In Windows PowerShell `curl` is an alias of `Invoke-WebRequest`, so call `curl.exe` explicitly.

**2. Choose the vocabulary.** YOLO-World only finds the categories you ask for, and the vocabulary is English.
In **Categorías** write something like `person,cup,bottle,cell phone,book,laptop`, then press
**Aplicar filtros**. A Spanish word such as `taza` detects nothing.

**3. Connect the camera.** Open <http://127.0.0.1:8000>, leave **Transporte** on `WebSocket JPEG` and press
**Conectar cámara**. The browser downscales to 960 px wide, encodes JPEG at quality 0.75 and sends up to ten
frames per second; it never sends a frame while the previous one is still being processed.

**4. Read what the server sees.** Each box shows `category · ID`, and the object list under the video adds the
confidence. Hover an entry to read the generated description. The status line shows the frame number and the
server side processing time.

**5. Teach a label.** Click the box of the object, type a name in **Etiqueta** and press **Guardar objeto**.
The server embeds the crops kept for that track, writes them to SQLite and returns the number of stored
examples. The box label changes on the next frame. The Spanish command `guarda como <name>` in the
**Comandos** box does the same for the selected object.

**6. Verify the memory is real.** Press **Detener**, connect again, and point the camera at the same object.
It comes back with your label because the vectors live on disk, not in the session.

### What to expect

On an RTX 5070 Ti Laptop with `cuda:0`, 960×540 JPEG and descriptions enabled, measured on 2026-09-19 by
streaming a local video file over the WebSocket path, not a physical camera: the first frame of a cold server
takes 8 to 10 seconds because YOLO-World, SigLIP and BLIP load lazily, and every frame after that costs 11 to
20 ms of server time. On CPU expect roughly one frame per second, and turn descriptions off to keep it usable.

### Labelling from the API while the stream runs

The browser owns the WebSocket, but the REST API stays open, so an operator or a script can inspect and label
the same session from outside. The session ID appears in the server log as part of the stream URL, and on
stderr when you use the headless client.

```powershell
$sid = "<session id>"
# current objects, their IDs, labels and descriptions
curl.exe "http://127.0.0.1:8000/api/v1/sessions/$sid/result"
# name the object the tracker calls 3
curl.exe -X POST "http://127.0.0.1:8000/api/v1/sessions/$sid/tracks/3/labels" `
  -H "Content-Type: application/json" -d '{"label": "Falcon"}'
```

The server serialises inference and memory access behind one lock, so a call that arrives while a frame is
being processed returns `429`; retry it. Labelling a track that has left the frame returns `404`.

### When it does not work

| Symptom | Cause and fix |
|---|---|
| No boxes at all | The category is not in the vocabulary, or it is in Spanish. Use English nouns. |
| The label does not come back | The similarity fell below the threshold. Lower **Similitud mínima** to about 0.85 and teach the object again from several angles. |
| `El objeto ya no está visible` | The track was lost before the label request arrived. Reselect the box and retry. |
| `Servidor ocupado` | A frame was in flight. Retry the call. |
| `Instale odin-vision[webrtc]` | The WebRTC transport needs aiortc. Use `WebSocket JPEG` instead. |
| The camera never starts | The page is not on a secure context. Use `127.0.0.1`, not a LAN address over HTTP. |

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
| `PUT /sessions/{sid}/config` | Replace the session config; resets tracking only when resolved values change |
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
- GPU recognition/descriptions have only the limited exploratory validation above; supported-stack acceptance
  and external network throughput remain pending.
- Tune the threshold and the margin with positive and negative examples from your own camera before trusting
  any result.
- The bundled page needs a secure context to open a camera: `127.0.0.1`, `localhost` or HTTPS. Serving it on a
  LAN address over plain HTTP loads the page but never gets camera permission.
- The detector vocabulary is English. Categories written in another language silently detect nothing.

## License

[PolyForm Noncommercial License 1.0.0](LICENSE.md). Free to use, modify and share for any noncommercial
purpose. Commercial use requires a separate license from the author.
