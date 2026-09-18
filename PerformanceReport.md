## Performance

The system was evaluated using the MediaPipe Pose Landmarker Lite model with a
fixed monocular webcam.

### Test Configuration

| Parameter | Configuration |
|---|---|
| Pose model | MediaPipe Pose Landmarker Lite |
| Camera | Monocular webcam |
| Camera resolution | 640 × 480 |
| Processing architecture | Python backend + Electron frontend |
| Camera capture | Dedicated capture thread with latest-frame buffering |
| Operating system | Windows |
| End-to-end measurement | Electron frontend |
| Target | ≥ 30 FPS |

### Measured Performance

The final application achieved approximately **30–31 FPS end-to-end**.

This measurement represents the complete pipeline from camera capture through
pose estimation, biomechanical calculations, frame rendering/encoding,
WebSocket transmission, and display in the Electron UI.

| Metric | Measured Result |
|---|---:|
| End-to-end FPS | **30–31 FPS** |
| Backend processing throughput | ~38 FPS |
| Target requirement | ≥ 30 FPS |

The backend processes the latest available camera frame rather than building
up a queue of stale frames. A dedicated camera capture thread continuously
updates a latest-frame buffer, allowing pose inference and camera capture to
operate independently. Older frames are discarded when processing falls
behind.

This design prioritizes **low latency and responsiveness** over processing
every camera frame.

### Pose Inference Benchmark

A standalone benchmark of the Pose Landmarker Lite model was also performed:

| Metric | Result |
|---|---:|
| Average inference time | 29.05 ms |
| Median inference time | 26.80 ms |
| P95 inference time | 41.82 ms |
| Minimum | 21.33 ms |
| Maximum | 61.47 ms |
| Theoretical inference throughput | 34.43 FPS |

The standalone inference benchmark is separate from the end-to-end FPS
measurement. The end-to-end result is lower because it also includes camera
handling, image conversion, biomechanical calculations, skeleton rendering,
JPEG encoding, WebSocket transmission, and Electron rendering.

### Performance Design Decisions

- **Pose Landmarker Lite** was selected to provide sufficient pose-estimation
  quality while maintaining real-time performance.
- Camera capture runs on a **dedicated thread**.
- Only the **latest camera frame** is retained.
- Stale frames are intentionally dropped instead of allowing a processing
  queue to grow.
- The application therefore remains responsive even when individual inference
  frames take longer than usual.