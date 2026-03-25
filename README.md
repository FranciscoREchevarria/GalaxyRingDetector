---
title: Galaxy Ring Detector
emoji: 🌌
colorFrom: indigo
colorTo: blue
sdk: docker
app_port: 7860
---

# Galaxy Ring Detector

Detect inner and outer rings in galaxy images using a fine-tuned Zoobot model with a custom multilabel head.

## License

This project is licensed under the [GPL-3.0](LICENSE). The model is derived from [Zoobot 2.0](https://github.com/mwalmsley/zoobot), which is also licensed under GPL-3.0.

## Local running
uvicorn server:app --host 127.0.0.1 --port 8000 --reload

## Configuration
- **FAST_PREDICTION** (env): Set to `true` to disable test-time augmentation (TTA). Uses a single forward pass instead of 8, trading ~8x speed for slightly lower accuracy.
- **fast** (request param): Same as above, per-request. Use `?fast=true` on GET endpoints or include `fast=true` in form/JSON body for POST endpoints.
