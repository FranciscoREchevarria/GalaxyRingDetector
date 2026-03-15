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
