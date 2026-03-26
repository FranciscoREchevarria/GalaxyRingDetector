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

## Run with Docker locally

### 1) Install Docker Desktop
1. Download Docker Desktop from https://www.docker.com/products/docker-desktop/
2. Install it and enable the default options (WSL2 backend on Windows is recommended).
3. Start Docker Desktop and wait until Docker shows it is running.

### 2) Build the image
From the project root (where `Dockerfile` is located), run:

```bash
docker build -t galaxy-ring-detector:local .
```

or run `build_docker_image.bat` in the root directory

### 3) Run the container
Run the app on port `7860` (container) and expose it on `7860` (host):

```bash
docker run --name galaxy-ring-detector -p 7860:7860 galaxy-ring-detector:local
```

or run `run_docker_container.bat` in the root directory.

Open: http://127.0.0.1:7860

### 4) Run with fast mode enabled (optional)
If you want faster inference (less accurate than full TTA), set `FAST_PREDICTION=true`:

```bash
docker run --name galaxy-ring-detector -p 7860:7860 -e FAST_PREDICTION=true galaxy-ring-detector:local
```

### 5) Useful container commands
- Stop container:

```bash
docker stop galaxy-ring-detector
```

- Start existing container again:

```bash
docker start galaxy-ring-detector
```

- View logs:

```bash
docker logs -f galaxy-ring-detector
```

- Remove container:

```bash
docker rm -f galaxy-ring-detector
```

## Configuration
- **FAST_PREDICTION** (env): Set to `true` to disable test-time augmentation (TTA). Uses a single forward pass instead of 8, trading ~8x speed for slightly lower accuracy.
- **fast** (request param): Same as above, per-request. Use `?fast=true` on GET endpoints or include `fast=true` in form/JSON body for POST endpoints.
