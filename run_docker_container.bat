docker stop galaxy-ring-detector
docker rm galaxy-ring-detector
docker run --name galaxy-ring-detector -p 7860:7860 galaxy-ring-detector:local
docker stop galaxy-ring-detector
docker rm galaxy-ring-detector