# Docker Setup & Usage Guide

This guide provides step-by-step instructions for running the **GCPBBB Renewable Energy Optimization** project using Docker. This approach ensures all dependencies (Python, Node.js, libraries) are isolated and consistent.

## 1. Install Docker

### macOS & Windows
1. Download **Docker Desktop** from the [official website](https://www.docker.com/products/docker-desktop).
2. Install the application and start it.
3. Ensure the Docker engine is running (you should see the Docker whale icon in your taskbar/menu bar).

### Linux (Ubuntu/Debian)
Run the following commands to install Docker Engine:
```bash
# Update package index
sudo apt-get update

# Install Docker
sudo apt-get install -y docker.io docker-compose

# Start Docker service
sudo systemctl start docker
sudo systemctl enable docker

# (Optional) Add your user to the docker group to run without 'sudo'
sudo usermod -aG docker $USER
# NOTE: Log out and back in for this to take effect.
```

## 2. Running the Application

### Prequisites
Ensure you are in the project root directory where the `docker-compose.yml` file is located:
```bash
cd SOLARPRED_PROJ_ML
```

### Start the Services
Run the following command to build the images and start the containers in the background:

```bash
docker compose up --build -d
```

*   `--build`: Rebuilds images if you've made changes to the `Dockerfile` or source code.
*   `-d`: Detached mode (runs in the background).

**Note:** The first run will take a few minutes as it downloads base images, installs dependencies, and trains the initial machine learning models.

## 3. Services & Ports

Once running, the application exposes the following services:

| Service | Container Name | Host Port | Internal Port | Description |
| :--- | :--- | :--- | :--- | :--- |
| **Frontend** | `helios_frontend` | **3000** | 80 | The React web interface. Access at `http://localhost:3000`. |
| **Backend** | `helios_backend` | **8001** | 8000 | The FastAPI server. Access API docs at `http://localhost:8001/docs`. |
| **Trainer** | `helios_trainer` | N/A | N/A | Runs once to train models and exits. Does not expose ports. |

## 4. Monitoring Logs

To see what's happening inside the containers (especially useful if something isn't working), use the logs command.

### View All Logs (Streaming)
```bash
docker compose logs -f
```
*   Press `Ctrl + C` to exit the log view.

### View Specific Service Logs
To view logs only for the backend or frontend:
```bash
docker compose logs -f backend
```
or
```bash
docker compose logs -f frontend
```

## 5. Stopping the Application

To stop the running containers:
```bash
docker compose stop
```

To stop and **remove** containers and networks (clean slate):
```bash
docker compose down
```

## 6. Common Troubleshooting

### "No configuration file provided: not found"
**Cause:** You are running the command from the wrong directory.
**Fix:** Ensure you are in the folder containing `docker-compose.yml`:
```bash
cd /path/to/SOLARPRED_PROJ_ML
```

### "Bind for 0.0.0.0:8001 failed: port is already allocated"
**Cause:** Another application is using port 8001 (or 3000).
**Fix:** Stop the other application or modify the `docker-compose.yml` to map to a different host port (e.g., `"8002:8000"`).

### Changes not showing up?
If you modified the code but don't see changes, force a rebuild:
```bash
docker compose up --build -d
```
