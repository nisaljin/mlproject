# Deployment Guide (EC2 + Docker + Nginx)

This guide explains how to deploy the GCPBBB Energy Optimization project on an AWS EC2 instance.

## Prerequisites

*   An AWS EC2 instance (Ubuntu 20.04 or 22.04 recommended).
*   SSH access to the instance.
*   Domain name (optional, but recommended for Nginx).

## Step 1: Clone the Repository

SSH into your EC2 instance and clone the repository:

```bash
git clone https://github.com/rahulwork252/SOLARPRED_PROJ_ML.git
cd SOLARPRED_PROJ_ML
```

## Step 2: Install Docker & Docker Compose

If Docker is not installed, run the following commands:

```bash
# Update package index
sudo apt-get update

# Install Docker
sudo apt-get install -y docker.io

# Install Docker Compose
sudo apt-get install -y docker-compose

# Start Docker service
sudo systemctl start docker
sudo systemctl enable docker

# Add current user to docker group (to run without sudo)
sudo usermod -aG docker $USER
```

*Note: You may need to log out and log back in for the group change to take effect.*

## Step 3: Run the Application

Start the application using Docker Compose. This will build the images and start the API (port 8000) and Dashboard (port 8501).

```bash
docker-compose up --build -d
```

Check if containers are running:

```bash
docker ps
```

You should see `gcpbbb_backend` and `gcpbbb_frontend` running.

## Step 4: Configure Nginx (Reverse Proxy)

Install Nginx if it's not already installed:

```bash
sudo apt-get install -y nginx
```

Create a new Nginx configuration file:

```bash
sudo nano /etc/nginx/sites-available/gcpbbb
```

Paste the following configuration (replace `yourdomain.com` with your actual domain or public IP):

```nginx
server {
    listen 80;
    server_name yourdomain.com; # Or your EC2 Public IP

    # Frontend (Streamlit)
    location / {
        proxy_pass http://localhost:8501/;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }

    # Backend API (FastAPI)
    location /api/ {
        proxy_pass http://localhost:8000/;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

Enable the configuration:

```bash
sudo ln -s /etc/nginx/sites-available/gcpbbb /etc/nginx/sites-enabled/
sudo rm /etc/nginx/sites-enabled/default  # Remove default config if needed
sudo nginx -t # Test configuration
sudo systemctl restart nginx
```

## Step 5: Access the Application

*   **Dashboard**: `http://your-ec2-ip-or-domain/`
*   **API Docs**: `http://your-ec2-ip-or-domain/api/docs`

## Troubleshooting

*   **Ports**: Ensure ports 80 (HTTP), 8000, and 8501 are open in your EC2 Security Group (Inbound Rules).
*   **Logs**: Check container logs if something fails:
    ```bash
    docker-compose logs -f
    ```
