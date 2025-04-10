# GitHub Repo with CI/CD Pipeline for FastAPI Model Deployment

This tutorial extends the `fastapi_docker` one by:
1. **Creating a GitHub repository** with all the files
2. **Setting up GitHub Actions** to automatically:
   - Test the code
   - Build a Docker image
   - Push it to **Docker Hub**

---

## **Step 1: Create a GitHub Repository**
1. Go to [GitHub](https://github.com) and create a new repository (e.g., `california-housing-api`).
2. Clone it locally:
   ```bash
   git clone https://github.com/your-username/california-housing-api.git
   cd california-housing-api
   ```

---

## **Step 2: Organize the Project Structure**
Your repo should contain:
```
california-housing-api/
├── .github/
│   └── workflows/
│       └── docker-publish.yml  # GitHub Actions workflow
├── main.py                     # FastAPI app
├── requirements.txt            # Python dependencies
├── Dockerfile                  # Docker configuration
├── train_model.py              # Model training script
└── README.md                   # Project documentation
```

As we don't want to upload a pickle with the model to github (very bad practice), we will train the model during build (which is also not the best practice, we should download it from a model repository). Modify the Dockerfile as follows:

```dockerfile
# Use official Python image
FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Copy requirements first to leverage Docker cache
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Train the model during build
RUN python train_model.py

# Expose the port the app runs on
EXPOSE 8000

# Command to run the application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

---

## **Step 3: Set Up GitHub Actions for Docker Hub**
Create `.github/workflows/docker-publish.yml`:
```yaml
name: Build and Push Docker Image

on:
  push:
    branches: [ "main" ]
  pull_request:
    branches: [ "main" ]

env:
  REGISTRY: docker.io
  IMAGE_NAME: ${{ secrets.DOCKER_HUB_USERNAME }}/california-housing-api

jobs:
  build-and-push:
    runs-on: ubuntu-latest
    permissions:
      contents: read
      packages: write

    steps:
      - name: Checkout repository
        uses: actions/checkout@v4

      - name: Log in to Docker Hub
        uses: docker/login-action@v3
        with:
          username: ${{ secrets.DOCKER_HUB_USERNAME }}
          password: ${{ secrets.DOCKER_HUB_TOKEN }}

      - name: Build and push Docker image
        uses: docker/build-push-action@v5
        with:
          context: .
          push: true
          tags: |
            ${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}:latest
            ${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}:${{ github.sha }}
```

---

## **Step 4: Configure Docker Hub Secrets in GitHub**
1. **Generate a Docker Hub Access Token**:
   - Go to [Docker Hub Account Settings](https://hub.docker.com/settings/security).
   - Under **Personal access tokens > Generate new token**, create a token with **Read/Write** permissions.
   - Copy the token.

2. **Add Secrets to GitHub**:
   - Go to your GitHub repo → **Settings → Secrets and variables → Actions**.
   - Add two **New repository secret**:
     - Name: `DOCKER_HUB_USERNAME`, secret: your Docker Hub username.
     - Name: `DOCKER_HUB_TOKEN`, secret: the token you just created.

---

## **Step 5: Commit and Push to GitHub**
```bash
git add .
git commit -m "Initial commit with FastAPI, Docker, and GitHub Actions"
git push origin main
```

---

## **Step 6: Verify GitHub Actions Workflow**
1. Go to your GitHub repo → **Actions** tab.
2. You should see the workflow running.
3. Once completed, check **Docker Hub** to confirm the image was pushed.

---

## **Step 7: Pull and Run the Docker Image**
```bash
docker pull mgraciarbunies/california-housing-api
docker run -p 8000:8000 mgraciarbunies/california-housing-api
```
- Test the API at `http://localhost:8000/docs`.

---

## **Step 8: Change something in your API**
- Now, every time you push changes to `main`, GitHub Actions will rebuild and update the Docker image automatically! 🚀
- Try to modify your API by creating a new mock endpoint (for example a 'GET /hello' endpoint and push it to your repo, pull the image again and see how it has changed).
