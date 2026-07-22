# Ducks vs Chickens YOLOv8 Detector

A Flask web application that deploys a custom YOLOv8 object-detection model trained to identify ducks and chickens in uploaded images.

## Features

- Upload PNG, JPG, JPEG, or WEBP images
- Detect ducks and chickens
- Display bounding boxes and class labels
- Show confidence scores
- Run locally using Docker
- Deploy online using Render

## Project Structure

```text
.
├── app.py
├── best.pt
├── Dockerfile
├── requirements.txt
├── render.yaml
├── templates/
│   └── index.html
└── static/
    ├── style.css
    ├── uploads/
    └── results/
```

## Run Locally with Docker

Build the Docker image:

```bash
docker build -t yolo-detector .
```

Run the container:

```bash
docker run -p 5000:5000 yolo-detector
```

Open:

```text
(https://ch9-duck-chicken-1.onrender.com)
```

## How to Use the Interface

1. Open the application.
2. Select an image containing a duck or chicken.
3. Click **Run Detection**.
4. Review the annotated image and confidence table.

## Deploy on Render

1. Upload the project files to a GitHub repository.
2. Log in to Render.
3. Select **New +** and then **Blueprint** or **Web Service**.
4. Connect the GitHub repository.
5. Select Docker as the runtime if it is not detected automatically.
6. Deploy the service.
7. Copy the public URL after deployment succeeds.

## Known Limitations

- Accuracy depends on the size and quality of the training dataset.
- The model may struggle with dark, blurry, or partially hidden animals.
- The app accepts images only; video inference is not included.
- Free hosting may have slow first-time startup.
- Uploaded and generated images are temporary on many cloud platforms.

## Application Preview

Add a screenshot or GIF here after testing the deployed application.
