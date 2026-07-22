
import os
import uuid
from pathlib import Path

from flask import Flask, render_template, request
from ultralytics import YOLO
from werkzeug.utils import secure_filename

BASE_DIR = Path(__file__).resolve().parent
UPLOAD_FOLDER = BASE_DIR / "static" / "uploads"
RESULT_FOLDER = BASE_DIR / "static" / "results"
MODEL_PATH = BASE_DIR / "best.pt"

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "webp"}

UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)
RESULT_FOLDER.mkdir(parents=True, exist_ok=True)

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = str(UPLOAD_FOLDER)
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024

model = YOLO(str(MODEL_PATH))


def allowed_file(filename):
    return (
        "." in filename
        and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS
    )


@app.route("/health")
def health():
    return {"status": "healthy"}, 200


@app.route("/", methods=["GET", "POST"])
def index():
    result_image = None
    detections = []
    error = None

    if request.method == "POST":
        if "image" not in request.files:
            error = "No image was included in the request."
            return render_template(
                "index.html",
                result_image=result_image,
                detections=detections,
                error=error,
            )

        image_file = request.files["image"]

        if image_file.filename == "":
            error = "Please select an image."
        elif not allowed_file(image_file.filename):
            error = "Supported formats are PNG, JPG, JPEG, and WEBP."
        else:
            try:
                safe_name = secure_filename(image_file.filename)
                unique_name = f"{uuid.uuid4().hex}_{safe_name}"
                upload_path = UPLOAD_FOLDER / unique_name
                image_file.save(upload_path)

                prediction = model.predict(
                    source=str(upload_path),
                    conf=0.25,
                    save=False,
                    verbose=False,
                )[0]

                annotated_image = prediction.plot()
                result_name = f"result_{Path(unique_name).stem}.jpg"
                result_path = RESULT_FOLDER / result_name

                import cv2
                cv2.imwrite(str(result_path), annotated_image)

                if prediction.boxes is not None:
                    for box in prediction.boxes:
                        class_id = int(box.cls.item())
                        confidence = float(box.conf.item())
                        detections.append(
                            {
                                "label": model.names[class_id],
                                "confidence": round(confidence * 100, 2),
                            }
                        )

                result_image = f"results/{result_name}"

            except Exception as exc:
                error = f"Prediction failed: {exc}"

    return render_template(
        "index.html",
        result_image=result_image,
        detections=detections,
        error=error,
    )


@app.errorhandler(413)
def file_too_large(error):
    return render_template(
        "index.html",
        result_image=None,
        detections=[],
        error="The uploaded image is larger than 10 MB.",
    ), 413


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
