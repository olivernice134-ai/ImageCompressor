from flask import Flask, render_template, request
import os
import cv2
import numpy as np
from sklearn.cluster import KMeans
from werkzeug.utils import secure_filename

app = Flask(__name__)
UPLOAD_FOLDER = "static/uploads"
OUTPUT_FOLDER = "static/outputs"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        file = request.files['image']
        k = int(request.form.get('k', 16))

        if file:
            filename = secure_filename(file.filename)
            input_path = os.path.join(UPLOAD_FOLDER, filename)
            file.save(input_path)

            
            image = cv2.imread(input_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = cv2.resize(image, (256, 256))
            pixel_values = image.reshape((-1, 3)).astype(np.float32)

            
            kmeans = KMeans(n_clusters=k, random_state=0, n_init=10)
            labels = kmeans.fit_predict(pixel_values)
            centers = np.uint8(kmeans.cluster_centers_)
            compressed = centers[labels.flatten()].reshape(image.shape)

            
            output_filename = f"compressed_{filename}"
            output_path = os.path.join(OUTPUT_FOLDER, output_filename)
            cv2.imwrite(output_path, cv2.cvtColor(compressed, cv2.COLOR_RGB2BGR))

            
            original_size = os.path.getsize(input_path) // 1024
            compressed_size = os.path.getsize(output_path) // 1024

            return render_template(
                "index.html",
                original=input_path,
                compressed=output_path,
                k=k,
                original_size=original_size,
                compressed_size=compressed_size
            )

    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
