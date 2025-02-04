# from fastapi import FastAPI, UploadFile, File, BackgroundTasks
# from fastapi.responses import JSONResponse, PlainTextResponse
# from src.pipeline.prediction_pipeline import PredictPipeline
# import shutil
# import os
# import uvicorn

# # Initialize FastAPI app
# app = FastAPI()

# # Initialize the prediction pipeline
# predict_pipeline = PredictPipeline()

# # Define a root route ("/")
# @app.get("/")
# async def root():
#     return PlainTextResponse("FastAPI is running! Go to /predict or /train")

# # Define a POST route to handle image uploads and predictions
# @app.post("/predict/")
# async def predict_image(file: UploadFile = File(...)):
#     try:
#         # Save the uploaded image to a temporary location
#         temp_image_path = f"temp_images/{file.filename}"
#         os.makedirs("temp_images", exist_ok=True)
#         with open(temp_image_path, "wb") as buffer:
#             shutil.copyfileobj(file.file, buffer)

#         # Perform prediction using the PredictPipeline class
#         predicted_class = predict_pipeline.predict(temp_image_path)

#         # Remove the temporary image after prediction
#         os.remove(temp_image_path)

#         # Return the predicted class in a JSON response
#         return JSONResponse(content={"predicted_class": int(predicted_class[0])})

#     except Exception as e:
#         return JSONResponse(content={"error": str(e)}, status_code=500)

# # Define the background training function
# def run_training():
#     os.system("python main.py")

# # Define a GET route for training the model in the background
# @app.get("/train")
# async def training(background_tasks: BackgroundTasks):
#     try:
#         # Run the training in the background
#         background_tasks.add_task(run_training)
#         return PlainTextResponse("Training has started in the background!")

#     except Exception as e:
#         return PlainTextResponse(f"Error Occurred! {e}")

# # Run the FastAPI app
# if __name__ == "__main__":
#     uvicorn.run(app, host="0.0.0.0", port=8080)
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
import os
import shutil
from src.pipeline.prediction_pipeline import PredictPipeline
import subprocess

# Initialize Flask app
app = Flask(__name__)

# Initialize the prediction pipeline
predict_pipeline = PredictPipeline()

# Define a route for the root ("/")
@app.route("/", methods=["GET"])
def root():
    return "Flask is running! Go to /predict or /train"

# Define a POST route to handle image uploads and predictions
@app.route("/predict/", methods=["POST"])
def predict_image():
    try:
        # Check if the file is present in the request
        if 'file' not in request.files:
            return jsonify({"error": "No file part in the request"}), 400

        file = request.files['file']

        # Check if the file has a filename
        if file.filename == '':
            return jsonify({"error": "No selected file"}), 400

        # Save the uploaded image to a temporary location
        temp_image_path = os.path.join("temp_images", secure_filename(file.filename))
        os.makedirs("temp_images", exist_ok=True)
        file.save(temp_image_path)

        # Perform prediction using the PredictPipeline class
        predicted_class = predict_pipeline.predict(temp_image_path)

        # Remove the temporary image after prediction
        os.remove(temp_image_path)

        # Return the predicted class in a JSON response
        return jsonify({"predicted_class": int(predicted_class[0])})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Define the background training function
def run_training():
    subprocess.Popen(["python", "main.py"])

# Define a GET route for training the model in the background
@app.route("/train", methods=["GET"])
def training():
    try:
        # Run the training in the background
        run_training()
        return "Training has started in the background!"

    except Exception as e:
        return f"Error Occurred! {e}", 500

# Run the Flask app
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
