from fastapi import FastAPI, UploadFile, File, BackgroundTasks
from fastapi.responses import JSONResponse, PlainTextResponse
from fastapi.middleware.cors import CORSMiddleware
from src.pipeline.prediction_pipeline import PredictPipeline
import pandas as pd

from uvicorn import run as app_run
from fastapi.responses import Response
from starlette.responses import RedirectResponse

import shutil
import os
import logging

# Initialize FastAPI app
app = FastAPI()

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize the prediction pipeline
predict_pipeline = PredictPipeline()

# Set up logging
logging.basicConfig(level=logging.INFO)

# Define a root route ("/")
@app.get("/", tags=["authentication"])
async def index():
    return RedirectResponse(url="/docs")

# Define a POST route to handle image uploads and predictions
@app.post("/predict")
async def predict_image(file: UploadFile = File(...)):
    try:
        # Save the uploaded image to a temporary location
        temp_image_path = f"temp_images/{file.filename}"
        os.makedirs("temp_images", exist_ok=True)
        with open(temp_image_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Perform prediction using the PredictPipeline class
        logging.info(f"Received image for prediction: {file.filename}")
        predicted_class = predict_pipeline.predict(temp_image_path)

        # Remove the temporary image after prediction
        os.remove(temp_image_path)
        logging.info(f"Prediction completed for {file.filename}, predicted class: {predicted_class[0]}")

        # Return the predicted class in a JSON response
        return JSONResponse(content={"predicted_class": int(predicted_class[0])})

    except Exception as e:
        logging.error(f"Error during prediction: {str(e)}")
        return JSONResponse(content={"error": str(e)}, status_code=500)

# Define the background training function
def run_training():
    try:
        logging.info("Training started in the background...")
        os.system("python main.py")
        logging.info("Training completed.")
    except Exception as e:
        logging.error(f"Error during training: {str(e)}")

# Define a GET route for training the model in the background
@app.get("/train")
async def training(background_tasks: BackgroundTasks):
    try:
        # Run the training in the background
        logging.info("Training process triggered.")
        background_tasks.add_task(run_training)
        return Response("Training has started in the background!")

    except Exception as e:
        logging.error(f"Error occurred during training: {str(e)}")
        return PlainTextResponse(f"Error Occurred! {e}")

# Run the FastAPI app
if __name__ == "__main__":
   app_run(app, host="0.0.0.0", port=8000)
