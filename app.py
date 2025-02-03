from fastapi import FastAPI, UploadFile, File, BackgroundTasks
from fastapi.responses import JSONResponse, PlainTextResponse
from src.pipeline.prediction_pipeline import PredictPipeline
import shutil
import os
import uvicorn

# Initialize FastAPI app
app = FastAPI()

# Initialize the prediction pipeline
predict_pipeline = PredictPipeline()

# Define a root route ("/")
@app.get("/")
async def root():
    return PlainTextResponse("FastAPI is running! Go to /predict or /train")

# Define a POST route to handle image uploads and predictions
@app.post("/predict/")
async def predict_image(file: UploadFile = File(...)):
    try:
        # Save the uploaded image to a temporary location
        temp_image_path = f"temp_images/{file.filename}"
        os.makedirs("temp_images", exist_ok=True)
        with open(temp_image_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Perform prediction using the PredictPipeline class
        predicted_class = predict_pipeline.predict(temp_image_path)

        # Remove the temporary image after prediction
        os.remove(temp_image_path)

        # Return the predicted class in a JSON response
        return JSONResponse(content={"predicted_class": int(predicted_class[0])})

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

# Define the background training function
def run_training():
    os.system("python main.py")

# Define a GET route for training the model in the background
@app.get("/train")
async def training(background_tasks: BackgroundTasks):
    try:
        # Run the training in the background
        background_tasks.add_task(run_training)
        return PlainTextResponse("Training has started in the background!")

    except Exception as e:
        return PlainTextResponse(f"Error Occurred! {e}")

# Run the FastAPI app
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)
