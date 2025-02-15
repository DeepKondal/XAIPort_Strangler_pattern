import os
import shutil
import zipfile
import uvicorn
import av
import numpy as np
import pandas as pd
from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from DataProcess import DataProcess
from dotenv import load_dotenv
import traceback

# Load environment variables from .env file
load_dotenv()

app = FastAPI()

data_processor = DataProcess(base_storage_address="datasets")

# ✅ Load Kinetics-400 metadata into DataProcess
data_processor.load_metadata(
    labels_csv="dataprocess/kinetics_400_labels.csv",
    video_list_txt="dataprocess/kinetics400_val_list_videos.txt"
)

@app.get("/")
async def root():
    return {"message": "Server is running"}

# ✅ Upload Dataset
@app.post("/upload-dataset/{dataset_id}")
async def upload_dataset(dataset_id: str, zip_file: UploadFile = File(...)):
    temp_dir = "temp"
    os.makedirs(temp_dir, exist_ok=True)
    temp_file = f"{temp_dir}/{zip_file.filename}"

    with open(temp_file, "wb") as buffer:
        shutil.copyfileobj(zip_file.file, buffer)

    dataset_dir = f"datasets/{dataset_id}"
    os.makedirs(dataset_dir, exist_ok=True)

    with zipfile.ZipFile(temp_file, 'r') as zip_ref:
        zip_ref.extractall(dataset_dir)

    os.remove(temp_file)
    return {"message": "Dataset uploaded and extracted successfully"}

# ✅ Process Kinetics Videos
@app.post("/process-kinetics-dataset")
async def process_kinetics_dataset(data: dict):
    try:
        video_dir = data.get("video_dir", "datasets/videos/")
        num_frames = data.get("num_frames", 8)

        if not os.path.exists(video_dir):
            raise HTTPException(status_code=400, detail=f"Invalid or missing 'video_dir': {video_dir}")

        results = []
        for video_file in os.listdir(video_dir):
            if video_file.endswith(".mp4"):
                video_path = os.path.join(video_dir, video_file)
                print(f"Processing video: {video_path}")

                try:
                    result = data_processor.process_kinetics_video(video_path, num_frames=num_frames)
                    results.append(result)
                except Exception as e:
                    error_message = traceback.format_exc()
                    print(f"❌ ERROR processing {video_path}: {error_message}")
                    raise HTTPException(status_code=500, detail=error_message)

        return {"message": "Kinetics-400 dataset processed successfully", "results": results}

    except Exception as e:
        error_message = traceback.format_exc()
        print(f"❌ ERROR in process_kinetics_dataset: {error_message}")
        raise HTTPException(status_code=500, detail=error_message)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)
