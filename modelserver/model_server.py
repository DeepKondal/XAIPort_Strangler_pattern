from fastapi import FastAPI, BackgroundTasks, HTTPException, File
from pydantic import BaseModel
import Model_ResNet
import os
import torch
from attention_extractor import AttentionExtractor  

app = FastAPI()

# Initialize the attention extractor
attention_extractor = AttentionExtractor(
    model_name="facebook/timesformer-base-finetuned-k400",
    device="cuda" if torch.cuda.is_available() else "cpu"
)

# Output directory for video results
OUTPUT_DIR = "video_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

class DatasetPaths(BaseModel):
    dataset_id: str  # Field to include dataset_id in the request body

def run_model_async(dataset_id, model_func):
    # Construct the dataset path based on dataset_id
    dataset_paths = f"dataprocess/datasets/{dataset_id}"
    model_func(dataset_paths)

@app.post("/resnet/{dataset_id}/{perturbation_func_name}/{severity}")
async def run_model_background(dataset_id: str, perturbation_func_name: str, severity: int, background_tasks: BackgroundTasks):
    # Define local paths for original and perturbed datasets
    local_original_dataset_path = f"dataprocess/datasets/{dataset_id}"
    local_perturbed_dataset_path = f"dataprocess/datasets/{dataset_id}_{perturbation_func_name}_{severity}"

    # Asynchronously run the model on the dataset paths
    background_tasks.add_task(Model_ResNet.model_run, [local_original_dataset_path, local_perturbed_dataset_path])

    return {
        "message": f"ResNet run for dataset {dataset_id} with perturbation {perturbation_func_name} and severity {severity} has started. Results will be available in Blob storage after completion."
    }

@app.post("/facebook/timesformer-base-finetuned-k400/{dataset_id}")
async def video_explain(dataset_id: str):
    video_dir = "dataprocess/videos"
    video_files = [f for f in os.listdir(video_dir) if f.endswith(".mp4")]

    results = []
    for video_file in video_files:
        video_path = os.path.join(video_dir, video_file)
        try:
            # Extract attention and logits from the video
            spatial_attention, temporal_attention, frames, logits = extractor.extract_attention(video_path)
            prediction_idx = torch.argmax(logits, dim=1).item()
            prediction = extractor.model.config.id2label[prediction_idx]

            # Create a directory for video results
            video_result_dir = os.path.join(OUTPUT_DIR, os.path.splitext(video_file)[0])
            os.makedirs(video_result_dir, exist_ok=True)

            # Save visualizations
            extractor.visualize_attention(spatial_attention, temporal_attention, frames, video_result_dir, prediction, "Unknown")
            
            results.append({
                "video_file": video_file,
                "prediction": prediction,
                "results_dir": video_result_dir 
            })
        except Exception as e:
            results.append({"video_file": video_file, "error": str(e)})

    return {"results": results}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8002)
