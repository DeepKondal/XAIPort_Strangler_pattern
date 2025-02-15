import os
import shutil
import datetime
import logging
import json
import numpy as np
import pandas as pd
import av  # Video Processing
import aiofiles
import asyncio
import pika  # RabbitMQ
from PIL import Image
from common.neo4j_client import ProvenanceModel  # ✅ Updated import

class DataProcess:
    def __init__(self, base_storage_address="datasets"):
        self.base_storage_address = base_storage_address
        self.metadata = {}  # Initialize metadata as an empty dictionary

        # Initialize Neo4j Client
        self.provenance = ProvenanceModel()

        # RabbitMQ Setup
        self.rabbitmq_host = os.getenv("RABBITMQ_HOST", "localhost")
        self.rabbitmq_queue = os.getenv("RABBITMQ_QUEUE_NAME", "pipeline_tasks")

    # ✅ Send Event Message to RabbitMQ
    def send_rabbitmq_message(self, event, dataset_id=None, error=None):
        """Send RabbitMQ messages for dataset events."""
        try:
            connection = pika.BlockingConnection(pika.ConnectionParameters(host=self.rabbitmq_host))
            channel = connection.channel()
            channel.queue_declare(queue=self.rabbitmq_queue, durable=True)

            message = {"event": event, "timestamp": str(datetime.datetime.now())}
            if dataset_id:
                message["dataset_id"] = dataset_id
            if error:
                message["error"] = str(error)

            channel.basic_publish(exchange='', routing_key=self.rabbitmq_queue, body=json.dumps(message))
            connection.close()
            print(f"✅ Sent RabbitMQ Message: {message}")
        except Exception as e:
            logging.error(f"❌ Failed to send RabbitMQ message: {e}")

    # ✅ Upload Dataset and Register in Neo4j
    async def upload_dataset(self, data_files, dataset_id, data_type):
        dataset_dir = os.path.join(self.base_storage_address, data_type, dataset_id)
        os.makedirs(dataset_dir, exist_ok=True)

        for file_path in data_files:
            label = os.path.basename(os.path.dirname(file_path))
            label_dir = os.path.join(dataset_dir, label)
            os.makedirs(label_dir, exist_ok=True)

            file_extension = os.path.splitext(file_path)[1]
            dest_file_name = os.path.splitext(os.path.basename(file_path))[0] + file_extension
            dest_file_path = os.path.join(label_dir, dest_file_name)

            async with aiofiles.open(file_path, 'rb') as src, aiofiles.open(dest_file_path, 'wb') as dst:
                await dst.write(await src.read())

        print(f"✅ Dataset '{dataset_id}' uploaded.")
        self.provenance.create_dataset(dataset_id, f"{data_type} Dataset", dataset_dir)
        self.send_rabbitmq_message("dataset_uploaded", dataset_id)




# kinetcs dataset process
    def load_metadata(self, labels_csv, video_list_txt):
        self.metadata['labels'] = pd.read_csv(labels_csv)
        self.metadata['video_map'] = {}
        with open(video_list_txt, "r") as f:
            for line in f:
                video_name, label = line.strip().split()
                self.metadata['video_map'][video_name] = label

    def get_label(self, video_name):
        if 'video_map' not in self.metadata:
            raise ValueError("Metadata not loaded. Call load_metadata() first.")
        return self.metadata['video_map'].get(video_name, "Unknown")

    # ✅ Process Kinetics Videos
    def process_kinetics_video(self, video_path, num_frames=8):
        """Processes Kinetics dataset videos."""
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        output_dir = os.path.join(self.base_storage_address, "videos", video_name)
        os.makedirs(output_dir, exist_ok=True)

        try:
            container = av.open(video_path)
            frames = [frame.to_image() for frame in container.decode(video=0)]
            sampled_frames = [frames[i] for i in np.linspace(0, len(frames) - 1, num_frames, dtype=int)]

            for idx, frame in enumerate(sampled_frames):
                frame.save(os.path.join(output_dir, f"frame_{idx + 1}.jpg"))

            self.provenance.create_dataset(video_name, "Kinetics Video", video_path)
            self.provenance.create_processing_step("Extract Frames", "video_processing", f"Extracted {num_frames} frames from {video_name}")
            self.provenance.link_dataset_to_processing(video_name, "Extract Frames")

            print(f"✅ Processed {video_name} - Frames saved in {output_dir}")
            return {"video_name": video_name, "frame_dir": output_dir}

        except Exception as e:
            print(f"❌ ERROR processing {video_path}: {e}")
            self.send_rabbitmq_message("video_processing_failed", video_name, str(e))
            return {"error": str(e)}
