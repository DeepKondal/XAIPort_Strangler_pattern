# Use an official Python runtime as a parent image
FROM python:3.9

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /usr/src/app
COPY . /app

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Make port 8002 available to the world outside this container
EXPOSE 8002

# Run model_server.py when the container launches
#CMD ["python", "model_server.py"]
CMD ["uvicorn", "model_server:app", "--host", "0.0.0.0", "--port", "8002"]
