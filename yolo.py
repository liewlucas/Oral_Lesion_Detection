from IPython import display
display.clear_output()
import ultralytics
ultralytics.checks()
from ultralytics import YOLO

from IPython.display import display, Image
import os
from roboflow import Roboflow

# Define your home directory and dataset path
home_dir = os.path.expanduser("~")
datasets_dir = os.path.join(home_dir, "datasets")

# Create the datasets directory if it doesn't exist
if not os.path.exists(datasets_dir):
    os.makedirs(datasets_dir)

# Change the current working directory to the datasets directory
os.chdir(datasets_dir)

# Install roboflow (if needed)
# !pip install roboflow

# Initialize Roboflow and download the dataset

from roboflow import Roboflow
rf = Roboflow(api_key="qVdeefhiqH0NWWECJqGG")
project = rf.workspace("test-mpt3g").project("for_training")
version = project.version(8)
dataset = version.download("yolov8")
import os

# Define parameter ranges
epochs_list = [100]
learning_rates = [0.01]
batch_sizes = [8,16]

# Base command
base_command = "yolo task=detect mode=train model=yolov8s.pt data={}/data.yaml imgsz=500 plots=True".format(dataset.location)

# Iterate over parameter combinations
for epoch in epochs_list:
    for lr in learning_rates:
        for batch_size in batch_sizes:
            # Construct command with current parameters
            command = base_command + " epochs={} batch={} lr0={}".format(epoch, batch_size, lr)

            # Print out statement for the current training configuration
            print(f"Training with Epochs: {epoch}, Batch Size: {batch_size}, Learning Rate: {lr}")

            # Execute command
            os.system(command)
