import streamlit as st
import pandas as pd
import os
from fastai.vision import *
from fastai.vision.all import *
from fastai.learner import *
from fastai.metrics import accuracy, Precision, Recall
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import subplots
import torch
import math
import subprocess  # For executing YOLO commands

os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

# Use st.cache_data instead of deprecated st.cache
@st.cache_data
def load_data(file_path):
    try:
        data = pd.read_csv(file_path)
        return data
    except FileNotFoundError:
        st.error("Data file not found. Please make sure the file is available.")
        return None

# Function to train and evaluate model
def train_and_evaluate_model(data_path, output_path, epoch, batch_size, learning_rate, model_type):
    td_path = Path(data_path)
    
    if model_type == "Fast.AI":
        # Read the CSV file
        df = pd.read_csv(td_path / '_annotations.csv')

        def get_label(fn):
            filename = fn.name
            row = df[df['filename'] == filename]
            if row.empty:
                return "Unknown"
            return row['class'].values[0]

        # Define the DataBlock
        dblock = DataBlock(
            blocks=(ImageBlock, CategoryBlock),
            get_items=get_image_files,
            get_y=get_label,
            splitter=RandomSplitter(0.2),
            item_tfms=Resize(128),
            batch_tfms=aug_transforms()
        )

        # Create DataLoaders
        dls = dblock.dataloaders(td_path, bs=batch_size, device='cpu')

        # Show a batch of images with their class labels
        dls.show_batch(max_n=9, figsize=(10, 10))

        # Define precision and recall as custom metrics
        precision = Precision(average='macro')
        recall = Recall(average='macro')
        metrics = [accuracy, precision, recall]

        learn = vision_learner(dls, models.resnet18, pretrained=True, metrics=metrics)
        learn.dls.device = 'cpu'
        learn.model.to('cpu')

        # Find optimal learning rate
        with st.spinner(text="Loading Optimal Learning Rate"):
            lr_find_result = learn.lr_find()

        valley_lr = lr_find_result.valley
        st.write("Obtained Learning Rate: ", valley_lr)

        with st.spinner(text="Model is Training"):
            learn.fit_one_cycle(epoch, slice(valley_lr))

        @patch
        @delegates(subplots)
        def plot_metrics(self: Recorder, nrows=None, ncols=None, figsize=None, **kwargs):
            metrics = np.stack(self.values)
            names = self.metric_names[1:-1]
            n = len(names) - 1
            if nrows is None and ncols is None:
                nrows = int(math.sqrt(n))
                ncols = int(np.ceil(n / nrows))
            elif nrows is None: nrows = int(np.ceil(n / ncols))
            elif ncols is None: ncols = int(np.ceil(n / nrows))
            figsize = figsize or (ncols * 6, nrows * 4)
            fig, axs = subplots(nrows, ncols, figsize=figsize, **kwargs)
            fig.subplots_adjust(hspace=0.5)
            axs = [ax if i < n else ax.set_axis_off() for i, ax in enumerate(axs.flatten())][:n]
            for i, (name, ax) in enumerate(zip(names, [axs[0]] + axs)):
                ax.plot(metrics[:, i], color='#1f77b4' if i == 0 else '#ff7f0e', label='valid' if i > 0 else 'train')
                ax.set_title(name if i > 1 else 'losses')
                ax.set_xlabel('Epoch')
                ax.set_ylabel('Data Loss' if 'loss' in name else 'Accuracy')
                ax.legend(loc='best')
            st.pyplot(fig)
        
        learn.recorder.plot_metrics()

        # Export the trained model
        td_path = Path(output_path)
        learn.export(td_path / 'FastAI.pkl')
        st.write("Training is Successful, Model is Exported to output path, Named FastAI.pkl")
    
    elif model_type == "YOLOv8":
        # Ensure output directory exists
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        
        # Define YOLOv8 training command
        command = f"yolo task=detect mode=train model=best.pt data={data_path}/data.yaml imgsz=640 epochs={epoch} batch={batch_size} lr0={learning_rate} plots=True"
        
        # Print the command to check
        st.write(f"Running command: {command}")

        # Execute the command
        try:
            result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
            st.write("Training complete.")
            st.write(result.stdout)  # Display training logs
            st.write("Training logs:", result.stderr)  # Display errors if any
            
            # Check if trained model is saved and move to output path
            trained_model_path = "runs/detect/train/weights/best.pt"  # Default path, adjust if necessary
            if os.path.exists(trained_model_path):
                os.makedirs(output_path, exist_ok=True)
                output_model_path = os.path.join(output_path, 'best_retrained.pt')
                os.rename(trained_model_path, output_model_path)
                st.write(f"Model saved to {output_model_path}")
            else:
                st.error("Trained model file not found. Ensure training completed successfully.")
                
        except subprocess.CalledProcessError as e:
            st.error(f"Error during YOLOv8 training: {e.stderr}")

# Streamlit UI
st.header('Model Training')

data_path = st.text_input("Enter data file path:", key="data_path")
output_path = st.text_input("Enter output file path for model export:", key="output_path")

model_type = st.radio('Select Model Type:', ['YOLOv8', 'Fast.AI'])

if model_type == "Fast.AI":
    epoch = st.number_input('Insert epoch', min_value=1, value=1, step=1)
    batch_size = st.number_input('Insert batch size', min_value=1, value=1, step=1)
    learning_rate = st.number_input('Insert a learning rate', min_value=0.0001, value=0.01, step=0.0001)
else:
    epoch = st.number_input('Insert epoch', min_value=1, value=1, step=1)
    batch_size = st.number_input('Insert batch size', min_value=1, value=1, step=1)
    learning_rate = st.number_input('Insert a learning rate', min_value=0.0001, value=0.01, step=0.0001)

training_method = st.radio("Select what you want to Train the Model with:", ("CPU (Default)", "GPU (will be used if selected and available)"))

if st.button("Start Training"):
    if data_path and output_path:
        if os.path.exists(data_path) and os.path.exists(output_path):
            train_and_evaluate_model(data_path, output_path, epoch, batch_size, learning_rate, model_type)
        else:
            st.error("One or more paths are invalid. Please check your file paths.")
    else:
        st.error("Please provide both data and output paths.")