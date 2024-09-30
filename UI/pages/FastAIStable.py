
from fastai.vision import *
from fastai.vision.all import *
from fastai.learner import *
from fastai.metrics import accuracy, Precision, Recall
from pathlib import Path
import os
import pandas as pd
import torch
import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import subplots
import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

if torch.cuda.is_available():
    print("GPU is available")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

else:
    print("GPU is not available")
if not 'notebookDir' in globals():
    notebookDir = os.getcwd()


def train():
    td_path = Path("/Users/lucasliew/Desktop/SIT/Year 2/Tri 3/ITP/GUI/ITPUserInterface/FASTAItrainingdata/train")
    # Read the CSV file
    df = pd.read_csv(td_path/'_annotations.csv')

    def get_label(fn):
        # Extract the filename from the path
        filename = fn.name
        # Find the row in the dataframe where the filename matches
        row = df[df['filename'] == filename]
        # Debugging: Print out the filename and row to understand the issue
        if row.empty:
            #print(f"No matching entry for {filename}")
            return "Unknown"  # or handle it in a way suitable for your case
        # Return the label; assuming there's one label per filename
        return row['class'].values[0]

    # Define the DataBlock
    dblock = DataBlock(
        blocks=(ImageBlock, CategoryBlock),
        get_items=get_image_files,
        get_y=get_label,  # Use the custom get_y function
        splitter=RandomSplitter(0.2),
        item_tfms=Resize(128),
        batch_tfms=aug_transforms()
    )

    # Create DataLoaders
    dls = dblock.dataloaders(td_path, bs=8, device='cpu')

    # Show a batch of images with their class labels
    dls.show_batch(max_n=9, figsize=(10, 10))



    # Define precision and recall as custom metrics
    precision = Precision(average='macro')
    recall = Recall(average='macro')

    # Combine all metrics
    metrics = [accuracy, precision, recall]
    print("DONE")

    import time
    time.sleep(3)

    learn = vision_learner(dls, models.resnet18, pretrained=True, metrics=metrics)

    # Make sure the model and data are on the same device
    learn.dls.device = 'cpu'  # or 'mps' if using Metal Performance Shaders
    learn.model.to('cpu')     # or 'mps' if using Metal Performance Shaders




    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)

    # Find optimal learning rate
    lr_find_result = learn.lr_find()

    # Extract the valley learning rate
    valley_lr = lr_find_result.valley


    learn.fit_one_cycle(1, slice(valley_lr)) # adjust the epoch and learning rate in the arguement of .fit_one_cycle(epoch,learning rate)



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
        fig.subplots_adjust(hspace=0.5)  # Adjust vertical spacing between subplots
        axs = [ax if i < n else ax.set_axis_off() for i, ax in enumerate(axs.flatten())][:n]
        for i, (name, ax) in enumerate(zip(names, [axs[0]] + axs)):
            ax.plot(metrics[:, i], color='#1f77b4' if i == 0 else '#ff7f0e', label='valid' if i > 0 else 'train')
            ax.set_title(name if i > 1 else 'losses')
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Data Loss' if 'loss' in name else 'Accuracy')
            ax.legend(loc='best')
        plt.show()
        
    # Now call the plot_metrics function
    learn.recorder.plot_metrics()


    # Export the trained model
    learn.export('bs8ep10.pkl')

# def classify():

#     # after export
#     #========================================================
#     from fastai.vision.all import load_learner, PILImage

#     # Load the exported model
#     learn_inf = load_learner('/Users/lucasliew/Desktop/SIT/Year 2/Tri 3/ITP/untitled folder/bs8ep10.pkl')

#     # Path to the new image you want to predict
#     img_path = Path('/Users/lucasliew/Desktop/SIT/Year 2/Tri 3/ITP/predictionimg/1200px-ZungenCa2a_jpg.rf.484a48535552eec46aa1144b0d5f2d6b.jpg')

#     # Open the image
#     img = PILImage.create(img_path)

#     # Make a prediction
#     pred_class, pred_idx, outputs = learn_inf.predict(img)
#     print(f"Predicted class: {pred_class}")



#     from fastai.vision.all import load_learner, PILImage
#     from pathlib import Path
#     import matplotlib.pyplot as plt

#     # Specify the correct path to the model file
#     model_path = Path('/Users/lucasliew/Desktop/fastai/newcode/for_training/train/bs8ep30.pkl')

#     # Load the exported model
#     learn_inf = load_learner(model_path)

#     # Path to the new image you want to predict
#     img_path = Path('/Users/lucasliew/Desktop/SIT/Year 2/Tri 3/ITP/predictionimg/1200px-ZungenCa2a_jpg.rf.484a48535552eec46aa1144b0d5f2d6b.jpg')

#     # Open the image
#     img = PILImage.create(img_path)

#     # Make a prediction
#     pred_class, pred_idx, outputs = learn_inf.predict(img)
#     print(f"Predicted class: {pred_class}")

#     # Display the image with the prediction
#     plt.imshow(img)
#     plt.title(f"Predicted class: {pred_class}")
#     plt.axis('off')
#     plt.show()

if __name__ == '__main__':
    train()