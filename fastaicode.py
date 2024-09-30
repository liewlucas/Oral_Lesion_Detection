{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Install FastAI Library (Optional if you installed fastai in anaconda)\n",
    "\n",
    "# pip install -U fastai"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Steering Model FastAI Neural Network Training Code"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "id": "fqiCu1AFH74N"
   },
   "outputs": [],
   "source": [
    "# FastAI Library for lightweight CNN for image deep learning neural network\n",
    "\n",
    "from fastai.vision import *\n",
    "from fastai.vision.all import *\n",
    "from fastai.learner import *\n",
    "from fastai.metrics import accuracy, Precision, Recall\n",
    "from pathlib import Path"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# store the notebook directory (which is the current working directory, make sure your training data is in the same folder)\n",
    "import os\n",
    "import torch\n",
    "\n",
    "if torch.cuda.is_available():\n",
    "    print(\"GPU is available\")\n",
    "    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')\n",
    "\n",
    "else:\n",
    "    print(\"GPU is not available\")\n",
    "if not 'notebookDir' in globals():\n",
    "    notebookDir = os.getcwd()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Recalling Data Analytics, three main hyperparameters that might affect the AI model's performance are learning rate, epoch and batch size."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# import os\n",
    "# import pandas as pd\n",
    "# from fastai.vision.all import *\n",
    "\n",
    "# # Set environment variable for MPS fallback\n",
    "# os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'\n",
    "\n",
    "# td_path = Path('/Users/lucasliew/Desktop/fastai/newcode/for_training/train')\n",
    "# # td_path = Path('/Users/lucasliew/Desktop/fastai/newcode/for_training.v5i.multiclass/train')\n",
    "# df = pd.read_csv(td_path/'_annotations.csv')\n",
    "# print(df)\n",
    "\n",
    "# # Explicitly specify CPU\n",
    "# data = ImageDataLoaders.from_df(df, path=td_path, valid_pct=0.2, \n",
    "#                                 bs=4, seed=42, \n",
    "#                                 item_tfms=Resize(224), \n",
    "#                                 batch_tfms=[*aug_transforms(), Normalize.from_stats(*imagenet_stats)],\n",
    "#                                 device='cpu')  # Use 'cpu' device\n",
    "\n",
    "# img, label = data.train_ds[0]\n",
    "# print(data.train_ds[0])\n",
    "# # label = data.train_ds[3]\n",
    "# print(\"IMAGE SHAPE: \", img.shape, \"LABEL: \", label)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "import os\n",
    "import pandas as pd\n",
    "from fastai.vision.all import *\n",
    "\n",
    "# Set environment variable for MPS fallback\n",
    "# os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'\n",
    "\n",
    "# Define path to data\n",
    "td_path = Path('/Users/lucasliew/Desktop/fastai/newcode/for_training/train')\n",
    "\n",
    "# Read the CSV file\n",
    "df = pd.read_csv(td_path/'_annotations.csv')\n",
    "\n",
    "# Print the dataframe to confirm\n",
    "# print(df.head())\n",
    "\n",
    "def get_label(fn):\n",
    "    # Extract the filename from the path\n",
    "    filename = fn.name\n",
    "    # Find the row in the dataframe where the filename matches\n",
    "    row = df[df['filename'] == filename]\n",
    "    # Debugging: Print out the filename and row to understand the issue\n",
    "    if row.empty:\n",
    "        #print(f\"No matching entry for {filename}\")\n",
    "        return \"Unknown\"  # or handle it in a way suitable for your case\n",
    "    # Return the label; assuming there's one label per filename\n",
    "    return row['class'].values[0]\n",
    "\n",
    "# Define the DataBlock\n",
    "dblock = DataBlock(\n",
    "    blocks=(ImageBlock, CategoryBlock),\n",
    "    get_items=get_image_files,\n",
    "    get_y=get_label,  # Use the custom get_y function\n",
    "    splitter=RandomSplitter(0.2),\n",
    "    item_tfms=Resize(128),\n",
    "    batch_tfms=aug_transforms()\n",
    ")\n",
    "\n",
    "# Create DataLoaders\n",
    "dls = dblock.dataloaders(td_path, bs=8, device='cpu')\n",
    "\n",
    "# Show a batch of images with their class labels\n",
    "dls.show_batch(max_n=9, figsize=(10, 10))\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "scrolled": true
   },
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Define precision and recall as custom metrics\n",
    "precision = Precision(average='macro')\n",
    "recall = Recall(average='macro')\n",
    "\n",
    "# Combine all metrics\n",
    "metrics = [accuracy, precision, recall]\n",
    "print(\"DONE\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "id": "lTk1hpGjNRpA"
   },
   "outputs": [],
   "source": [
    "# Load CNN model \n",
    "# Here you can choose different models e.g. resnet18 and etc\n",
    "# In FastAI, it is based on a pretrain model resnet and it supports the following weights namely resnet18, resnet34 and resnet 50\n",
    "# resne18 and 34 are smaller network where it can only have up to 2 hidden layers, while resnet 50 can have up to 3. Try with these models to compare the performance.\n",
    "\n",
    "# Move the data to GPU (if necessary)\n",
    "# data = data.to(device)\n",
    "\n",
    "#learn = Learner(data, models.resnet18, loss_func=CrossEntropyLossFlat(), metrics=metrics)\n",
    "\n",
    "learn = vision_learner(dls, models.resnet18, pretrained=True, metrics=metrics)\n",
    "\n",
    "# Make sure the model and data are on the same device\n",
    "learn.dls.device = 'cpu'  # or 'mps' if using Metal Performance Shaders\n",
    "learn.model.to('cpu')     # or 'mps' if using Metal Performance Shaders\n",
    "\n",
    "\n",
    "#learn = vision_learner(data, models.resnet18, pretrained=True, metrics=metrics)\n",
    "\n",
    "# # Move the model to GPU\n",
    "# learn.model = learn.model\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Quick Analysis on Good Learning Rate for the CNN Model\n",
    "#learn.lr_find()\n",
    "# Find optimal learning rate\n",
    "lr_find_result = learn.lr_find()\n",
    "\n",
    "# Extract the valley learning rate\n",
    "valley_lr = lr_find_result.valley"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## FastAI Parameters (https://fastai1.fast.ai/basic_train.html)\n",
    "\n",
    "Two different fit functions namely .fit or fit_one_cycle. Read the documentation in the URL above and investigate how does different fitting, alongside different hyperparameters (pay attention to epoch, batch size and learning rate)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "colab": {
     "base_uri": "https://localhost:8080/",
     "height": 677
    },
    "executionInfo": {
     "elapsed": 104643,
     "status": "ok",
     "timestamp": 1571253492816,
     "user": {
      "displayName": "Maël Hörz",
      "photoUrl": "",
      "userId": "05333326775108365077"
     },
     "user_tz": -120
    },
    "id": "KE_1eGURNUkP",
    "outputId": "b92debf3-3e2a-41f2-a56e-533fdcf59c11"
   },
   "outputs": [],
   "source": [
    "# use the learning rate stated in the valley above (e.g. 0.001). Note, the larger the learning rate, the faster the trying\n",
    "# All training in this script uses CPU\n",
    "\n",
    "learn.fit_one_cycle(1, slice(valley_lr)) # adjust the epoch and learning rate in the arguement of .fit_one_cycle(epoch,learning rate)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Plot Assessment Metric Graph\n",
    "\n",
    "import math\n",
    "import numpy as np\n",
    "import matplotlib.pyplot as plt\n",
    "from matplotlib.pyplot import subplots\n",
    "\n",
    "@patch\n",
    "@delegates(subplots)\n",
    "def plot_metrics(self: Recorder, nrows=None, ncols=None, figsize=None, **kwargs):\n",
    "    metrics = np.stack(self.values)\n",
    "    names = self.metric_names[1:-1]\n",
    "    n = len(names) - 1\n",
    "    if nrows is None and ncols is None:\n",
    "        nrows = int(math.sqrt(n))\n",
    "        ncols = int(np.ceil(n / nrows))\n",
    "    elif nrows is None: nrows = int(np.ceil(n / ncols))\n",
    "    elif ncols is None: ncols = int(np.ceil(n / nrows))\n",
    "    figsize = figsize or (ncols * 6, nrows * 4)\n",
    "    fig, axs = subplots(nrows, ncols, figsize=figsize, **kwargs)\n",
    "    fig.subplots_adjust(hspace=0.5)  # Adjust vertical spacing between subplots\n",
    "    axs = [ax if i < n else ax.set_axis_off() for i, ax in enumerate(axs.flatten())][:n]\n",
    "    for i, (name, ax) in enumerate(zip(names, [axs[0]] + axs)):\n",
    "        ax.plot(metrics[:, i], color='#1f77b4' if i == 0 else '#ff7f0e', label='valid' if i > 0 else 'train')\n",
    "        ax.set_title(name if i > 1 else 'losses')\n",
    "        ax.set_xlabel('Epoch')\n",
    "        ax.set_ylabel('Data Loss' if 'loss' in name else 'Accuracy')\n",
    "        ax.legend(loc='best')\n",
    "    plt.show()\n",
    "    \n",
    "# Now call the plot_metrics function\n",
    "learn.recorder.plot_metrics()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "From the above graphs, these are the following summary we can draw:\n",
    "\n",
    "1. After epoch 17, the data loss is quite low, hence the effecient epoch can be kept at 20\n",
    "2. This is also clarified by the accuracy, precision and recall score being at high levels beyond epoch 17. "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Export the trained model\n",
    "learn.export('bs8ep10.pkl')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "from fastai.vision.all import load_learner, PILImage\n",
    "\n",
    "# Load the exported model\n",
    "learn_inf = load_learner('/Users/lucasliew/Desktop/SIT/Year 2/Tri 3/ITP/untitled folder/bs8ep10.pkl')\n",
    "\n",
    "# Path to the new image you want to predict\n",
    "img_path = Path('/Users/lucasliew/Desktop/SIT/Year 2/Tri 3/ITP/predictionimg/1200px-ZungenCa2a_jpg.rf.484a48535552eec46aa1144b0d5f2d6b.jpg')\n",
    "\n",
    "# Open the image\n",
    "img = PILImage.create(img_path)\n",
    "\n",
    "# Make a prediction\n",
    "pred_class, pred_idx, outputs = learn_inf.predict(img)\n",
    "print(f\"Predicted class: {pred_class}\")\n",
    "\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "from fastai.vision.all import load_learner, PILImage\n",
    "from pathlib import Path\n",
    "import matplotlib.pyplot as plt\n",
    "\n",
    "# Specify the correct path to the model file\n",
    "model_path = Path('/Users/lucasliew/Desktop/fastai/newcode/for_training/train/bs8ep30.pkl')\n",
    "\n",
    "# Load the exported model\n",
    "learn_inf = load_learner(model_path)\n",
    "\n",
    "# Path to the new image you want to predict\n",
    "img_path = Path('/Users/lucasliew/Desktop/SIT/Year 2/Tri 3/ITP/predictionimg/1200px-ZungenCa2a_jpg.rf.484a48535552eec46aa1144b0d5f2d6b.jpg')\n",
    "\n",
    "# Open the image\n",
    "img = PILImage.create(img_path)\n",
    "\n",
    "# Make a prediction\n",
    "pred_class, pred_idx, outputs = learn_inf.predict(img)\n",
    "print(f\"Predicted class: {pred_class}\")\n",
    "\n",
    "# Display the image with the prediction\n",
    "plt.imshow(img)\n",
    "plt.title(f\"Predicted class: {pred_class}\")\n",
    "plt.axis('off')\n",
    "plt.show()\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "colab": {
     "base_uri": "https://localhost:8080/",
     "height": 725
    },
    "executionInfo": {
     "elapsed": 734,
     "status": "ok",
     "timestamp": 1571277919315,
     "user": {
      "displayName": "Maël Hörz",
      "photoUrl": "",
      "userId": "05333326775108365077"
     },
     "user_tz": -120
    },
    "id": "XOYh0BQeN8bm",
    "outputId": "a9fbb1a3-46c2-45d7-ad86-4bad776a1077"
   },
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "accelerator": "GPU",
  "colab": {
   "collapsed_sections": [],
   "name": "Untitled0.ipynb",
   "provenance": []
  },
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.9.13"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}
