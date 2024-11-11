import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend
import os
import random
import numpy as np
import tensorflow as tf
from dataset import CustomDataset, CustomConfig
from mrcnn import model as modellib
import matplotlib.pyplot as plt
import skimage.io
from keras.callbacks import Callback

ROOT_DIR = "/mrcnn"
DATASET_DIR = "/mrcnn/dataset/images"
ANNOTATIONS_PATH = "/mrcnn/dataset/annotations.json"  # Path to your JSON file

# Root directory of the project
ROOT_DIR = os.path.abspath(".")

# Directory to save logs and model checkpoints
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# Path to COCO pre-trained weights
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")

# Configuration for Custom Dataset
config = CustomConfig()
config.IMAGES_PER_GPU = 1  # Minimize batch size to avoid memory issues

def load_and_split_dataset(dataset_dir, annotations_path, split_ratio=0.8):
    dataset_full = CustomDataset()
    dataset_full.load_custom(dataset_dir=dataset_dir, annotations_path=annotations_path)
    dataset_full.prepare()

    image_ids = dataset_full.image_ids
    split_index = int(split_ratio * len(image_ids))
    random.shuffle(image_ids)
    train_ids = image_ids[:split_index]
    val_ids = image_ids[split_index:]

    dataset_train = CustomDataset()
    dataset_train.load_custom(dataset_dir=dataset_dir, annotations_path=annotations_path)
    dataset_train.image_info = [dataset_full.image_info[i] for i in train_ids]
    dataset_train.prepare()

    dataset_val = CustomDataset()
    dataset_val.load_custom(dataset_dir=dataset_dir, annotations_path=annotations_path)
    dataset_val.image_info = [dataset_full.image_info[i] for i in val_ids]
    dataset_val.prepare()

    return dataset_train, dataset_val

def save_sample_images(dataset, sample_type="train", num_samples=5):
    os.makedirs("sample_images/{}".format(sample_type), exist_ok=True)
    sample_ids = random.sample(list(dataset.image_ids), num_samples)
    for i, image_id in enumerate(sample_ids):
        image = dataset.load_image(image_id)
        mask, class_ids = dataset.load_mask(image_id)
        merged_mask = np.any(mask, axis=-1).astype("uint8") * 255
        skimage.io.imsave("sample_images/{}/image_{}.jpg".format(sample_type, i), image)
        skimage.io.imsave("sample_images/{}/mask_{}.png".format(sample_type, i), merged_mask)

dataset_train, dataset_val = load_and_split_dataset(DATASET_DIR, ANNOTATIONS_PATH)
save_sample_images(dataset_train, "train", num_samples=5)
save_sample_images(dataset_val, "val", num_samples=5)

# Initialize model in training mode
model = modellib.MaskRCNN(mode="training", config=config, model_dir=MODEL_DIR)
model.load_weights(COCO_MODEL_PATH, by_name=True, exclude=[
    "mrcnn_class_logits", "mrcnn_bbox_fc", "mrcnn_mask"
])

class LossHistory(Callback):
    def __init__(self):
        super().__init__()
        self.epoch_losses = []
        self.epoch_val_losses = []

    def on_epoch_end(self, epoch, logs=None):
        self.epoch_losses.append(logs.get("loss"))
        self.epoch_val_losses.append(logs.get("val_loss"))
        with open("training_logs.txt", "a") as log_file:
            log_file.write("Epoch {}, Loss: {}, Validation Loss: {}\n".format(
                epoch + 1, logs['loss'], logs['val_loss']
            ))

loss_history = LossHistory()
epochs = 10
history = model.train(
    dataset_train,
    dataset_val,
    learning_rate=config.LEARNING_RATE,
    epochs=epochs,
    layers="heads",
    custom_callbacks=[loss_history]
)

def plot_loss_graph(train_losses, val_losses):
    plt.plot(range(1, epochs + 1), train_losses, label="Train Loss")
    plt.plot(range(1, epochs + 1), val_losses, label="Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss over Epochs")
    plt.legend()
    plt.savefig("training_validation_loss.png")
    plt.close()

plot_loss_graph(loss_history.epoch_losses, loss_history.epoch_val_losses)





