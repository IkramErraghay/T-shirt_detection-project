import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import os
import random
import skimage.io
import matplotlib.pyplot as plt
from mrcnn import model as modellib
from mrcnn import visualize
from dataset import CustomConfig, CustomDataset  # Make sure CustomConfig matches your training config

# Set up inference configuration
class InferenceConfig(CustomConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

config = InferenceConfig()

# Create model in inference mode
model = modellib.MaskRCNN(mode="inference", model_dir="/mrcnn/logs/tshirt_dataset20241110T2124", config=config)

# Load the trained weights
model.load_weights("/mrcnn/logs/tshirt_dataset20241110T2124/mask_rcnn_tshirt_dataset_0010.h5", by_name=True)

# Define the folder containing test images and output folder for results
test_images_folder = "image t-shirt 2/"  # Replace with the actual path to the test images folder
output_folder = "output/"  # Folder to save results

# Ensure output directory exists
os.makedirs(output_folder, exist_ok=True)
# Run detection on each image in the test folder
for image_name in os.listdir(test_images_folder):
    if image_name.lower().endswith(('.png', '.jpg', '.jpeg')):  # Check for image files
        image_path = os.path.join(test_images_folder, image_name)
        image = skimage.io.imread(image_path)

        # Run detection
        results = model.detect([image], verbose=1)

        # Extract results
        r = results[0]
        save_path = os.path.join(output_folder, "inference_" + image_name)

        # Visualize and save the results
        visualize.display_instances(
            image, r['rois'], r['masks'], r['class_ids'],
            ["BG", "tshirt"], r['scores']
        )
        plt.savefig(save_path)
        plt.close()
        print("Saved result for", image_name, "at", save_path)
