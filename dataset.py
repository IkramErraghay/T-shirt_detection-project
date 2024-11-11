import os
import json
import numpy as np
import skimage.io
import skimage.draw
from mrcnn import utils
from mrcnn.config import Config

class CustomConfig(Config):
    NAME = "tshirt_dataset"
    IMAGES_PER_GPU = 1  # Reduced for Docker memory management
    NUM_CLASSES = 1 + 1  # Background + 1 custom class
    STEPS_PER_EPOCH = 50
    IMAGE_MAX_DIM = 256  # Further reduce image size to save memory
    IMAGE_MIN_DIM = 256
    LEARNING_RATE = 0.01
    OPTIMIZER = 'adam'
class CustomDataset(utils.Dataset):
    def load_custom(self, dataset_dir, annotations_path):
        """
        Load the dataset from a single JSON file with annotations.
        Args:
        - dataset_dir: Directory containing images.
        - annotations_path: Path to the annotations JSON file.
        """
        # Add the class label (Ensure consistency in class name)
        self.add_class("tshirt_dataset", 1, "tshirt")

        # Load the JSON file
        with open(annotations_path) as f:
            annotations_data = json.load(f)

        # Loop over each image entry in the JSON file
        for image_id, data in annotations_data["_via_img_metadata"].items():
            filename = data["filename"]
            file_path = os.path.join(dataset_dir, filename)

            # Check if dimensions are available; if not, read from the image
            height = data.get("height")
            width = data.get("width")
            if height is None or width is None:
                image = skimage.io.imread(file_path)
                height, width = image.shape[:2]

            # Add the image
            self.add_image(
                "tshirt_dataset",  # Class source must match the name used in add_class
                image_id=image_id,
                path=file_path,
                width=width,
                height=height,
                polygons=[{
                    "all_points_x": region["shape_attributes"]["all_points_x"],
                    "all_points_y": region["shape_attributes"]["all_points_y"]
                } for region in data["regions"]]
            )

    def load_mask(self, image_id):
        """
        Load instance masks for an image.
        Returns:
        - masks: A binary mask array of shape [height, width, instance_count]
        - class_ids: A 1D array of class IDs of the instance masks
        """
        info = self.image_info[image_id]
        polygons = info["polygons"]

        # Create an empty mask array, one layer per object
        mask = np.zeros([info["height"], info["width"], len(polygons)], dtype=np.uint8)

        for i, p in enumerate(polygons):
            rr, cc = skimage.draw.polygon(p['all_points_y'], p['all_points_x'])
            mask[rr, cc, i] = 1  # Set mask for each object

        # Return mask and class IDs (all objects are assumed to be class "1" for tshirts)
        class_ids = np.ones([mask.shape[-1]], dtype=np.int32)
        return mask, class_ids
