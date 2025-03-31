import os
from config import ORIGINAL_IMAGES_PATH, DOWNSCALED_IMAGES_PATH, OUTPUT_IMAGES_PATH

def get_image_name_cli():
    while True:
        image_name = input(f"Enter the name of an image inside the folders ./{ORIGINAL_IMAGES_PATH}/ and ./{DOWNSCALED_IMAGES_PATH}/:\n>>> ").strip()
        if not image_name:
            print("Error: Image name cannot be empty.")
            continue
        original_image_path = os.path.join(ORIGINAL_IMAGES_PATH, image_name)
        downscaled_image_path = os.path.join(DOWNSCALED_IMAGES_PATH, image_name)
        if not os.path.exists(original_image_path) and not os.path.exists(downscaled_image_path):
            raise ValueError(f"Error: Image '{image_name}' does not exist.")
            
        return image_name

def get_image_paths(image_name):
    original_image_path = os.path.join(ORIGINAL_IMAGES_PATH, image_name)
    downscaled_image_path = os.path.join(DOWNSCALED_IMAGES_PATH, image_name)
    output_image_path = os.path.join(OUTPUT_IMAGES_PATH, image_name)

    return original_image_path, downscaled_image_path, output_image_path
