import cv2

from utils import get_image_name_cli, get_image_paths

def decrease_resolution(original_image_path, downscaled_image_path):
    img = cv2.imread(original_image_path)

    if img is None:
        print(f"Error: Could not read image file {original_image_path}")
        return

    new_h, new_w = img.shape[0] // 4, img.shape[1] // 4
    resized_img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

    cv2.imwrite(downscaled_image_path, resized_img)
    print(f"Image successfully downscaled and saved to {downscaled_image_path}")

if __name__ == '__main__':
    image_name = get_image_name_cli()
    original_image_path, downscaled_image_path, output_image_path = get_image_paths(image_name)
    decrease_resolution(original_image_path, downscaled_image_path)
