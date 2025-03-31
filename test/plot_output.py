import cv2
import matplotlib.pyplot as plt
import random
import numpy as np
from typing import Tuple, Dict, List, Optional

from utils import get_image_paths, get_image_name_cli

KEY_INPUT = "Input"
KEY_ORIGINAL = "Original image"
KEY_OUTPUT = "Output (4x upscaled)"

def _load_images(input_path: str, output_path: str, original_path: Optional[str]) -> Dict[str, Optional[np.ndarray]]:
    """Loads images using specified paths and standard keys. Returns empty dict on fatal error."""
    images = {
        KEY_INPUT: cv2.imread(input_path),
        KEY_OUTPUT: cv2.imread(output_path),
        KEY_ORIGINAL: None
    }
    if original_path:
        images[KEY_ORIGINAL] = cv2.imread(original_path)

    if images[KEY_INPUT] is None:
        print(f"Error: Could not read input image at {input_path}")
        return {}
    if images[KEY_OUTPUT] is None:
        print(f"Error: Could not read output image at {output_path}")
        return {}

    if original_path and images[KEY_ORIGINAL] is None:
        print(f"Warning: Could not read original image at {original_path}. Proceeding without it.")
        images.pop(KEY_ORIGINAL)


    return images

def _calculate_scale_factors(images: Dict[str, Optional[np.ndarray]]) -> Tuple[Optional[float], Dict[str, float]]:
    """Calculates scale factor between output and input, and relative scales for all images."""
    output_image = images.get(KEY_OUTPUT)
    input_image = images.get(KEY_INPUT)
    original_image = images.get(KEY_ORIGINAL)

    if output_image is None or input_image is None:
        return None, {}

    out_h, out_w = output_image.shape[:2]
    in_h, in_w = input_image.shape[:2]

    if in_h == 0 or in_w == 0:
        print("Error: Input image has zero dimension.")
        return None, {}

    scale_factor_h = out_h / in_h
    scale_factor_w = out_w / in_w

    if abs(scale_factor_h - scale_factor_w) > 0.1:
        print(f"Warning: Non-uniform scaling detected (H: {scale_factor_h:.2f}, W: {scale_factor_w:.2f}). Using width scale factor.")

    scale_factor = scale_factor_w
    if scale_factor <= 0:
        print("Error: Invalid scale factor calculated.")
        return None, {}

    img_scales = {
        KEY_OUTPUT: 1.0,
        KEY_INPUT: 1.0 / scale_factor
    }

    if original_image is not None:
        orig_h, orig_w = original_image.shape[:2]
        if orig_h == 0 or orig_w == 0:
            print("Warning: Original image has zero dimension. Skipping original scale calculation.")
            images.pop(KEY_ORIGINAL, None)
        elif orig_h == out_h and orig_w == out_w:
             img_scales[KEY_ORIGINAL] = 1.0
        else:
            orig_scale_factor_w = out_w / orig_w if orig_w > 0 else scale_factor
            orig_scale_factor_h = out_h / orig_h if orig_h > 0 else scale_factor
            if abs(orig_scale_factor_h - orig_scale_factor_w) > 0.1:
                 print(f"Warning: Original image has non-uniform scaling relative to output. Using width factor.")

            img_scales[KEY_ORIGINAL] = 1.0 / orig_scale_factor_w
            print(f"Info: Original image scale relative to output: {1.0 / img_scales[KEY_ORIGINAL]:.1f}x")

    valid_keys = list(images.keys())
    img_scales = {k: v for k, v in img_scales.items() if k in valid_keys}

    return scale_factor, img_scales

def _get_plot_config(images: Dict[str, Optional[np.ndarray]]) -> Tuple[List[str], int]:
    potential_titles = [KEY_INPUT, KEY_ORIGINAL, KEY_OUTPUT]
    titles = [title for title in potential_titles if title in images and images[title] is not None]
    num_cols = len(titles)
    return titles, num_cols

def _get_scaled_crop_coords(y_out: int, x_out: int, crop_size: int, scale: float, img_shape: Tuple[int, int]) -> Optional[Tuple[int, int, int]]:
    y = int(round(y_out * scale))
    x = int(round(x_out * scale))
    cs = int(round(crop_size * scale))
    img_h, img_w = img_shape

    if y + cs > img_h:
        y = max(0, img_h - cs)
    if x + cs > img_w:
        x = max(0, img_w - cs)
    y = max(0, y)
    x = max(0, x)

    if y < 0 or x < 0 or cs <= 0 or y + cs > img_h or x + cs > img_w:
        return None
    return y, x, cs

def plot_images(input_image_path: str,
                output_image_path: str,
                original_image_path: Optional[str] = None,
                crop_size: int = 300,
                zoom_factor: int = 4,
                num_rows: int = 4,
                seed: Optional[int] = None):
    if seed is not None:
        random.seed(seed)

    loaded_images = _load_images(input_image_path, output_image_path, original_image_path)
    if not loaded_images: return

    images = {k: v for k, v in loaded_images.items() if v is not None}

    scale_factor, img_scales = _calculate_scale_factors(images)
    if scale_factor is None: return

    titles, num_cols = _get_plot_config(images)
    if num_cols == 0: 
        print("Error: No valid images available for plotting.")
        return

    output_image = images[KEY_OUTPUT]
    out_h, out_w = output_image.shape[:2]

    effective_crop_size = crop_size
    if effective_crop_size > out_h or effective_crop_size > out_w:
        print(f"Warning: Crop size ({crop_size}) is larger than output image ({out_h}x{out_w}). Adjusting.")
        effective_crop_size = min(out_h, out_w)
    if effective_crop_size <= 0:
        print(f"Error: Invalid crop size ({effective_crop_size}) determined.")
        return

    fig, axs = plt.subplots(num_rows, num_cols, figsize=(num_cols * 4.5, num_rows * 4), squeeze=False)
    target_zoom_size = (int(round(effective_crop_size * zoom_factor)), int(round(effective_crop_size * zoom_factor)))
    if target_zoom_size[0] <= 0 or target_zoom_size[1] <= 0:
        print(f"Error: Invalid target zoom size calculated: {target_zoom_size}")
        return

    for i in range(num_rows):
        max_y = out_h - effective_crop_size
        max_x = out_w - effective_crop_size
        if max_y < 0 or max_x < 0:
            print(f"Error: Cannot determine valid crop window. Output: {out_h}x{out_w}, Crop: {effective_crop_size}. Skipping row {i+1}.")
            for j, title in enumerate(titles):
                 ax = axs[i, j]
                 ax.imshow(np.zeros((target_zoom_size[1], target_zoom_size[0], 3), dtype=np.uint8))
                 ax.set_title(f'{title} Crop {i+1} (Error)' if i == 0 else "")
                 ax.axis('off')
            continue

        y_out = random.randint(0, max_y)
        x_out = random.randint(0, max_x)

        for j, title in enumerate(titles):
            img = images[title]
            scale = img_scales[title]

            crop_coords = _get_scaled_crop_coords(y_out, x_out, effective_crop_size, scale, img.shape[:2])

            ax = axs[i, j]

            if crop_coords is None:
                print(f"Warning: Invalid crop calculated for {title} in row {i+1}. Skipping.")
                zoom = np.zeros((target_zoom_size[1], target_zoom_size[0], 3), dtype=np.uint8)
                plot_title = f'{title} Crop {i+1} (Error)' if i == 0 else ""
            else:
                y, x, cs = crop_coords
                if cs <= 0:
                    print(f"Warning: Zero crop size calculated for {title} in row {i+1}. Skipping.")
                    zoom = np.zeros((target_zoom_size[1], target_zoom_size[0], 3), dtype=np.uint8)
                    plot_title = f'{title} Crop {i+1} (Error)' if i == 0 else ""
                else:
                    crop = img[y:y+cs, x:x+cs]
                    if crop.size == 0:
                         print(f"Warning: Empty crop obtained for {title} in row {i+1}. Skipping.")
                         zoom = np.zeros((target_zoom_size[1], target_zoom_size[0], 3), dtype=np.uint8)
                         plot_title = f'{title} Crop {i+1} (Error)' if i == 0 else ""
                    else:
                         zoom = cv2.resize(crop, target_zoom_size, interpolation=cv2.INTER_NEAREST)
                         plot_title = f'{title} Crop {i+1}' if i == 0 else ""

            ax.imshow(cv2.cvtColor(zoom, cv2.COLOR_BGR2RGB))
            ax.set_title(plot_title)
            ax.axis('off')

    plt.tight_layout(pad=0.5, h_pad=1.0, w_pad=0.5)
    plt.show()

if __name__ == '__main__':
    image_name = get_image_name_cli()
    input_path, output_path, original_path = get_image_paths(image_name)
    plot_images(input_path, output_path, original_path)
