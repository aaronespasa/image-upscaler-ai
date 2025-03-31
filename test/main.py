from image_ops import decrease_resolution
from inference import inference_super_resolution, ModelType, Precision
from plot_output import plot_images
from utils import get_image_paths, get_image_name_cli

def test_increase_resolution(decrease_original_resolution=False):
    image_name = get_image_name_cli()
    original_image_path, downscaled_image_path, output_image_path = get_image_paths(image_name)

    if decrease_original_resolution:
        decrease_resolution(original_image_path, downscaled_image_path)

    inference_super_resolution(
        downscaled_image_path, 
        output_image_path, 
        model_type=ModelType.DRCT, 
        precision=Precision.FP16
    )
    
    plot_images(downscaled_image_path, output_image_path, original_image_path if decrease_original_resolution else None)

if __name__ == '__main__':
    test_increase_resolution(decrease_original_resolution=False)