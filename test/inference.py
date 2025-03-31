import argparse
import cv2
import glob
import numpy as np
import os
import torch
from enum import Enum
from torch.cuda.amp import autocast

from drct.archs.DRCT_arch import DRCT
from utils import get_image_name_cli, get_image_paths

class ModelType(Enum):
    DRCT = 'DRCT'

class Precision(Enum):
    FP32 = 'fp32'
    FP16 = 'fp16'

# This class is a modified version of the original DRCT inference script:
# https://github.com/ming053l/DRCT/blob/main/inference_fp16.py
class DcrtInference:
    def __init__(self, model_path, scale, drct_params, precision=Precision.FP32, device=None):
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.precision = precision
        self.scale = scale
        self.tile = drct_params.get('tile')
        self.tile_overlap = drct_params.get('tile_overlap', 32)
        # Hardcoded window_size based on the original script
        self.window_size = 16

        self.model = self._load_model(model_path)
        self.model.eval()
        self.model = self.model.to(self.device)

        if self.precision == Precision.FP16:
            if self.device == torch.device('cpu'):
                print("Warning: FP16 is selected but running on CPU. FP16 is most effective on CUDA.")
            else:
                self.model = self.model.half()

    def _load_model(self, model_path):
        # Parameters from the original script, consider making these configurable
        model = DRCT(upscale=self.scale, in_chans=3, img_size=64, window_size=self.window_size, compress_ratio=3, squeeze_factor=30,
                     conv_scale=0.01, overlap_ratio=0.5, img_range=1., depths=[6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6],
                     embed_dim=180, num_heads=[6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6], gc=32,
                     mlp_ratio=2, upsampler='pixelshuffle', resi_connection='1conv')

        load_net = torch.load(model_path, map_location=self.device)
        params = load_net['params']
        model.load_state_dict(params, strict=True)
        print(f"Model loaded from {model_path}")
        return model

    def _preprocess(self, img_path):
        img = cv2.imread(img_path, cv2.IMREAD_COLOR).astype(np.float32) / 255.
        img = torch.from_numpy(np.transpose(img[:, :, [2, 1, 0]], (2, 0, 1))).float()
        img = img.unsqueeze(0)
        if self.precision == Precision.FP16 and self.device != torch.device('cpu'):
             img = img.half()
        img = img.to(self.device)
        return img

    def _inference_tile(self, img_lq):
        b, c, h, w = img_lq.size()
        tile = min(self.tile, h, w)
        if tile % self.window_size != 0:
             # Adjust tile size to be divisible by window_size if necessary
             tile = (tile // self.window_size) * self.window_size
             print(f"Adjusted tile size to {tile} to be divisible by window_size {self.window_size}")
        assert tile > self.tile_overlap, "Tile size must be larger than tile overlap"

        stride = tile - self.tile_overlap
        sf = self.scale

        # Calculate padding if needed to make dimensions multiples of stride
        h_pad = (stride - (h - tile) % stride) % stride
        w_pad = (stride - (w - tile) % stride) % stride

        img_lq_padded = torch.nn.functional.pad(img_lq, (0, w_pad, 0, h_pad), mode='reflect')
        h_padded, w_padded = img_lq_padded.shape[2:]

        h_idx_list = list(range(0, h_padded - tile + 1, stride))
        w_idx_list = list(range(0, w_padded - tile + 1, stride))

        E = torch.zeros(b, c, h_padded * sf, w_padded * sf, dtype=img_lq.dtype, device=self.device)
        W = torch.zeros_like(E)

        for h_idx in h_idx_list:
            for w_idx in w_idx_list:
                in_patch = img_lq_padded[..., h_idx:h_idx + tile, w_idx:w_idx + tile]
                out_patch = self.model(in_patch)
                out_patch_mask = torch.ones_like(out_patch)

                E[..., h_idx * sf:(h_idx + tile) * sf, w_idx * sf:(w_idx + tile) * sf].add_(out_patch)
                W[..., h_idx * sf:(h_idx + tile) * sf, w_idx * sf:(w_idx + tile) * sf].add_(out_patch_mask)

        output = E.div_(W)
        # Crop back to original size * scale
        return output[..., :h * sf, :w * sf]


    def _inference_full(self, img_lq):
         _, _, h_old, w_old = img_lq.size()
         # Pad to be divisible by window_size
         h_pad = (h_old // self.window_size + 1) * self.window_size - h_old
         w_pad = (w_old // self.window_size + 1) * self.window_size - w_old
         img = torch.cat([img_lq, torch.flip(img_lq, [2])], 2)[:, :, :h_old + h_pad, :]
         img = torch.cat([img, torch.flip(img, [3])], 3)[:, :, :, :w_old + w_pad]

         output = self.model(img)
         # Crop back to original size * scale
         output = output[..., :h_old * self.scale, :w_old * self.scale]
         return output

    def _postprocess(self, output_tensor):
        output = output_tensor.data.squeeze().float().cpu().clamp_(0, 1).numpy()
        output = np.transpose(output[[2, 1, 0], :, :], (1, 2, 0))
        output = (output * 255.0).round().astype(np.uint8)
        return output

    def upscale(self, img_path, output_path):
        imgname = os.path.splitext(os.path.basename(img_path))[0]
        print(f'Processing {imgname}...')

        img_lq = self._preprocess(img_path)

        try:
            with torch.no_grad():
                with autocast(enabled=(self.precision == Precision.FP16 and self.device != torch.device('cpu'))):
                    if self.tile is None:
                        output_tensor = self._inference_full(img_lq)
                    else:
                         output_tensor = self._inference_tile(img_lq)

            output_img = self._postprocess(output_tensor)
            cv2.imwrite(output_path, output_img)
            print(f"Saved {output_path}")

        except Exception as error:
            print(f'Error processing {imgname}: {error}')


def inference_super_resolution(
    input_image_path: str,
    output_image_path: str,
    model_type: ModelType,
    model_path: str = "./models/DRCT_L_x4.pth",
    precision: Precision = Precision.FP32,
    drct_params: dict = None
):
    if drct_params is None:
        drct_params = {'scale': 4, 'tile': None, 'tile_overlap': 32}

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if model_type == ModelType.DRCT:
        inferencer = DcrtInference(
            model_path=model_path,
            scale=drct_params.get('scale', 4),
            drct_params=drct_params,
            precision=precision,
            device=device
        )
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    print(f"--- Processing image ---")
    inferencer.upscale(input_image_path, output_image_path)

if __name__ == '__main__':
    image_name = get_image_name_cli()
    _, downscaled_image_path, output_image_path = get_image_paths(image_name)
    
    parser = argparse.ArgumentParser(description="Image Super-Resolution Inference")

    parser.add_argument('--model_type', type=str, default=ModelType.DRCT.value, choices=[m.value for m in ModelType], help='Type of model to use')
    parser.add_argument('--model_path', type=str, default="./models/DRCT_L_x4.pth", help='Path to the pre-trained model weights')
    parser.add_argument('--precision', type=str, default=Precision.FP32.value, choices=[p.value for p in Precision], help='Inference precision (fp32 or fp16)')

    # DRCT specific parameters bundled
    parser.add_argument('--scale', type=int, default=4, help='Super-resolution scale factor (DRCT specific)')
    parser.add_argument('--tile', type=int, default=None, help='Tile size for tiled inference (DRCT specific). None for full image.')
    parser.add_argument('--tile_overlap', type=int, default=32, help='Overlap size for tiled inference (DRCT specific)')

    args = parser.parse_args()

    drct_params = {
        'scale': args.scale,
        'tile': args.tile,
        'tile_overlap': args.tile_overlap
    }

    inference_super_resolution(
        input_image_path=downscaled_image_path,
        output_image_path=output_image_path,
        model_type=ModelType(args.model_type),
        model_path=args.model_path,
        precision=Precision(args.precision),
        drct_params=drct_params
    )