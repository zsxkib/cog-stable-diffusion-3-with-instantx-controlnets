# Prediction interface for Cog ⚙️
# https://cog.run/python

import os
import time
from typing import List
import cv2
import torch
import mimetypes
import subprocess
import numpy as np
from PIL import Image
from cog import BasePredictor, Input, Path
from diffusers.models import SD3ControlNetModel
from diffusers import StableDiffusion3ControlNetPipeline
from diffusers.pipelines.stable_diffusion.safety_checker import (
    StableDiffusionSafetyChecker,
)
from transformers import CLIPImageProcessor

mimetypes.add_type("image/webp", ".webp")

MODEL_CACHE = "checkpoints"
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HOME"] = MODEL_CACHE
os.environ["TORCH_HOME"] = MODEL_CACHE
os.environ["HF_DATASETS_CACHE"] = MODEL_CACHE
os.environ["TRANSFORMERS_CACHE"] = MODEL_CACHE
os.environ["HUGGINGFACE_HUB_CACHE"] = MODEL_CACHE

SD3_MODEL_CACHE = "./sd3-cache"
SAFETY_CACHE = "./safety-cache"
FEATURE_EXTRACTOR = "./feature-extractor"
SD3_URL = "https://weights.replicate.delivery/default/sd3/sd3-fp16.tar"
SAFETY_URL = "https://weights.replicate.delivery/default/sdxl/safety-1.0.tar"
BASE_URL = f"https://weights.replicate.delivery/default/sd3-controlnet/{MODEL_CACHE}/"

DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available() else "cpu"
)
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32


def resize_image(
    input_path: str, output_path: str, target_height: int
) -> tuple[str, int, int]:
    # Open the input image
    img = Image.open(input_path)

    # Calculate the aspect ratio of the original image
    original_width, original_height = img.size
    original_aspect_ratio = original_width / original_height

    # Calculate the new width while maintaining the aspect ratio and the target height
    new_width = int(target_height * original_aspect_ratio)

    # Resize the image while maintaining the aspect ratio and fixing the height
    img = img.resize((new_width, target_height), Image.LANCZOS)

    # Save the resized image
    img.save(output_path)

    return output_path, new_width, target_height


def download_weights(url: str, dest: str) -> None:
    # NOTE WHEN YOU EXTRACT SPECIFY THE PARENT FOLDER
    start = time.time()
    print("[!] Initiating download from URL: ", url)
    print("[~] Destination path: ", dest)
    if ".tar" in dest:
        dest = os.path.dirname(dest)
    command = ["pget", "-vf" + ("x" if ".tar" in url else ""), url, dest]
    try:
        print(f"[~] Running command: {' '.join(command)}")
        subprocess.check_call(command, close_fds=False)
    except subprocess.CalledProcessError as e:
        print(
            f"[ERROR] Failed to download weights. Command '{' '.join(e.cmd)}' returned non-zero exit status {e.returncode}."
        )
        raise
    print("[+] Download completed in: ", time.time() - start, "seconds")


class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the model into memory to make running multiple predictions efficient"""

        if not os.path.exists(MODEL_CACHE):
            os.makedirs(MODEL_CACHE)

        model_files = [
            "models--InstantX--SD3-Controlnet-Canny.tar",
            "models--InstantX--SD3-Controlnet-Pose.tar",
            "models--InstantX--SD3-Controlnet-Tile.tar",
            "models--stabilityai--stable-diffusion-3-medium-diffusers.tar",
        ]

        for model_file in model_files:
            url = BASE_URL + model_file

            filename = url.split("/")[-1]
            dest_path = os.path.join(MODEL_CACHE, filename)
            if not os.path.exists(dest_path.replace(".tar", "")):
                download_weights(url, dest_path)

        self.previous_seed = None
        self.generator = torch.Generator(device=DEVICE)

        print("Loading safety checker...")
        if not os.path.exists(SAFETY_CACHE):
            download_weights(SAFETY_URL, SAFETY_CACHE)
        self.safety_checker = StableDiffusionSafetyChecker.from_pretrained(
            SAFETY_CACHE, torch_dtype=torch.float16
        ).to("cuda")
        self.feature_extractor = CLIPImageProcessor.from_pretrained(FEATURE_EXTRACTOR)

    def aspect_ratio_to_width_height(self, aspect_ratio: str):
        aspect_ratios = {
            "1:1": (1024, 1024),
            "16:9": (1344, 768),
            "21:9": (1536, 640),
            "3:2": (1216, 832),
            "2:3": (832, 1216),
            "4:5": (896, 1088),
            "5:4": (1088, 896),
            "9:16": (768, 1344),
            "9:21": (640, 1536),
        }
        return aspect_ratios.get(aspect_ratio)

    def run_safety_checker(self, images):
        safety_checker_input = self.feature_extractor(images, return_tensors="pt").to(
            DEVICE
        )
        np_images = [np.array(val) for val in images]
        _, has_nsfw_content = self.safety_checker(
            images=np_images,
            clip_input=safety_checker_input.pixel_values.to(torch.float16),
        )
        return has_nsfw_content

    @torch.inference_mode()
    def predict(
        self,
        input_image: Path = Input(description="Input image"),
        prompt: str = Input(description="Prompt"),
        negative_prompt: str = Input(
            description="Negative prompt", default="NSFW, nude, naked, porn, ugly"
        ),
        structure: str = Input(
            description="Structure type",
            choices=[
                "canny",
                "pose",
                "tile",
            ],
            default="canny",
        ),
        aspect_ratio: str = Input(
            description="Aspect ratio for the generated image. Note that the model performs best at 1024x1024 resolution. Other sizes may yield suboptimal results.",
            choices=["1:1", "16:9", "21:9", "2:3", "3:2", "4:5", "5:4", "9:16", "9:21"],
            default="1:1",
        ),
        num_outputs: int = Input(
            description="Number of images to output.",
            ge=1,
            le=4,
            default=1,
        ),
        inference_steps: int = Input(
            description="Inference steps", ge=1, le=50, default=25
        ),
        guidance_scale: float = Input(
            description="Guidance scale", ge=0, le=50, default=7.0
        ),
        control_weight: float = Input(
            description="Control weight", ge=0.0, le=1.0, default=0.7
        ),
        seed: int = Input(
            description="Random seed. Leave blank to randomize the seed", default=None
        ),
        output_format: str = Input(
            description="Format of the output images",
            choices=["webp", "jpg", "png"],
            default="webp",
        ),
        output_quality: int = Input(
            description="Quality of the output images, from 0 to 100. 100 is best quality, 0 is lowest quality.",
            default=80,
            ge=0,
            le=100,
        ),
        disable_safety_checker: bool = Input(
            description="Disable safety checker for generated images. This feature is only available through the API. See [https://replicate.com/docs/how-does-replicate-work#safety](https://replicate.com/docs/how-does-replicate-work#safety)",
            default=False,
        ),
    ) -> List[Path]:
        """Run a single prediction on the model"""
        image_in = input_image
        n_prompt = negative_prompt

        if seed is None:
            seed = int.from_bytes(os.urandom(2), "big")
        elif seed != self.previous_seed:
            self.generator.manual_seed(seed)
            self.previous_seed = seed

        print(f"Using seed: {seed}")

        # Preprocess the input image
        input_image = Image.open(str(image_in))
        if input_image.mode != "RGB":
            input_image = input_image.convert("RGB")

        # Load the required model based on the selected structure
        if structure == "canny":
            controlnet = SD3ControlNetModel.from_pretrained(
                "InstantX/SD3-Controlnet-Canny",
                cache_dir=MODEL_CACHE,
                force_download=False,
                local_files_only=True,
            )
            # Canny preprocessing
            image_to_canny = np.array(input_image)
            image_to_canny = cv2.Canny(image_to_canny, 100, 200)
            image_to_canny = image_to_canny[:, :, None]
            image_to_canny = np.concatenate(
                [image_to_canny, image_to_canny, image_to_canny], axis=2
            )
            control_image = Image.fromarray(image_to_canny)
        elif structure == "pose":
            controlnet = SD3ControlNetModel.from_pretrained(
                "InstantX/SD3-Controlnet-Pose",
                cache_dir=MODEL_CACHE,
                force_download=False,
                local_files_only=True,
            )
            # Pose preprocessing (assuming the input image is already a pose image)
            control_image = input_image
        elif structure == "tile":
            controlnet = SD3ControlNetModel.from_pretrained(
                "InstantX/SD3-Controlnet-Tile",
                cache_dir=MODEL_CACHE,
                force_download=False,
                local_files_only=True,
            )
            # Tile preprocessing (assuming the input image is already a tile image)
            control_image = input_image
        else:
            raise ValueError(f"Unsupported structure: {structure}")

        pipe = StableDiffusion3ControlNetPipeline.from_pretrained(
            "stabilityai/stable-diffusion-3-medium-diffusers",
            controlnet=controlnet,
            cache_dir=MODEL_CACHE,
            force_download=False,
            local_files_only=True,
        ).to(DEVICE, DTYPE)

        # Get width and height from aspect ratio
        width, height = self.aspect_ratio_to_width_height(aspect_ratio)

        # infer
        images = pipe(
            prompt=[prompt] * num_outputs,
            negative_prompt=[n_prompt] * num_outputs,
            control_image=control_image,
            controlnet_conditioning_scale=control_weight,
            num_inference_steps=inference_steps,
            guidance_scale=guidance_scale,
            generator=self.generator,
            width=width,
            height=height,
        ).images

        if not disable_safety_checker:
            has_nsfw_content = self.run_safety_checker(images)
        else:
            has_nsfw_content = [False] * len(images)

        output_paths = []
        for i, image in enumerate(images):
            if has_nsfw_content[i]:
                print(f"NSFW content detected in image {i}. Skipping...")
                continue

            image_redim, w, h = resize_image(
                str(image_in), f"resized_input_{i}.jpg", 1024
            )
            image = image.resize((w, h), Image.LANCZOS)

            # Save the image with the specified format and quality
            image = image.convert("RGB")
            extension = output_format.lower()
            extension = "jpeg" if extension == "jpg" else extension
            output_path = f"output_{i}.{extension}"

            print(f"[~] Saving to {output_path}...")
            print(f"[~] Output format: {extension.upper()}")
            if output_format != "png":
                print(f"[~] Output quality: {output_quality}")

            save_params = {"format": extension.upper()}
            if output_format != "png":
                save_params["quality"] = output_quality
                save_params["optimize"] = True

            image.save(output_path, **save_params)
            output_paths.append(Path(output_path))

        if len(output_paths) == 0:
            raise Exception(
                "NSFW content detected in all generated images. Try running it again, or try a different prompt."
            )

        return output_paths
