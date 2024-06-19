# Prediction interface for Cog ⚙️
# https://cog.run/python

import os
import time
import cv2
import torch
import mimetypes
import subprocess
import numpy as np
from PIL import Image
from diffusers.utils import load_image
from cog import BasePredictor, Input, Path
from diffusers.models import SD3ControlNetModel
from diffusers import StableDiffusion3ControlNetPipeline

mimetypes.add_type("image/webp", ".webp")

MODEL_CACHE = "checkpoints"
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HOME"] = MODEL_CACHE
os.environ["TORCH_HOME"] = MODEL_CACHE
os.environ["HF_DATASETS_CACHE"] = MODEL_CACHE
os.environ["TRANSFORMERS_CACHE"] = MODEL_CACHE
os.environ["HUGGINGFACE_HUB_CACHE"] = MODEL_CACHE

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
            "models--stabilityai--stable-diffusion-3-medium-diffusers.tar",
        ]

        for model_file in model_files:
            url = BASE_URL + model_file

            filename = url.split("/")[-1]
            dest_path = os.path.join(MODEL_CACHE, filename)
            if not os.path.exists(dest_path.replace(".tar", "")):
                download_weights(url, dest_path)

        controlnet_canny = SD3ControlNetModel.from_pretrained(
            "InstantX/SD3-Controlnet-Canny",
            cache_dir=MODEL_CACHE,
            force_download=True,
        )
        self.pipe = StableDiffusion3ControlNetPipeline.from_pretrained(
            "stabilityai/stable-diffusion-3-medium-diffusers",
            controlnet=controlnet_canny,
            cache_dir=MODEL_CACHE,
            force_download=True,
        ).to(DEVICE, DTYPE)

        self.previous_seed = None
        self.generator = torch.Generator(device=DEVICE)

    def predict(
        self,
        input_image: Path = Input(description="Input image"),
        prompt: str = Input(description="Prompt"),
        negative_prompt: str = Input(
            description="Negative prompt", default="NSFW, nude, naked, porn, ugly"
        ),
        inference_steps: int = Input(
            description="Inference steps", ge=1, le=50, default=25
        ),
        guidance_scale: float = Input(
            description="Guidance scale", ge=1.0, le=10.0, default=7.0
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
    ) -> Path:
        """Run a single prediction on the model"""
        image_in = input_image
        n_prompt = negative_prompt

        if seed is None:
            seed = int.from_bytes(os.urandom(2), "big")
        elif seed != self.previous_seed:
            self.generator.manual_seed(seed)
            self.previous_seed = seed

        print(f"Using seed: {seed}")

        # Canny preprocessing
        image_to_canny = load_image(str(image_in))
        image_to_canny = np.array(image_to_canny)
        image_to_canny = cv2.Canny(image_to_canny, 100, 200)
        image_to_canny = image_to_canny[:, :, None]
        image_to_canny = np.concatenate(
            [image_to_canny, image_to_canny, image_to_canny], axis=2
        )
        image_to_canny = Image.fromarray(image_to_canny)

        control_image = image_to_canny

        # infer
        image = self.pipe(
            prompt=prompt,
            negative_prompt=n_prompt,
            control_image=control_image,
            controlnet_conditioning_scale=control_weight,
            num_inference_steps=inference_steps,
            guidance_scale=guidance_scale,
            num_images_per_prompt=1,
            generator=self.generator,
        ).images[0]

        image_redim, w, h = resize_image(str(image_in), "resized_input.jpg", 1024)
        image = image.resize((w, h), Image.LANCZOS)

        # Save the image with the specified format and quality
        extension = output_format.lower()
        extension = "jpeg" if extension == "jpg" else extension
        output_path = f"output.{extension}"

        print(f"[~] Saving to {output_path}...")
        print(f"[~] Output format: {extension.upper()}")
        if output_format != "png":
            print(f"[~] Output quality: {output_quality}")

        save_params = {"format": extension.upper()}
        if output_format != "png":
            save_params["quality"] = output_quality
            save_params["optimize"] = True

        image.save(output_path, **save_params)

        return Path(output_path)
