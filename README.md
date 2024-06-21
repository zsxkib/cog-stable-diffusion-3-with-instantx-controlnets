# Stable Diffusion 3 with ControlNet 🎨

Generate images with Stable Diffusion 3 and ControlNet using Canny edges, pose estimation, and tiling. This model offers incredible photorealism, typography, and prompt understanding.

## 🚀 Quickstart

### Canny Edge Detection

```bash
cog predict \
  -i 'prompt="A detailed image of a hand holding a smartphone. The smartphone is made entirely of vivid, colorful ecclesiastical stained glass, with intricate designs typical of church windows. The stained glass elements should be translucent, allowing light to pass through and creating a vibrant, luminous effect. The hand should be realistic and gently cradling the smartphone, emphasizing the contrast between modern technology and traditional stained glass artistry. Background should be neutral to highlight the details of the stained glass smartphone."' \
  -i 'input_image="canny.jpg"' \
  -i 'output_format="jpg"' \
  -i 'control_weight=0.9' \
  -i 'guidance_scale=7' \
  -i 'output_quality=80' \
  -i 'inference_steps=25' \
  -i 'negative_prompt="NSFW, nude, naked, porn, ugly, helmet, astronaut helmet"'
```

### Pose Estimation

```bash 
cog predict \
  -i 'prompt="A stunning realistic dynamic shot of an astronaut with cowboy hat"' \
  -i 'structure="pose"' \  
  -i 'input_image="pose.jpg"' \
  -i 'output_format="jpg"' \
  -i 'control_weight=0.5' \
  -i 'guidance_scale=6' \
  -i 'output_quality=80' \ 
  -i 'inference_steps=25' \
  -i 'negative_prompt="NSFW, nude, naked, porn, ugly, helmet, astronaut helmet"'
```

### Tile Control

```bash
cog predict \
  -i 'prompt="vivid artistic style, women wearing a pinstripe suit"' \  
  -i 'structure="tile"' \
  -i 'input_image="tile.jpg"' \
  -i 'output_format="jpg"' \
  -i 'control_weight=0.3' \
  -i 'guidance_scale=7' \
  -i 'output_quality=80' \
  -i 'inference_steps=25' \  
  -i 'negative_prompt="NSFW, nude, naked, porn, ugly"'
```

## 📋 Requirements

- Python
- Cog

## 📖 Documentation

For more details on using the model, safety considerations, and the model architecture, check out the [full documentation](https://github.com/yourusername/yourrepository).

## 🙌 Acknowledgements

- [Stability AI](https://stability.ai/) for developing Stable Diffusion 3
- [InstantX Team](https://huggingface.co/InstantX) for training the ControlNet models

## 🐦 Connect 

Have questions or feedback? Follow me on Twitter [@zsakib_](https://twitter.com/zsakib_) and let's chat!