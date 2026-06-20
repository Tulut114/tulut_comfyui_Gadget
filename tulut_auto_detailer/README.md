# Tulut Auto Detailer for ComfyUI

A  suite of custom nodes for ComfyUI designed for automated local refinement and detailing of faces, hands, and feet. It leverages YOLOv8 models for precise object detection, supports hardware-level upscaling, intelligent resampling, and seamless blending with AnimaLLLite models.

## Features
- **Tulut Face Detailer**: Automatically detects and refines faces using `face_yolov8m.pt`.
- **Tulut Hand Detailer**: Automatically detects and refines hands using `hand_yolov8s.pt`.
- **Tulut Foot Detailer**: Automatically detects and refines feet using `FootYolov8x_v20.pt`.
- **AnimaLLLite Integration**: Supports precision localized network injection with region masking.
- **Hardware Upscaling & Dynamic Resampling**: Seamlessly incorporates upscale models with smart grid resampling based on the target guide size.

---

## Model Installation (Crucial)

Before using these nodes, you **must** download the required YOLO detection models and place them in the correct directory. The node will automatically create the subfolders if they do not exist.

### Target Directory Path:
Put your `.pt` files into your ComfyUI models directory under the following structure:
```text
<Your-ComfyUI-Root-Folder>/models/ultralytics/bbox/
