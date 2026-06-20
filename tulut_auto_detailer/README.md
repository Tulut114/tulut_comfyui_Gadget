# Tulut Auto Detailer for ComfyUI

A suite of custom nodes for ComfyUI designed for automated local refinement and detailing of faces, hands, and feet. It leverages YOLOv8 models for precise object detection, supports hardware-level upscaling, intelligent resampling, and seamless blending with AnimaLLLite models.

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
```
*(Replace `<Your-ComfyUI-Root-Folder>` with the actual path where your ComfyUI is installed, such as `C:/ComfyUI_windows_portable/ComfyUI` or your custom installation root).*

### Required Model Filenames:
Ensure the downloaded models are named **exactly** as follows:
1. **Face Detection**: `face_yolov8m.pt`
2. **Hand Detection**: `hand_yolov8s.pt`
3. **Foot Detection**: `FootYolov8x_v20.pt`

---

## Installation

1. Open your terminal and navigate to your ComfyUI custom nodes directory:
   ```bash
   cd <Your-ComfyUI-Root-Folder>/custom_nodes/
   ```
2. Clone this repository into a folder named `tulut_detailer`:
   ```bash
   git clone <Your-Repository-URL> tulut_detailer
   ```
3. Navigate into the folder:
   ```bash
   cd tulut_detailer
   ```
4. Install the required dependencies:
   - **For ComfyUI Portable version**:
     ```bash
     ..\..\..\python_embeded\python.exe -m pip install -r requirements.txt
     ```
   - **For standard Python/Conda environments**:
     ```bash
     pip install -r requirements.txt
     ```
5. Restart ComfyUI.

## Parameters Guide
- **guide_size**: The target optimal latent resolution for the localized inpainting region (multiples of 8).
- **denoise**: Denoising strength for the detailer sampler. Lower values (e.g., 0.3 - 0.5) preserve original structure while adding detail.
- **lllite_name**: Choose your AnimaLLLite network model for enhanced localized structural control.
