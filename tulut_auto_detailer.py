import os
import torch
import torch.nn.functional as F
import numpy as np
import cv2
import folder_paths
import nodes
import traceback
from ultralytics import YOLO

tulut_yolo_path = os.path.join(folder_paths.models_dir, "ultralytics", "bbox")
if not os.path.exists(tulut_yolo_path):
    os.makedirs(tulut_yolo_path)

def get_bboxes(img_tensor, model_name):
    model_path = os.path.join(tulut_yolo_path, model_name)
    if not os.path.exists(model_path):
        print(f"[Tulut Detailer] Warning: YOLO model not found -> {model_path}")
        return []

    model = YOLO(model_path)
    img_np = (img_tensor[0].cpu().numpy() * 255).astype(np.uint8)
    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    results = model(img_bgr, verbose=False)
    
    bboxes = []
    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            bboxes.append((x1, y1, x2, y2))
    return bboxes

def apply_tulut_detailer(image, model, clip, vae, upscale_model, prompt, neg_prompt, steps, cfg, denoise, seed, yolo_model, guide_size, lllite_name):
    bboxes = get_bboxes(image, yolo_model)
    if not bboxes:
        return (image,)

    clip_encoder = nodes.CLIPTextEncode()
    vae_encoder = nodes.VAEEncode()
    sampler = nodes.KSampler()
    vae_decoder = nodes.VAEDecode()
    
    upscaler = nodes.NODE_CLASS_MAPPINGS["ImageUpscaleWithModel"]()
    AnimaLLLiteApply_class = nodes.NODE_CLASS_MAPPINGS.get("AnimaLLLiteApply")

    cond = clip_encoder.encode(clip, prompt)[0]
    uncond = clip_encoder.encode(clip, neg_prompt)[0]
    
    final_image = image.clone()
    B, H, W, C = final_image.shape
    
    for (x1, y1, x2, y2) in bboxes:
        w, h = x2 - x1, y2 - y1
        pad_x = int(w * 0.2)
        pad_y = int(h * 0.2)
        cx1, cy1 = max(0, x1 - pad_x), max(0, y1 - pad_y)
        cx2, cy2 = min(W, x2 + pad_x), min(H, y2 + pad_y)
        
        crop_w = cx2 - cx1
        crop_h = cy2 - cy1
        if crop_w < 16 or crop_h < 16:
            continue
            
        crop_img = final_image[:, cy1:cy2, cx1:cx2, :]
        
        upscaled_crop = upscaler.upscale(upscale_model, crop_img)[0]
        _, up_h, up_w, _ = upscaled_crop.shape
        
        scale = guide_size / max(crop_w, crop_h)
        target_w, target_h = int(crop_w * scale), int(crop_h * scale)
        target_w, target_h = (target_w // 8) * 8, (target_h // 8) * 8
        
        upscaled_crop_nchw = upscaled_crop.permute(0, 3, 1, 2)
        
        if up_w < target_w or up_h < target_h:
            highres_crop = F.interpolate(upscaled_crop_nchw, size=(target_h, target_w), mode='bicubic', align_corners=False).permute(0, 2, 3, 1)
        else:
            highres_crop = F.interpolate(upscaled_crop_nchw, size=(target_h, target_w), mode='area').permute(0, 2, 3, 1)
        
        mask_np = np.zeros((crop_h, crop_w), dtype=np.float32)
        ix1, iy1 = max(0, x1 - cx1), max(0, y1 - cy1)
        ix2, iy2 = min(crop_w, x2 - cx1), min(crop_h, y2 - cy1)
        mask_np[iy1:iy2, ix1:ix2] = 1.0 
        
        blur_amount = int(min(crop_w, crop_h) * 0.15)
        if blur_amount % 2 == 0: blur_amount += 1
        if blur_amount > 0:
            mask_np = cv2.GaussianBlur(mask_np, (blur_amount, blur_amount), 0)
        
        mask_tensor_base = torch.from_numpy(mask_np).unsqueeze(0).unsqueeze(0)
        highres_mask = F.interpolate(mask_tensor_base, size=(target_h, target_w), mode='bilinear', align_corners=False)
        highres_mask_out = highres_mask.squeeze(0)
        
        current_model = model
        if lllite_name != "none" and AnimaLLLiteApply_class is not None:
            try:
                lllite_node = AnimaLLLiteApply_class()
                func_name = getattr(lllite_node, "FUNCTION", "apply")
                lllite_func = getattr(lllite_node, func_name)
                
                lllite_result = lllite_func(
                    model=current_model, 
                    lllite_name=lllite_name, 
                    image=highres_crop, 
                    strength=1.0, 
                    start_percent=0.0, 
                    end_percent=1.0,
                    mask=highres_mask_out,       
                    preserve_wrapper=True
                )
                current_model = lllite_result[0]
            except Exception as e:
                print(f"[Tulut Detailer] LLLite injection skipped. Error details: {e}")
                traceback.print_exc()
        
        latent = vae_encoder.encode(vae, highres_crop)[0]
        latent = latent.copy()
        latent["noise_mask"] = highres_mask_out.to(image.device)
        
        sampled_latent = sampler.sample(
            model=current_model, seed=seed, steps=steps, cfg=cfg, 
            sampler_name="dpmpp_2m_sde", scheduler="sgm_uniform", 
            positive=cond, negative=uncond, latent_image=latent, denoise=denoise
        )[0]
        highres_result = vae_decoder.decode(vae, sampled_latent)[0]
        
        highres_result_nchw = highres_result.permute(0, 3, 1, 2)
        restored_crop = F.interpolate(highres_result_nchw, size=(crop_h, crop_w), mode='bicubic', align_corners=False).permute(0, 2, 3, 1)
            
        mask_tensor_final = torch.from_numpy(mask_np).unsqueeze(0).unsqueeze(-1).to(image.device)
        final_image[:, cy1:cy2, cx1:cx2, :] = restored_crop * mask_tensor_final + final_image[:, cy1:cy2, cx1:cx2, :] * (1.0 - mask_tensor_final)
        
    return (final_image,)

try:
    controlnet_list = ["none"] + folder_paths.get_filename_list("controlnet")
    anima_models = [m for m in controlnet_list if "anima" in m.lower() or m == "none"]
    if anima_models and len(anima_models) > 1:
        controlnet_list = anima_models
except:
    controlnet_list = ["none"]

class TulutFaceDetailer:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",), "model": ("MODEL",), "clip": ("CLIP",), "vae": ("VAE",),
                "upscale_model": ("UPSCALE_MODEL",),
                "lllite_name": (controlnet_list, ),
                "prompt": ("STRING", {"multiline": True, "default": ""}),
                "neg_prompt": ("STRING", {"multiline": True, "default": ""}),
                "guide_size": ("INT", {"default": 512, "min": 256, "max": 1024, "step": 64}),
                "steps": ("INT", {"default": 35, "min": 1, "max": 100, "step": 1}),
                "cfg": ("FLOAT", {"default": 4.5, "min": 1.0, "max": 12.0, "step": 0.1}),
                "denoise": ("FLOAT", {"default": 0.45, "min": 0.01, "max": 1.0, "step": 0.01}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            }
        }
    
    RETURN_TYPES = ("IMAGE", "MODEL", "CLIP", "VAE", "UPSCALE_MODEL")
    RETURN_NAMES = ("IMAGE", "MODEL", "CLIP", "VAE", "UPSCALE_MODEL")
    FUNCTION = "execute"
    CATEGORY = "Tulut/Detailer"

    def execute(self, image, model, clip, vae, upscale_model, lllite_name, prompt, neg_prompt, guide_size, steps, cfg, denoise, seed):
        result_img = apply_tulut_detailer(image, model, clip, vae, upscale_model, prompt, neg_prompt, steps, cfg, denoise, seed, "face_yolov8m.pt", guide_size, lllite_name)[0]
        return (result_img, model, clip, vae, upscale_model)

class TulutHandDetailer:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",), "model": ("MODEL",), "clip": ("CLIP",), "vae": ("VAE",),
                "upscale_model": ("UPSCALE_MODEL",),
                "lllite_name": (controlnet_list, ),
                "prompt": ("STRING", {"multiline": True, "default": ""}),
                "neg_prompt": ("STRING", {"multiline": True, "default": ""}),
                "guide_size": ("INT", {"default": 512, "min": 256, "max": 1024, "step": 64}),
                "steps": ("INT", {"default": 35, "min": 1, "max": 100, "step": 1}),
                "cfg": ("FLOAT", {"default": 4.5, "min": 1.0, "max": 12.0, "step": 0.1}),
                "denoise": ("FLOAT", {"default": 0.50, "min": 0.01, "max": 1.0, "step": 0.01}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            }
        }
    
    RETURN_TYPES = ("IMAGE", "MODEL", "CLIP", "VAE", "UPSCALE_MODEL")
    RETURN_NAMES = ("IMAGE", "MODEL", "CLIP", "VAE", "UPSCALE_MODEL")
    FUNCTION = "execute"
    CATEGORY = "Tulut/Detailer"

    def execute(self, image, model, clip, vae, upscale_model, lllite_name, prompt, neg_prompt, guide_size, steps, cfg, denoise, seed):
        result_img = apply_tulut_detailer(image, model, clip, vae, upscale_model, prompt, neg_prompt, steps, cfg, denoise, seed, "hand_yolov8s.pt", guide_size, lllite_name)[0]
        return (result_img, model, clip, vae, upscale_model)

class TulutFootDetailer:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",), "model": ("MODEL",), "clip": ("CLIP",), "vae": ("VAE",),
                "upscale_model": ("UPSCALE_MODEL",),
                "lllite_name": (controlnet_list, ),
                "prompt": ("STRING", {"multiline": True, "default": ""}),
                "neg_prompt": ("STRING", {"multiline": True, "default": ""}),
                "guide_size": ("INT", {"default": 512, "min": 256, "max": 1024, "step": 64}),
                "steps": ("INT", {"default": 35, "min": 1, "max": 100, "step": 1}),
                "cfg": ("FLOAT", {"default": 4.5, "min": 1.0, "max": 12.0, "step": 0.1}),
                "denoise": ("FLOAT", {"default": 0.50, "min": 0.01, "max": 1.0, "step": 0.01}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            }
        }
        
    RETURN_TYPES = ("IMAGE", "MODEL", "CLIP", "VAE", "UPSCALE_MODEL")
    RETURN_NAMES = ("IMAGE", "MODEL", "CLIP", "VAE", "UPSCALE_MODEL")
    FUNCTION = "execute"
    CATEGORY = "Tulut/Detailer"

    def execute(self, image, model, clip, vae, upscale_model, lllite_name, prompt, neg_prompt, guide_size, steps, cfg, denoise, seed):
        result_img = apply_tulut_detailer(image, model, clip, vae, upscale_model, prompt, neg_prompt, steps, cfg, denoise, seed, "FootYolov8x_v20.pt", guide_size, lllite_name)[0]
        return (result_img, model, clip, vae, upscale_model)

NODE_CLASS_MAPPINGS = {
    "TulutFaceDetailer": TulutFaceDetailer,
    "TulutHandDetailer": TulutHandDetailer,
    "TulutFootDetailer": TulutFootDetailer
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "TulutFaceDetailer": "Tulut Face Detailer",
    "TulutHandDetailer": "Tulut Hand Detailer",
    "TulutFootDetailer": "Tulut Foot Detailer"
}