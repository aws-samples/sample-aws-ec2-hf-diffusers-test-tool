import os
import time
import argparse
import uuid
import csv
from PIL import Image, ImageOps
import torch
from diffusers.utils import load_image
from urllib.parse import urlparse
from diffusers import GGUFQuantizationConfig

def main():
    parser = argparse.ArgumentParser(
        description="Run Qwen Image Edit with customizable parameters."
    )

    # ---------------- CLI ARGUMENTS ---------------- #
    parser.add_argument("--pipeline_vendor", type=str, required=True,
                        choices=["QwenImageEditPlusPipeline", "FluxKontextPipeline"],
                        help="Model vendor.")
    parser.add_argument("--model_path", type=str,
                        help="Model path or repo ID.")
    parser.add_argument("--model_type", type=str, default="vanilla",
                        choices=["vanilla", "GGUF"],
                        help="Choose model type: 'vanilla' or 'GGUF'.")
    parser.add_argument("--image_input", type=str,
                        default="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/cat.png",
                        help="Path or URL to input image.")
    parser.add_argument("--image_output_dir", type=str, default="output_images",
                        help="Directory to save output images.")
    parser.add_argument("--prompt_positive", type=str,
                        default="replace the cat with a dalmatian",
                        help="Positive prompt (what to add/replace).")
    parser.add_argument("--prompt_negative", type=str, default="",
                        help="Negative prompt (what to avoid).")
    parser.add_argument("--torch_seed", type=int, default=0,
                        help="Random seed for reproducibility.")
    parser.add_argument("--cfg", type=float, default=4.0,
                        help="Classifier-free guidance scale.")
    parser.add_argument("--steps", type=int, default=10,
                        help="Number of inference steps.")
    parser.add_argument("--height", type=int, default=1024,
                        help="Output image height.")
    parser.add_argument("--width", type=int, default=1024,
                        help="Output image width.")
    parser.add_argument("--device", type=str, default="cuda",
                        choices=["cpu", "cuda"],
                        help="Device to run on (cpu or cuda).")
    # -------------------------------------------------- #

    args = parser.parse_args()


    # ---- Create output directories ---- #
    os.makedirs(args.image_output_dir, exist_ok=True)
    os.makedirs("output_metrics", exist_ok=True)

    # ---- Generate shared UUID ---- #
    run_uuid = str(uuid.uuid4())
    image_output_path = os.path.join(args.image_output_dir, f"{run_uuid}.png")
    csv_output_path = os.path.join("output_metrics", f"{run_uuid}.csv")

    # ---- Load model based on type ---- #
    if args.pipeline_vendor == "QwenImageEditPlusPipeline":
        print("Building QwenImageEditPlusPipeline.")
        
        from diffusers import QwenImageEditPlusPipeline, QwenImageTransformer2DModel
        # ---- Set model_path default depending on model_type ---- #
        if not args.model_path:
            if args.model_type == "GGUF":
                args.model_path = "https://huggingface.co/calcuis/qwen-image-edit-plus-gguf/blob/main/qwen-image-edit-plus-v2-iq4_nl.gguf"
            else:
                args.model_path = "Qwen/Qwen-Image-Edit-2509"
        
        if args.model_type == "vanilla":
            print("Loading VANILLA model.")
            pipeline = QwenImageEditPlusPipeline.from_pretrained(args.model_path, torch_dtype=torch.bfloat16)
        elif args.model_type == "GGUF":
            print("Loading GGUF model.")
            transformer = QwenImageTransformer2DModel.from_single_file(
                args.model_path,
                quantization_config=GGUFQuantizationConfig(compute_dtype=torch.bfloat16),
                torch_dtype=torch.bfloat16,
                config="callgg/image-edit-plus",
                subfolder="transformer"
            )
            pipeline = QwenImageEditPlusPipeline.from_pretrained(
                "Qwen/Qwen-Image-Edit-2509",
                transformer=transformer,
                torch_dtype=torch.bfloat16
            )
    elif args.pipeline_vendor =="FluxKontextPipeline":
        print("Building FluxKontextPipeline.")
        from diffusers import FluxKontextPipeline, FluxTransformer2DModel
        
        if not args.model_path:
            if args.model_type == "GGUF":
                args.model_path = "https://huggingface.co/calcuis/kontext-gguf/blob/main/flux-kontext-lite-q8_0.gguf"
            else:
                args.model_path = "black-forest-labs/FLUX.1-Kontext-dev"

        if args.model_type == "vanilla":
            print("Loading VANILLA model...")
            pipeline = FluxKontextPipeline.from_pretrained(args.model_path, torch_dtype=torch.bfloat16)
        
        elif args.model_type == "GGUF":
            transformer = FluxTransformer2DModel.from_single_file(
                args.model_path,
                quantization_config=GGUFQuantizationConfig(compute_dtype=torch.bfloat16),
                torch_dtype=torch.bfloat16,
                config="black-forest-labs/FLUX.1-Kontext-dev",
                subfolder="transformer"
            )
            pipeline = FluxKontextPipeline.from_pretrained(
                "black-forest-labs/FLUX.1-Kontext-dev", 
                transformer=transformer, 
                torch_dtype=torch.bfloat16
                )

    
    pipeline.set_progress_bar_config(disable=None)
    pipeline.to(args.device)


    # ---- Load and pad input image ---- #
    img_padded = ImageOps.pad(
        load_image(args.image_input),
        (args.height, args.width),
        color=(0, 0, 0),
        centering=(0.5, 0.5)
    )
    # ---- Step timing callback ---- #
    # step_times = []
    step_metrics = []
    last_time = [None]
    def callback_on_step_end(pipe, step_index, timestep, callback_kwargs):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        now = time.time()
        if last_time[0] is not None:
            # step_times.append(now - last_time[0])
            elapsed = now - last_time[0]
            timestamp = time.time()
            step_metrics.append((step_index, timestamp, elapsed))
        last_time[0] = now
        return callback_kwargs

    # ---- Prepare inputs ---- #
    inputs = {
        "image": [img_padded],
        "prompt": args.prompt_positive,
        "generator": torch.manual_seed(args.torch_seed),
        "true_cfg_scale": args.cfg,
        "negative_prompt": args.prompt_negative,
        "num_inference_steps": args.steps,
        "callback_on_step_end": callback_on_step_end,
        "callback_on_step_end_tensor_inputs": [],
        "height": args.height,
        "width": args.width
    }

    # ---- Run inference ---- #
    with torch.inference_mode():
        output = pipeline(**inputs)

    # ---- Save output image ---- #
    output_image = output.images[0]
    output_image.save(image_output_path)
    print(f"Image saved to: {image_output_path}")

    # ---- Save step timing results ---- #
    with open(csv_output_path, mode="w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        # include run info and per-step data
        writer.writerow(["run_id", "model_type", "model_path", "step", "timestamp", "time_sec"])
        for step_index, timestamp, elapsed in step_metrics:
            writer.writerow([
                run_uuid,
                args.model_type,
                args.model_path,
                step_index,
                timestamp,
                f"{elapsed:.6f}"
            ])
    print(f"Metrics saved to: {csv_output_path}")

    # ---- Print timings ---- #
    print("\nStep timings:")
    print("run_id, model_type, model_path, step, timestamp, time_sec")
    for step_index, timestamp, elapsed in step_metrics:
        print(f"{run_uuid},{args.model_type},{args.model_path},{step_index},{timestamp},{elapsed:.6f}")


if __name__ == "__main__":
    main()
