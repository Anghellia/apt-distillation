from logger import logger, log_tensor_sizes, log_gpu_memory_usage
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import os
import torch.nn.functional as F
from omegaconf import OmegaConf
from accelerate import Accelerator
import argparse
from torchvision import transforms

from PIL import Image
from micro_dit.model import create_latent_diffusion
import gc
import sys
import torch.random
from torch.cuda.amp import GradScaler, autocast

from micro_dit.utils import DATA_TYPES
from micro_dit.discriminator import EMA


CKPT_PATH = "/workspace/micro_diffusion/ckpts/dit_4_channel_37M_real_and_synthetic_data.pt"


def load_original_model(device, ckpt_path: str = CKPT_PATH):
    logger.debug(f"Loading micro-DiT model...")
    if not os.path.exists(ckpt_path):
        logger.error(f"Checkpoint not found: {ckpt_path}")
        sys.exit(1)

    checkpoint_data = torch.load(ckpt_path, map_location=device)
    model = create_latent_diffusion(latent_res=64, in_channels=4, pos_interp_scale=2.0).to(device)
    model.dit.load_state_dict(checkpoint_data)

    logger.debug("Model loaded successfully.")
    return model, checkpoint_data


def run_consistency_distillation(config, train_dataloader, output_dir, device, accelerator):
    os.makedirs(output_dir, exist_ok=True)

    torch.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)

    student_model, checkpoint_data = load_original_model(device=device)

    optimizer = optim.AdamW(student_model.parameters(), lr=config.learning_rate,
        weight_decay=0.01, betas=(0.9, 0.999), eps=1e-8
    )

    student_model, optimizer, train_dataloader = accelerator.prepare(
        student_model, optimizer, train_dataloader
    )
    student_model.train()

    # Setup EMA
    ema_model = create_latent_diffusion(latent_res=64, in_channels=4, pos_interp_scale=2.0).to(device)
    ema_model.dit.load_state_dict(checkpoint_data)
    ema_model.eval()
    ema_model.requires_grad_(False)
    ema_model = ema_model.cpu()

    scaler = GradScaler()

    global_step = 0

    for epoch in range(config.num_epochs):
        logger.debug(f"Starting epoch {epoch+1}/{config.num_epochs}")
        epoch_loss = 0.0

        optimizer.zero_grad(set_to_none=True)

        for batch_idx, batch in enumerate(tqdm(train_dataloader)):

            step_output = training_step(
                batch=batch,
                student_model=student_model,
                config=config,
                device=device,
                scaler=scaler,
                optimizer=optimizer
            )

            batch_loss = step_output["loss"]
            epoch_loss += batch_loss

            #torch.nn.utils.clip_grad_norm_(student_model.parameters(), max_norm=1.0)

            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

            global_step += 1

            unwrapped_model = accelerator.unwrap_model(student_model)
            update_ema_model(ema_model, unwrapped_model, config.ema_decay)

            # Save checkpoint
            if global_step % config.save_interval == 0:
                checkpoint_dir = os.path.join(output_dir, f"checkpoint_{global_step}")
                os.makedirs(checkpoint_dir, exist_ok=True)

                logger.debug(f"Saving checkpoint at step {global_step}")
                if hasattr(accelerator, "save_state"):
                    accelerator.save_state(checkpoint_dir)
                else:
                    # Manual save as fallback - save unwrapped model
                    if accelerator.is_main_process:
                        unwrapped_model = accelerator.unwrap_model(student_model)
                        torch.save({
                            'model': unwrapped_model.state_dict(),
                            'optimizer': optimizer.state_dict(),
                            'scaler': scaler.state_dict() if scaler is not None else None,
                            'step': global_step,
                            'epoch': epoch,
                        }, os.path.join(checkpoint_dir, "pytorch_model.bin"))

                # Also save EMA model separately
                if accelerator.is_main_process:
                    ema_checkpoint_path = f"{output_dir}/ema_model_step_{global_step}.pt"
                    logger.debug(f"Saving EMA model to {ema_checkpoint_path}")
                    torch.save(ema_model.state_dict(), ema_checkpoint_path)

        # End of epoch processing
        avg_epoch_loss = epoch_loss / len(train_dataloader)
        logger.debug(f"Epoch {epoch+1} completed with average loss: {avg_epoch_loss}")

        # Save epoch checkpoint
        epoch_checkpoint_dir = os.path.join(output_dir, f"checkpoint_epoch_{epoch+1}")
        os.makedirs(epoch_checkpoint_dir, exist_ok=True)

        logger.debug(f"Saving checkpoint for epoch {epoch+1}")
        if hasattr(accelerator, "save_state"):
            accelerator.save_state(epoch_checkpoint_dir)
        else:
            if accelerator.is_main_process:
                unwrapped_model = accelerator.unwrap_model(student_model)
                torch.save({
                    'model': unwrapped_model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scaler': scaler.state_dict() if scaler is not None else None,
                    'step': global_step,
                    'epoch': epoch,
                }, os.path.join(epoch_checkpoint_dir, "pytorch_model.bin"))

        # Save EMA model for this epoch
        if accelerator.is_main_process:
            checkpoint_path = f"{output_dir}/ema_model_epoch_{epoch+1}.pt"
            logger.debug(f"Saving epoch EMA checkpoint to {checkpoint_path}")
            torch.save(ema_model.state_dict(), checkpoint_path)
            logger.debug(f"Epoch EMA checkpoint saved")

    # Save final model
    if accelerator.is_main_process:
        final_path = f"{output_dir}/ema_model_final.pt"
        logger.debug(f"Saving final EMA model to {final_path}")
        torch.save(ema_model.state_dict(), final_path)
        logger.debug("Final EMA model saved")

    logger.debug("Training completed successfully")
    return ema_model

def training_step(batch, student_model, config, device, scaler, optimizer):
    image, caption, v_teacher = batch
    image = image.to(device)
    v_teacher = v_teacher.to(device)

    image_vae = student_model.vae.encode(image.to(DATA_TYPES[student_model.dtype]))['latent_dist'].sample().data
    image_vae *= student_model.latent_scale

    noise = torch.randn_like(image_vae)

    v_teacher_latents = student_model.vae.encode(v_teacher.to(DATA_TYPES[student_model.dtype]))['latent_dist'].sample().data
    v_teacher_latents *= student_model.latent_scale

    out = student_model.tokenizer.tokenize(caption)
    tokenized_prompts = out['input_ids'].to(device)
    conditioning = student_model.text_encoder.encode(tokenized_prompts)[0]

    T = torch.ones(noise.shape[0], device=device) * 0.002 # config.num_train_timesteps

    with autocast():
        try:
            v_student_latents = student_model(noise, conditioning, T)
            #print(f"Predicted: {v_student_latents} \nTarget: {v_teacher_latents}")
            loss = F.mse_loss(v_student_latents, v_teacher_latents)
            logger.debug(f"Calculated loss: {loss}")

        except Exception as e:
            logger.error(f"Error in forward pass: {str(e)}")
            raise

    scaler.scale(loss).backward()

    return {
        "loss": loss
    }


def update_ema_model(ema_model, model, decay):
    with torch.no_grad():
        for target, source in zip(ema_model.parameters(), model.parameters()):
            cpu_param = source.detach().cpu()
            target.data.mul_(decay).add_(cpu_param, alpha=1 - decay)
    torch.cuda.empty_cache()

def resize_to_multiple_of_32(image):
    w, h = image.size
    new_w = (w // 32) * 32
    new_h = (h // 32) * 32
    new_w = max(32, new_w)
    new_h = max(32, new_h)
    return image.resize((new_w, new_h), Image.LANCZOS)


class Config:
    def __init__(self):
        self.num_train_timesteps = 1000

        # Training configuration
        self.num_epochs = 10
        self.save_interval = 350
        self.seed = 42
        self.learning_rate = 8e-5
        self.ema_decay = 0.995


class TextImageDataset(Dataset):
    def __init__(self, dataset_path: str, img_size: int = 512):
        super().__init__()
        self.img_dir = os.path.join(dataset_path, "image")
        self.cap_dir = os.path.join(dataset_path, "caption")
        self.vt_dir  = os.path.join(dataset_path, "v_teacher")

        all_imgs = []
        for f in os.listdir(self.img_dir):
            if f.lower().endswith(('.png', '.jpg', '.jpeg')) and os.path.exists(os.path.join(self.vt_dir, f.replace("jpg", "png"))):
                all_imgs.append(f)
        self.images = sorted(all_imgs)

        self.img_size = img_size

        self.transform = transforms.Compose([
            transforms.Resize(self.img_size, interpolation=transforms.InterpolationMode.LANCZOS),
            transforms.Lambda(resize_to_multiple_of_32),
            transforms.CenterCrop((512, 512)),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        base_name = os.path.splitext(img_name)[0]

        img_path = os.path.join(self.img_dir, img_name)
        img = Image.open(img_path).convert('RGB')
        img = self.transform(img)

        cap_path = os.path.join(self.cap_dir, base_name + '.txt')
        with open(cap_path, 'r', encoding='utf-8') as f:
            caption_str = f.read().strip()

        vt_path = os.path.join(self.vt_dir, img_name.replace("jpg", "png"))
        v_teacher = Image.open(vt_path).convert('RGB')
        v_teacher = self.transform(v_teacher)

        return img, caption_str, v_teacher


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Consistency distillation training")
    parser.add_argument("--dataset_dir", type=str, default="./dataset_path", help="Dataset directory with image, text pairs")
    parser.add_argument("--output_dir", type=str, default="./consistent_distilled_model_5", help="Output directory for checkpoints")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    args = parser.parse_args()

    # Create config object
    config = Config()

    accelerator = Accelerator(mixed_precision="bf16")
    device = accelerator.device

    args.dataset_dir = "/workspace/micro_diffusion/micro_diffusion/datasets/prepare/jdb/output_jdb"
    train_dataset = TextImageDataset(dataset_path=args.dataset_dir)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=1, pin_memory=True)

    student_model = run_consistency_distillation(
        config=config,
        train_dataloader=train_dataloader,
        output_dir=args.output_dir,
        device=device,
        accelerator=accelerator
    )
