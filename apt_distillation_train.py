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
import time

import wandb

from PIL import Image
from micro_dit.model import create_latent_diffusion
import gc
import sys
import torch.random
from torch.cuda.amp import GradScaler, autocast

import argparse
from accelerate import Accelerator
from omegaconf import OmegaConf

from micro_dit.discriminator import APTDiscriminator, approximated_r1_loss, EMA
from micro_dit.utils import DATA_TYPES

wandb.init(project="APT_Training", name="APT_Distillation")

BASE_CKPT_PATH = "/workspace/micro_diffusion/ckpts/dit_4_channel_37M_real_and_synthetic_data.pt"
DISTILLED_CKPT_PATH = "/workspace/micro_diffusion/ckpts/dit_4_channel_37M_real_and_synthetic_data.pt"

def load_original_model(device, ckpt_path: str = BASE_CKPT_PATH):
    logger.debug(f"Loading micro-DiT model...")
    if not os.path.exists(ckpt_path):
        logger.error(f"Checkpoint not found: {ckpt_path}")
        sys.exit(1)

    checkpoint_data = torch.load(ckpt_path, map_location=device)
    model = create_latent_diffusion(latent_res=64, in_channels=4, pos_interp_scale=2.0).to(device)
    model.dit.load_state_dict(checkpoint_data)

    logger.debug("Model loaded successfully.")
    return model

def filter_and_strip_dit(state_dict):
    new_state_dict = {}

    for key, value in state_dict.items():
        if key.startswith("dit."):
            new_key = key[len("dit."):]
            new_state_dict[new_key] = value

    return new_state_dict

def load_distilled_model(device, ckpt_path: str = DISTILLED_CKPT_PATH):
    logger.debug(f"Loading distilled 1-step micro-DiT model...")
    if not os.path.exists(ckpt_path):
        logger.error(f"Checkpoint not found: {ckpt_path}")
        sys.exit(1)

    checkpoint_data = torch.load(ckpt_path, map_location=device)
    filtered_weights = filter_and_strip_dit(checkpoint_data)

    model = create_latent_diffusion(latent_res=64, in_channels=4, pos_interp_scale=2.0).to(device)
    current_state_dict = model.dit.state_dict()
    matched_weights = {k: v for k, v in filtered_weights.items() if k in current_state_dict}

    model.dit.load_state_dict(matched_weights)

    logger.debug("Model loaded successfully.")
    return model

def run_apt_distillation(config, train_dataloader, output_dir, device, accelerator):
    os.makedirs(output_dir, exist_ok=True)

    torch.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)

    original_model = load_original_model(device)
    generator = load_original_model(device)

    discriminator = APTDiscriminator(original_model)

    #generator.requires_grad_(True)
    #discriminator.requires_grad_(True)

    for param in generator.dit.parameters():
        param.requires_grad = True

    for param in discriminator.backbone.dit.parameters():
        param.requires_grad = True

    generator.train()
    discriminator.train()

    #print([p for p in generator.parameters() if p.requires_grad])
    #print([p for p in discriminator.parameters() if p.requires_grad])

    g_optimizer = optim.RMSprop([p for p in generator.dit.parameters() if p.requires_grad], lr=config.g_lr, alpha=0.9)
    d_optimizer = optim.RMSprop([p for p in discriminator.backbone.dit.parameters() if p.requires_grad], lr=config.d_lr, alpha=0.9)

    print([name for name, value in generator.named_parameters() if value.requires_grad])
    print([name for name, value in discriminator.named_parameters() if value.requires_grad])

    generator, discriminator, g_optimizer, d_optimizer, train_dataloader = accelerator.prepare(
        generator, discriminator, g_optimizer, d_optimizer, train_dataloader
    )

    # Setup EMA
    ema = EMA(accelerator.unwrap_model(generator), decay=config.ema_decay)

    global_step = 0

    for epoch in range(config.num_epochs):
        logger.debug(f"Starting epoch {epoch+1}/{config.num_epochs}")

        for batch_idx, batch in enumerate(tqdm(train_dataloader)):

            step_output = apt_training_step(
                global_step=global_step,
                batch=batch,
                config=config,
                device=device,
                accelerator=accelerator,
                generator=generator,
                discriminator=discriminator,
                d_optimizer=d_optimizer,
                g_optimizer=g_optimizer
            )
            g_loss = step_output["g_loss"]
            d_loss = step_output["d_loss"]

            if accelerator.sync_gradients:
                with accelerator.main_process_first():
                    ema.update(accelerator.unwrap_model(generator))

            if accelerator.is_main_process:
                print(f"[Image] Batch {batch_idx}, G Loss: {g_loss:.4f}, D Loss: {d_loss:.4f}")

            global_step += 1

            # Save checkpoint
            if global_step % config.save_interval == 0:
                checkpoint_dir = os.path.join(output_dir, f"checkpoint_{global_step}")
                os.makedirs(checkpoint_dir, exist_ok=True)

                logger.debug(f"Saving checkpoint at step {global_step}")
                unwrapped_generator = accelerator.unwrap_model(generator)
                unwrapped_discriminator = accelerator.unwrap_model(discriminator)
                torch.save({
                'generator': unwrapped_generator.state_dict(),
                'ema_generator': ema.model.state_dict(),
                'discriminator': unwrapped_discriminator.state_dict(),
                'g_optimizer': g_optimizer.state_dict(),
                'd_optimizer': d_optimizer.state_dict(),
                'step': global_step,
                'epoch': epoch
                }, os.path.join(checkpoint_dir, "pytorch_model.bin"))


                prompts = [
                    "a full body shot of a beautiful young goth woman with white hair and blue eyes with a subtle smile, double arm amputee, dsd, double-shoulder disarticulation, armless, no arms, wearing black leather corsette, intricate, high detail, 8k",
                    "desert landscape with dunes and hills, multi dimensional paper cut craft, persian ambience, an evil djinn emerging from sand, wind blowin, paper illustration, sand storm, ornate, detailed, golden ratio",
                    "miniature disney santa's village, christmas, in lights, under a christmas tree, photorealistic, mood lighting, tilt shift photography",
                    "cinematic, realistic, 1920s girls at college, girls studying, university, dark academia, ghost story, horror, haunted, creepy"
                    ]
                gen_images = generator.generate(
                    prompt=prompts,
                    num_inference_steps=1,
                    guidance_scale=7.5,
                    seed=2131
                )
                image_list = []
                for idx, img_tensor in enumerate(gen_images):
                    img_tensor = (img_tensor.clamp(0, 1) * 255).byte().cpu()
                    img_pil = Image.fromarray(img_tensor.permute(1, 2, 0).numpy())
                    image_list.append(wandb.Image(img_pil, caption=f"Prompt {idx+1}"))

                wandb.log({
                    "Generator Loss": g_loss,
                    "Discriminator Loss": d_loss,
                    "Generated Sample": image_list
                })

                for idx, img_pil in enumerate(image_list):
                    img_pil.image.save(f"{output_dir}/generated_sample_step_{global_step}_prompt_{idx+1}.png")

        # Save epoch checkpoint
        epoch_checkpoint_dir = os.path.join(output_dir, f"checkpoint_epoch_{epoch+1}")
        os.makedirs(epoch_checkpoint_dir, exist_ok=True)

        logger.debug(f"Saving checkpoint for epoch {epoch+1}")
        if accelerator.is_main_process:
            unwrapped_generator = accelerator.unwrap_model(generator)
            unwrapped_discriminator = accelerator.unwrap_model(discriminator)
            torch.save({
                'generator': unwrapped_generator.state_dict(),
                'ema_generator': ema.model.state_dict(),
                'discriminator': unwrapped_discriminator.state_dict(),
                'g_optimizer': g_optimizer.state_dict(),
                'd_optimizer': d_optimizer.state_dict(),
                'step': global_step,
                'epoch': epoch
                }, os.path.join(checkpoint_dir, "pytorch_model.bin"))

    # Save final model
    if accelerator.is_main_process:
        final_path = f"{output_dir}/ema_model_final.pt"
        logger.debug(f"Saving final EMA model to {final_path}")
        torch.save(ema_model.state_dict(), final_path)
        logger.debug("Final EMA model saved")

    logger.debug("Training completed successfully")
    return ema.model


def apt_training_step(global_step, batch, config, device, accelerator, generator, discriminator, d_optimizer, g_optimizer):
    image, caption = batch
    image = image.to(device)

    image_vae = generator.vae.encode(image.to(DATA_TYPES[generator.dtype]))['latent_dist'].sample().data
    image_vae *= generator.latent_scale

    noise = torch.randn_like(image_vae)

    out = generator.tokenizer.tokenize(caption)
    tokenized_prompts = out['input_ids']
    conditioning = generator.text_encoder.encode(tokenized_prompts.to(device))[0]


    rnd_normal = torch.randn([noise.shape[0], 1, 1, 1], device=device)
    sigma = (rnd_normal * generator.edm_config.P_std + generator.edm_config.P_mean).exp()

    # Train discriminator
    with torch.no_grad():
        fake_image = generator(image_vae, conditioning, device, sigma, final_step=True)

    real_logits = discriminator(image_vae, conditioning, device, sigma)
    fake_logits = discriminator(fake_image, conditioning, device, sigma)

    d_loss_real = -torch.log(torch.sigmoid(real_logits) + 1e-8).mean()
    d_loss_fake = -torch.log(1.0 - torch.sigmoid(fake_logits) + 1e-8).mean()
    d_loss = d_loss_real + d_loss_fake

    r1_loss = approximated_r1_loss(discriminator, image_vae, device, sigma, conditioning)
    d_loss = d_loss + config.lambda_r1 * r1_loss

    #if global_step % 2 == 0:
        # Update discriminator
    d_optimizer.zero_grad()
    accelerator.backward(d_loss, retain_graph=True)
    d_optimizer.step()

    # Train generator
    fake_image = generator(image_vae, conditioning, device, sigma, final_step=True)
    fake_logits = discriminator(fake_image, conditioning, device, sigma)
    g_loss = -torch.log(torch.sigmoid(fake_logits) + 1e-8).mean()

    g_optimizer.zero_grad()

    # Update generator
    accelerator.backward(g_loss)
    g_optimizer.step()

    logger.debug(f"Calculated g_loss: {g_loss}, d_loss: {d_loss}")

    return {
        "g_loss": g_loss.item(),
        "d_loss": d_loss.item()
    }


class Config:
    def __init__(self):
        self.num_train_timesteps = 1000

        # Training configuration
        self.num_epochs = 10
        self.save_interval = 200
        self.g_lr = 5 #5e-2
        self.d_lr = 5 #1e-2

        self.ema_decay = 0.995
        self.lambda_r1 = 100

        self.seed = 123213


def resize_to_multiple_of_32(image):
    w, h = image.size
    new_w = (w // 32) * 32
    new_h = (h // 32) * 32
    new_w = max(32, new_w)
    new_h = max(32, new_h)
    return image.resize((new_w, new_h), Image.LANCZOS)


class TextImageDataset(Dataset):
    def __init__(self, dataset_path: str, img_size: int = 512):
        super().__init__()
        self.img_dir = os.path.join(dataset_path, "image")
        self.cap_dir = os.path.join(dataset_path, "caption")

        all_imgs = []
        for f in os.listdir(self.img_dir):
            if f.lower().endswith(('.png', '.jpg', '.jpeg')) and os.path.exists(os.path.join(self.cap_dir, f.replace("jpg", "txt"))):
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

        return img, caption_str


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="APT distillation training")
    parser.add_argument("--dataset_dir", type=str, default="./dataset_path", help="Dataset directory with image, text pairs")
    parser.add_argument("--output_dir", type=str, default="./apt_final_model_exp2", help="Output directory for saving checkpoints")
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size")
    args = parser.parse_args()

    # Create config object
    config = Config()

    accelerator = Accelerator()
    device = accelerator.device

    args.dataset_dir = "/workspace/micro_diffusion/micro_diffusion/datasets/prepare/jdb/output_jdb"
    train_dataset = TextImageDataset(dataset_path=args.dataset_dir)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=1, pin_memory=True)

    final_model = run_apt_distillation(
        config=config,
        train_dataloader=train_dataloader,
        output_dir=args.output_dir,
        device=device,
        accelerator=accelerator
    )
