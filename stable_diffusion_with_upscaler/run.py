import os
import sys

sys.path.extend(["./taming-transformers", "./stable-diffusion", "./latent-diffusion"])
import io
import time
from functools import partial
from subprocess import Popen

import click
import k_diffusion as K
import ldm
import numpy as np
import requests
import torch
import torch.nn.functional as F
from einops import rearrange
from omegaconf import OmegaConf
from PIL import Image
from pytorch_lightning import seed_everything
from torch import nn
from torchvision.transforms import functional as TF
from torchvision.utils import make_grid
from tqdm.notebook import tqdm, trange

import stable_diffusion_with_upscaler.fetch_models as fetch_models
from stable_diffusion_with_upscaler.nn_modules import (
    CLIPEmbedder,
    CLIPTokenizerTransform,
)


@click.command()
@click.option("--seed", default=0, help="Set seed to 0 to use the current time")
@click.option("--prompt", help="Set text prompt")
@click.option("--num_samples", default=1)
@click.option("--batch_size", default=1)
@click.option("--guidance_scale", default=5)
@click.option("--steps", default=20)
@click.option("--eta", default=0.0)
def main(
    seed: int,
    prompt: str,
    num_samples: int,
    batch_size: int,
    guidance_scale: float,
    steps: int,
    eta: float,
):
    sd_model, vae_model_840k, vae_model_560k, model_up = load_model_on_gpu()
    low_res_latent = gen_low_res_latent(
        sd_model,
        batch_size=batch_size,
        prompt=prompt,
        steps=steps,
        guidance_scale=guidance_scale,
        eta=eta,
    )


def download() -> tuple[str, str, str]:
    """
    Download models and save to cache.
    """
    sd_model_path = fetch_models.download_from_huggingface(
        "CompVis/stable-diffusion-v-1-4-original", "sd-v1-4.ckpt"
    )
    vae_840k_model_path = fetch_models.download_from_huggingface(
        "stabilityai/sd-vae-ft-mse-original", "vae-ft-mse-840000-ema-pruned.ckpt"
    )
    vae_560k_model_path = fetch_models.download_from_huggingface(
        "stabilityai/sd-vae-ft-ema-original", "vae-ft-ema-560000-ema-pruned.ckpt"
    )
    return sd_model_path, vae_840k_model_path, vae_560k_model_path


def load_model_on_gpu() -> tuple:
    """Load and mount models on GPU."""
    sd_model_path, vae_840k_model_path, vae_560k_model_path = download()
    cpu = torch.device("cpu")
    device = torch.device("cuda")

    sd_model = fetch_models.load_model_from_config(
        "stable-diffusion/configs/stable-diffusion/v1-inference.yaml",
        sd_model_path,
        cpu,
    )
    vae_model_840k = fetch_models.load_model_from_config(
        "latent-diffusion/models/first_stage_models/kl-f8/config.yaml",
        vae_840k_model_path,
        cpu,
    )
    vae_model_560k = fetch_models.load_model_from_config(
        "latent-diffusion/models/first_stage_models/kl-f8/config.yaml",
        vae_560k_model_path,
        cpu,
    )

    sd_model = sd_model.to(device)
    vae_model_840k = vae_model_840k.to(device)
    vae_model_560k = vae_model_560k.to(device)

    model_up = fetch_models.make_upscaler_model(
        fetch_models.fetch(
            "https://models.rivershavewings.workers.dev/config_laion_text_cond_latent_upscaler_2.json"
        ),
        fetch_models.fetch(
            "https://models.rivershavewings.workers.dev/laion_text_cond_latent_upscaler_2_1_00470000_slim.pth"
        ),
    )
    model_up = model_up.to(device)
    return sd_model, vae_model_840k, vae_model_560k, model_up


@torch.no_grad()
def gen_low_res_latent(
    sd_model,
    *,
    batch_size,
    prompt,
    steps,
    guidance_scale,
    eta,
    SD_C=4,
    H=512,
    W=512,
    SD_F=8,
):
    with sd_model.ema_scope():
        uc = sd_model.get_learned_conditioning(batch_size * [""])
        c = sd_model.get_learned_conditioning(batch_size * [prompt])
        shape = [SD_C, H // SD_F, W // SD_F]
        test_sampler = ldm.models.diffusion.ddim.DDIMSampler(sd_model)

        samples_ddim, _ = test_sampler.sample(
            S=steps,
            conditioning=c,
            batch_size=batch_size,
            shape=shape,
            verbose=False,
            unconditional_guidance_scale=guidance_scale,
            unconditional_conditioning=uc,
            eta=eta,
        )
        return samples_ddim


if __name__ == "__main__":
    main()
