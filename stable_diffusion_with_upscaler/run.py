import sys

sys.path.extend(["./taming-transformers", "./stable-diffusion", "./latent-diffusion"])
import time
from typing import Any

import click
import k_diffusion as K
import ldm
import numpy as np
import torch
from pytorch_lightning import seed_everything
from torchvision.transforms import functional as TF

import stable_diffusion_with_upscaler.fetch_models as fetch_models
from stable_diffusion_with_upscaler.nn_modules import (
    CFGUpscaler,
    CLIPEmbedder,
    CLIPTokenizerTransform,
)
from stable_diffusion_with_upscaler.save_files import save_image


@torch.no_grad()
def condition_up(prompts: list[str], device: str):
    tok_up = CLIPTokenizerTransform()
    text_encoder_up = CLIPEmbedder(device=device)
    return text_encoder_up(tok_up(prompts))


def do_sample(
    sampler: str,
    model_wrap: CFGUpscaler,
    steps: int,
    noise: torch.Tensor,
    tol_scale: float,
    device: str,
    eta: float,
    extra_args: dict[str, Any],
):
    # Noise levels from stable diffusion.
    sigma_min, sigma_max = 0.029167532920837402, 14.614642143249512
    # We take log-linear steps in noise-level from sigma_max to sigma_min, using one of the k diffusion samplers.
    sigmas = (
        torch.linspace(np.log(sigma_max), np.log(sigma_min), steps + 1).exp().to(device)
    )
    if sampler == "k_euler":
        return K.sampling.sample_euler(
            model_wrap, noise * sigma_max, sigmas, extra_args=extra_args
        )
    elif sampler == "k_euler_ancestral":
        return K.sampling.sample_euler_ancestral(
            model_wrap, noise * sigma_max, sigmas, extra_args=extra_args, eta=eta
        )
    elif sampler == "k_dpm_2_ancestral":
        return K.sampling.sample_dpm_2_ancestral(
            model_wrap, noise * sigma_max, sigmas, extra_args=extra_args, eta=eta
        )
    elif sampler == "k_dpm_fast":
        return K.sampling.sample_dpm_fast(
            model_wrap,
            noise * sigma_max,
            sigma_min,
            sigma_max,
            steps,
            extra_args=extra_args,
            eta=eta,
        )
    elif sampler == "k_dpm_adaptive":
        sampler_opts = dict(
            s_noise=1.0,
            rtol=tol_scale * 0.05,
            atol=tol_scale / 127.5,
            pcoeff=0.2,
            icoeff=0.4,
            dcoeff=0.0,
        )
        return K.sampling.sample_dpm_adaptive(
            model_wrap,
            noise * sigma_max,
            sigma_min,
            sigma_max,
            extra_args=extra_args,
            eta=eta,
            **sampler_opts,
        )


@click.command()
@click.option("--seed", default=0, help="Set seed to 0 to use the current time")
@click.option("--prompt", required=True, help="Set text prompt")
@click.option("--n_samples", default=1, type=int)
@click.option("--batch_size", default=1, type=int)
@click.option("--scale", default=5, type=int)
@click.option("--steps", default=20, type=int)
@click.option(
    "--eta",
    default=0.0,
    help="Amount of noise to add per step (0.0=deterministic). Used in all samplers except `k_euler`.",
)
@click.option(
    "--outdir",
    default="outputs",
    help="Location to save generated image",
)
@click.option(
    "--noise_aug_level",
    default=0,
    type=int,
    help="Add noise to the latent vectors before upscaling. This theoretically can make the model work better on out-of-distribution inputs, but mostly just seems to make it match the input less, so it's turned off by default.",
)
@click.option(
    "--noise_aug_type",
    default="gaussian",
    type=click.Choice(
        ["gaussian", "fake"],
        case_sensitive=False,
    ),
)
@click.option(
    "--sampler",
    default="k_dpm_adaptive",
    type=click.Choice(
        [
            "k_euler",
            "k_euler_ancestral",
            "k_dpm_2_ancestral",
            "k_dpm_fast",
            "k_dpm_adaptive",
        ],
        case_sensitive=False,
    ),
)
@click.option(
    "--tol_scale",
    default=0.25,
    type=float,
    help="For the `k_dpm_adaptive` sampler, which uses an adaptive solver with error tolerance tol_scale",
)
@torch.no_grad()
def main(
    seed: int,
    prompt: str,
    n_samples: int,
    batch_size: int,
    scale: float,
    steps: int,
    eta: float,
    outdir: str,
    noise_aug_level: int,
    noise_aug_type: str,
    sampler: str,
    tol_scale: float,
):
    if torch.cuda.is_available():
        click.echo("GPU is available")
    else:
        click.echo("GPU is not available")
    run_model(
        seed,
        prompt,
        n_samples,
        batch_size,
        scale,
        steps,
        eta,
        outdir,
        noise_aug_level,
        noise_aug_type,
        sampler,
        tol_scale,
    )


def run_model(
    seed: int,
    prompt: str,
    n_samples: int,
    batch_size: int,
    scale: float,
    steps: int,
    eta: float,
    outdir: str,
    noise_aug_level: int,
    noise_aug_type: str,
    sampler: str,
    tol_scale: float,
):
    timestamp = int(time.time())
    if not seed:
        click.echo("No seed was provided, using the current time.")
        seed = timestamp
    click.echo(f"Generating with seed={seed}")
    seed_everything(seed)

    device = torch.device("cuda")
    sd_model, model_up = load_model_on_gpu(device=device)
    low_res_latent = gen_low_res_latent(
        sd_model,
        batch_size=batch_size,
        prompt=prompt,
        steps=steps,
        guidance_scale=scale,
        eta=eta,
    )

    [_, C, H, W] = low_res_latent.shape
    uc = condition_up(batch_size * [""], device=device)
    c = condition_up(batch_size * [prompt], device=device)

    model_wrap = CFGUpscaler(model_up, uc, cond_scale=scale)
    low_res_sigma = torch.full([batch_size], noise_aug_level, device=device)
    x_shape = [batch_size, C, 2 * H, 2 * W]

    image_id = 0
    save_location = f"{outdir}/%T-%I-%P.png"
    for _ in range((n_samples - 1) // batch_size + 1):
        if noise_aug_type == "gaussian":
            latent_noised = low_res_latent + noise_aug_level * torch.randn_like(
                low_res_latent
            )
        elif noise_aug_type == "fake":
            latent_noised = low_res_latent * (noise_aug_level**2 + 1) ** 0.5
        extra_args = {"low_res": latent_noised, "low_res_sigma": low_res_sigma, "c": c}
        noise = torch.randn(x_shape, device=device)
        up_latents = do_sample(
            sampler=sampler,
            model_wrap=model_wrap,
            steps=steps,
            noise=noise,
            tol_scale=tol_scale,
            device=device,
            eta=eta,
            extra_args=extra_args,
        )

        pixels = sd_model.decode_first_stage(up_latents)
        pixels = pixels.add(1).div(2).clamp(0, 1)

        for j in range(pixels.shape[0]):
            img = TF.to_pil_image(pixels[j])
            save_image(
                img,
                save_location=save_location,
                timestamp=timestamp,
                index=image_id,
                prompt=prompt,
                seed=seed,
            )
            image_id += 1


def load_model_on_gpu(device: str) -> tuple:
    """Save, load and mount models on GPU."""
    sd_model_path = fetch_models.download_from_huggingface(
        "CompVis/stable-diffusion-v-1-4-original", "sd-v1-4.ckpt"
    )
    cpu = torch.device("cpu")

    sd_model = fetch_models.load_model_from_config(
        "stable-diffusion/configs/stable-diffusion/v1-inference.yaml",
        sd_model_path,
        cpu,
    )
    sd_model = sd_model.to(device)

    model_up = fetch_models.make_upscaler_model(
        fetch_models.fetch(
            "https://models.rivershavewings.workers.dev/config_laion_text_cond_latent_upscaler_2.json"
        ),
        fetch_models.fetch(
            "https://models.rivershavewings.workers.dev/laion_text_cond_latent_upscaler_2_1_00470000_slim.pth"
        ),
    )
    model_up = model_up.to(device)
    return sd_model, model_up


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
