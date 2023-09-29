import os
import re


def clean_prompt(prompt: str) -> str:
    badchars = re.compile(r"[/\\]")
    prompt = badchars.sub("_", prompt)
    if len(prompt) > 100:
        prompt = prompt[:100] + "…"
    return prompt


def format_filename(save_location: str, timestamp, seed, index, prompt: str) -> str:
    return (
        save_location.replace("%T", f"{timestamp}")
        .replace("%S", f"{seed}")
        .replace("%I", f"{index:02}")
        .replace("%P", clean_prompt(prompt))
    )


def save_image(image, **kwargs):
    filename = format_filename(**kwargs)
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    image.save(filename)
