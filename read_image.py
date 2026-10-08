"""
read_image.py - Let an AI model (Claude) read and describe an image.

Setup:
    pip install anthropic
    export ANTHROPIC_API_KEY="your-key-here"   # Windows: set ANTHROPIC_API_KEY=your-key-here

Usage:
    python read_image.py photo.jpg
    python read_image.py photo.png "Extract all text from this image"
    python read_image.py https://example.com/image.jpg "What is in this picture?"
"""

import base64
import mimetypes
import sys

import anthropic

MODEL = "claude-sonnet-5-5"
SUPPORTED_TYPES = {"image/jpeg", "image/png", "image/gif", "image/webp"}


def build_image_block(source: str) -> dict:
    """Turn a local file path or a URL into an image content block."""
    if source.startswith(("http://", "https://")):
        return {"type": "image", "source": {"type": "url", "url": source}}

    media_type, _ = mimetypes.guess_type(source)
    if media_type not in SUPPORTED_TYPES:
        raise ValueError(
            f"Unsupported image type: {media_type}. Use JPEG, PNG, GIF or WebP."
        )

    with open(source, "rb") as f:
        data = base64.standard_b64encode(f.read()).decode("utf-8")

    return {
        "type": "image",
        "source": {"type": "base64", "media_type": media_type, "data": data},
    }


def read_image(source: str, prompt: str = "Describe this image in detail.") -> str:
    """Send an image plus a question to the AI and return its answer."""
    client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from the environment

    response = client.messages.create(
        model=MODEL,
        max_tokens=1024,
        messages=[
            {
                "role": "user",
                "content": [
                    build_image_block(source),
                    {"type": "text", "text": prompt},
                ],
            }
        ],
    )
    return response.content[0].text


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    image_source = sys.argv[1]
    user_prompt = sys.argv[2] if len(sys.argv) > 2 else "Describe this image in detail."
    print(read_image(image_source, user_prompt))
