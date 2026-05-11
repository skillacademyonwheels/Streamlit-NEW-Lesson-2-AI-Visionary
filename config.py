import os

from dotenv import load_dotenv

load_dotenv()

HF_API_KEY = os.getenv("HF_API_KEY", "")

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_VISION_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"

# You can change this model later if needed
HF_IMAGE_MODEL = "stabilityai/stable-diffusion-xl-base-1.0"
