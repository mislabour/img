#!/usr/bin/env python3
import os
import sys
import random
import requests
import json
import asyncio
import logging
import pickle
import torch
import time
import threading
import uuid
from pathlib import Path
from datetime import datetime
from cryptography.fernet import Fernet

def check_and_install_packages():
    required_packages = [
        'diffusers', 'transformers', 'accelerate',
        'python-telegram-bot', 'instagrapi', 'requests', 'Pillow', 'cryptography'
    ]
    missing_packages = []
    for package in required_packages:
        try:
            if package == 'python-telegram-bot':
                __import__('telegram')
            elif package == 'Pillow':
                __import__('PIL')
            else:
                __import__(package.replace('-', '_'))
        except ImportError:
            missing_packages.append(package)
    if missing_packages:
        print(f"Missing packages: {', '.join(missing_packages)}")
        print("Please install them with:")
        print(f"pip install {' '.join(missing_packages)}")
        return False
    return True

if not check_and_install_packages():
    sys.exit(1)

from diffusers import DiffusionPipeline
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
from instagrapi import Client
from instagrapi.exceptions import LoginRequired, ClientError, PleaseWaitFewMinutes, ChallengeRequired
from PIL import Image, ImageEnhance, ImageFilter
import accelerate

logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO,
    handlers=[
        logging.FileHandler('autobot.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class Config:
    TELEGRAM_BOT_TOKEN = "7579821658:AAFXrSfOwlTQGjLx9O15NCDDsak5Yq1A89c"
    INSTAGRAM_USERNAME = None
    INSTAGRAM_PASSWORD = None
    ADMIN_CHAT_ID = None
    SESSION_FILE = "instagram_session.pkl"
    CREDENTIALS_FILE = "instagram_credentials.encrypted"
    ENCRYPTION_KEY_FILE = "encryption.key"
    IMAGES_DIR = "generated_images"
    IMAGE_HEIGHT = 1080
    IMAGE_WIDTH = 1080
    MIN_INTERVAL = 600   # 10 minutes
    MAX_INTERVAL = 900   # 15 minutes
    MAX_RETRIES = 4
    RETRY_DELAY = 45
    RATE_LIMIT_DELAY = 300
    MODEL_NAME = "runwayml/stable-diffusion-v1-5"

class CredentialManager:
    def __init__(self):
        self.key_file = Config.ENCRYPTION_KEY_FILE
        self.credentials_file = Config.CREDENTIALS_FILE
        self.encryption_key = self.load_or_create_key()

    def load_or_create_key(self):
        try:
            if os.path.exists(self.key_file):
                with open(self.key_file, 'rb') as f:
                    return f.read()
            else:
                key = Fernet.generate_key()
                with open(self.key_file, 'wb') as f:
                    f.write(key)
                print("Created new encryption key")
                return key
        except Exception as e:
            logger.error(f"Key management error: {e}")
            raise

    def encrypt_credentials(self, username, password):
        try:
            fernet = Fernet(self.encryption_key)
            credentials = {
                'username': username,
                'password': password,
                'timestamp': time.time(),
                'device_id': str(uuid.uuid4())
            }
            encrypted_data = fernet.encrypt(json.dumps(credentials).encode())
            with open(self.credentials_file, 'wb') as f:
                f.write(encrypted_data)
            Config.INSTAGRAM_USERNAME = username
            Config.INSTAGRAM_PASSWORD = password
            logger.info(f"Credentials encrypted and saved for: {username}")
            return True
        except Exception as e:
            logger.error(f"Failed to encrypt credentials: {e}")
            return False

    def decrypt_credentials(self):
        try:
            if not os.path.exists(self.credentials_file):
                return False
            fernet = Fernet(self.encryption_key)
            with open(self.credentials_file, 'rb') as f:
                encrypted_data = f.read()
            decrypted_data = fernet.decrypt(encrypted_data)
            credentials = json.loads(decrypted_data.decode())
            Config.INSTAGRAM_USERNAME = credentials['username']
            Config.INSTAGRAM_PASSWORD = credentials['password']
            logger.info(f"Credentials decrypted for: {credentials['username']}")
            return True
        except Exception as e:
            logger.error(f"Failed to decrypt credentials: {e}")
            return False

    def delete_credentials(self):
        try:
            if os.path.exists(self.credentials_file):
                os.remove(self.credentials_file)
            if os.path.exists(self.key_file):
                os.remove(self.key_file)
            Config.INSTAGRAM_USERNAME = None
            Config.INSTAGRAM_PASSWORD = None
            logger.info("Credentials deleted")
            return True
        except Exception as e:
            logger.error(f"Failed to delete credentials: {e}")
            return False

class WordManager:
    def __init__(self):
        self.art_styles = [
            "digital art", "concept art", "fantasy art", "abstract art", "surreal art",
            "cyberpunk", "steampunk", "art nouveau", "impressionist", "expressionist"
        ]
        self.descriptors = [
            "ethereal", "mystical", "vibrant", "luminous", "crystalline", "iridescent",
            "holographic", "neon", "golden", "silver", "cosmic", "dreamy", "magical",
            "ancient", "futuristic", "organic", "geometric", "flowing", "radiant"
        ]
        self.elements = [
            "forest", "ocean", "mountain", "desert", "city", "galaxy", "nebula",
            "waterfall", "storm", "aurora", "sunset", "moonlight", "starlight",
            "fire", "ice", "lightning", "crystal", "diamond", "pearl", "jade"
        ]
        self.qualities = [
            "beautiful", "stunning", "magnificent", "breathtaking", "mesmerizing",
            "peaceful", "dynamic", "intense", "serene", "powerful", "gentle",
            "mysterious", "enchanting", "captivating", "extraordinary"
        ]

    def generate_prompt(self, seed=None):
        if seed:
            random.seed(seed)
        style = random.choice(self.art_styles)
        descriptor1 = random.choice(self.descriptors)
        descriptor2 = random.choice(self.descriptors)
        element = random.choice(self.elements)
        quality1 = random.choice(self.qualities)
        quality2 = random.choice(self.qualities)
        prompt = f"{quality1} {descriptor1} {element} {descriptor2} {quality2}"
        enhancement = f", {style}, highly detailed, masterpiece, 8k, professional"
        full_prompt = prompt + enhancement
        logger.info(f"Generated prompt: {full_prompt}")
        return full_prompt

class AdvancedInstagramManager:
    def __init__(self):
        self.client = Client()
        self.session_file = Config.SESSION_FILE
        self.logged_in = False
        self.last_post_time = 0
        self.consecutive_failures = 0
        self.setup_client()

    def setup_client(self):
        try:
            device_id = f"android-{self.generate_device_id()}"
            phone_id = str(uuid.uuid4())
            self.client.set_user_agent("Instagram 275.0.0.27.98 Android")
            self.client.set_device({
                'phone_id': phone_id,
                'uuid': str(uuid.uuid4()),
                'client_session_id': str(uuid.uuid4()),
                'advertising_id': str(uuid.uuid4()),
                'device_id': device_id,
            })
            logger.info("Instagram client configured with unique device ID")
        except Exception as e:
            logger.warning(f"Client setup warning: {e}")

    def generate_device_id(self):
        chars = "0123456789abcdef"
        return ''.join(random.choice(chars) for _ in range(16))

    def login_with_comprehensive_retry(self):
        max_attempts = Config.MAX_RETRIES
        for attempt in range(max_attempts):
            try:
                logger.info(f"Instagram login attempt {attempt + 1}/{max_attempts}")
                if attempt > 0:
                    delay = Config.RETRY_DELAY * attempt
                    logger.info(f"Waiting {delay} seconds before retry...")
                    time.sleep(delay)
                if attempt == 0 and os.path.exists(self.session_file):
                    try:
                        with open(self.session_file, 'rb') as f:
                            session = pickle.load(f)
                            self.client.set_settings(session)
                        user_info = self.client.account_info()
                        logger.info(f"Loaded valid session for: {user_info.username}")
                        self.logged_in = True
                        self.consecutive_failures = 0
                        return True
                    except (LoginRequired, ClientError):
                        logger.info("Existing session invalid, creating new one")
                        if os.path.exists(self.session_file):
                            os.remove(self.session_file)
                if not Config.INSTAGRAM_USERNAME or not Config.INSTAGRAM_PASSWORD:
                    logger.error("No Instagram credentials available")
                    return False
                logger.info(f"Performing fresh login for: {Config.INSTAGRAM_USERNAME}")
                self.client.login(Config.INSTAGRAM_USERNAME, Config.INSTAGRAM_PASSWORD)
                with open(self.session_file, 'wb') as f:
                    pickle.dump(self.client.get_settings(), f)
                logger.info("Instagram login successful")
                self.logged_in = True
                self.consecutive_failures = 0
                return True
            except PleaseWaitFewMinutes as e:
                logger.warning(f"Rate limit hit: {e}")
                if attempt < max_attempts - 1:
                    time.sleep(Config.RATE_LIMIT_DELAY)
            except ChallengeRequired as e:
                logger.error(f"Challenge required: {e}")
                logger.error("Please login via Instagram app and complete any challenges")
                return False
            except ClientError as e:
                error_msg = str(e).lower()
                logger.error(f"Client error: {e}")
                if "checkpoint" in error_msg:
                    logger.error("Account checkpoint required. Login via Instagram app")
                    return False
                elif "challenge" in error_msg:
                    logger.error("Account challenge required. Login via Instagram app")
                    return False
                elif "feedback" in error_msg:
                    logger.error("Account restricted. Cannot login at this time")
                    return False
                elif "two_factor" in error_msg:
                    logger.error("Two-factor authentication required. Please disable or use app password")
                    return False
            except requests.RequestException as e:
                logger.error(f"Network error during Instagram login: {e}")
                time.sleep(Config.RETRY_DELAY)
            except Exception as e:
                logger.error(f"Login attempt {attempt + 1} failed: {e}")
        logger.error("All login attempts failed")
        self.logged_in = False
        self.consecutive_failures += 1
        return False

    def optimize_image_for_instagram(self, image_path):
        try:
            logger.info(f"Optimizing image: {os.path.basename(image_path)}")
            with Image.open(image_path) as img:
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                width, height = img.size
                logger.info(f"Original size: {width}x{height}")
                if width != height:
                    size = min(width, height)
                    left = (width - size) // 2
                    top = (height - size) // 2
                    img = img.crop((left, top, left + size, top + size))
                img = img.resize((1080, 1080), Image.Resampling.LANCZOS)
                img = img.filter(ImageFilter.UnsharpMask(radius=1, percent=110, threshold=3))
                enhancer = ImageEnhance.Color(img)
                img = enhancer.enhance(1.05)
                enhancer = ImageEnhance.Contrast(img)
                img = enhancer.enhance(1.02)
                optimized_path = image_path.replace('.png', '_instagram.jpg')
                img.save(
                    optimized_path,
                    'JPEG',
                    quality=95,
                    optimize=True,
                    progressive=True,
                    dpi=(72, 72)
                )
                logger.info(f"Image optimized: {os.path.basename(optimized_path)}")
                return optimized_path
        except Exception as e:
            logger.error(f"Image optimization failed: {e}")
            return image_path

    def post_with_advanced_retry(self, image_path, caption, seed):
        if not self.logged_in:
            if not self.login_with_comprehensive_retry():
                return False, "Login failed"
        current_time = time.time()
        time_since_last_post = current_time - self.last_post_time
        if time_since_last_post < 60:
            wait_time = 60 - time_since_last_post
            logger.info(f"Rate limiting: waiting {wait_time:.1f} seconds")
            time.sleep(wait_time)
        optimized_path = self.optimize_image_for_instagram(image_path)
        upload_methods = [
            self.upload_method_standard,
            self.upload_method_with_metadata,
            self.upload_method_basic
        ]
        last_error = None
        for method_idx, upload_method in enumerate(upload_methods):
            for attempt in range(Config.MAX_RETRIES):
                try:
                    logger.info(f"Upload method {method_idx + 1}, attempt {attempt + 1}")
                    if attempt > 0:
                        delay = Config.RETRY_DELAY * attempt
                        logger.info(f"Waiting {delay} seconds before retry...")
                        time.sleep(delay)
                    media = upload_method(optimized_path, caption)
                    if media:
                        logger.info(f"Successfully posted! Media ID: {media.pk}")
                        self.last_post_time = time.time()
                        self.consecutive_failures = 0
                        if optimized_path != image_path:
                            try:
                                os.remove(optimized_path)
                            except:
                                pass
                        return True, f"Posted successfully (Media ID: {media.pk})"
                except PleaseWaitFewMinutes as e:
                    logger.warning(f"Rate limit: {e}")
                    time.sleep(Config.RATE_LIMIT_DELAY)
                    last_error = str(e)
                except ClientError as e:
                    error_msg = str(e).lower()
                    logger.error(f"Client error: {e}")
                    last_error = str(e)
                    if "400" in error_msg:
                        if "photo" in error_msg or "media" in error_msg:
                            logger.info("Photo format error, trying next method")
                            break
                        elif "caption" in error_msg:
                            caption = caption[:1000]
                            logger.info("Trying with shorter caption")
                            continue
                    elif "feedback_required" in error_msg:
                        logger.error("Account restricted")
                        return False, "Account restricted"
                    elif "challenge" in error_msg or "checkpoint" in error_msg:
                        logger.error("Account verification required")
                        return False, "Account verification required"
                except LoginRequired:
                    logger.warning("Session expired during upload")
                    if self.login_with_comprehensive_retry():
                        continue
                    else:
                        logger.error("Re-login failed after session expiry.")
                        return False, "Session expired and re-login failed"
                except requests.RequestException as e:
                    logger.error(f"Network error during upload: {e}")
                    time.sleep(Config.RETRY_DELAY)
                    last_error = str(e)
                except Exception as e:
                    logger.error(f"Upload failed: {e}")
                    last_error = str(e)
        logger.error(f"All upload attempts failed. Last error: {last_error}")
        self.consecutive_failures += 1
        return False, last_error or "Upload failed after retries"

    def upload_method_standard(self, image_path, caption):
        return self.client.photo_upload(image_path, caption)

    def upload_method_with_metadata(self, image_path, caption):
        return self.client.photo_upload(
            image_path,
            caption,
            extra_data={
                "custom_accessibility_caption": caption[:100] if caption else "",
                "like_and_view_counts_disabled": 0,
                "disable_comments": 0
            }
        )

    def upload_method_basic(self, image_path, caption):
        return self.client.photo_upload(image_path, caption or "")

def prompt_to_hashtags(prompt):
    words = prompt.split()
    hashtags = ''.join(f"#{word.strip(',').strip('.')}" for word in words)
    return hashtags

def generate_and_post_loop(
    pipeline,
    ig_manager,
    prompt_manager,
    admin_chat_id=None
):
    while True:
        prompt = prompt_manager.generate_prompt()
        seed = random.randint(0, 2**32 - 1)
        logger.info(f"Generating image for prompt: '{prompt}' with seed: {seed}")
        image_filename = f"img_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{seed}.png"
        image_path = os.path.join(Config.IMAGES_DIR, image_filename)
        os.makedirs(Config.IMAGES_DIR, exist_ok=True)
        try:
            # Generate with CPU only
            result = pipeline(
                prompt,
                num_inference_steps=32,
                generator=torch.manual_seed(seed)
            )
            image = result.images[0]
            image.save(image_path)
            logger.info(f"Image generated: {image_path}")

            hashtags_caption = prompt_to_hashtags(prompt)
            caption = f"{hashtags_caption}\n\nAI-generated art. Seed: {seed} #AIart #Diffusion"

            success, status = ig_manager.post_with_advanced_retry(image_path, caption, seed)
            logger.info(f"Post result: {status}")
        except Exception as e:
            logger.error(f"Generation or posting failed: {e}")
            if admin_chat_id:
                pass  # Optionally notify admin via Telegram here
        sleep_time = random.randint(Config.MIN_INTERVAL, Config.MAX_INTERVAL)
        logger.info(f"Sleeping for {sleep_time // 60} minutes before next post")
        time.sleep(sleep_time)

async def main():
    cred_mgr = CredentialManager()
    if not cred_mgr.decrypt_credentials():
        print("Please run the setup to add your encrypted Instagram credentials first.")
        sys.exit(1)
    word_manager = WordManager()
    ig_manager = AdvancedInstagramManager()
    try:
        pipeline = DiffusionPipeline.from_pretrained(
            Config.MODEL_NAME, torch_dtype=torch.float16
        )
        # Force CPU
        pipeline.to("cpu")
    except Exception as e:
        logger.error(f"Failed to load model pipeline: {e}")
        sys.exit(1)
    threading.Thread(
        target=generate_and_post_loop,
        args=
