#!/usr/bin/env python3
"""
AI Image Generator Bot with Instagram Integration - Fixed Version
Fixes the enable_model_cpu_offload error and other issues
"""

import os
import sys
import random
import requests
import json
import asyncio
import logging
import pickle
import torch
from pathlib import Path
from datetime import datetime
from typing import List, Optional

def check_and_install_packages():
    """Check if required packages are installed"""
    required_packages = [
        'diffusers', 'transformers', 'accelerate', 
        'python-telegram-bot', 'instagrapi', 'requests', 'Pillow'
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

try:
    from diffusers import DiffusionPipeline
    from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
    from telegram.ext import Application, CommandHandler, CallbackQueryHandler, ContextTypes
    from instagrapi import Client
    from instagrapi.exceptions import LoginRequired
    from PIL import Image
    import accelerate
except ImportError as e:
    print(f"Import error: {e}")
    print("Please make sure all required packages are installed.")
    sys.exit(1)

logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

class Config:
    """Configuration class"""
    
    # Telegram bot token from @BotFather
    TELEGRAM_BOT_TOKEN = "7933552663:AAGp2eQVrT5pZNgKRwWde07yW-CYcxlWyHo"
    
    # Instagram credentials
    INSTAGRAM_USERNAME = "your_username"
    INSTAGRAM_PASSWORD = "your_password"
    
    # File paths
    SESSION_FILE = "instagram_session.pkl"
    IMAGES_DIR = "generated_images"
    
    # Image settings
    IMAGE_HEIGHT = 512
    IMAGE_WIDTH = 512
    
    # BIP word list URL
    BIP_WORDS_URL = "https://raw.githubusercontent.com/bitcoin/bips/master/bip-0039/english.txt"

class WordManager:
    """Manages word list for prompt generation"""
    
    def __init__(self):
        self.words = []
        self.load_words()
    
    def load_words(self):
        """Load words from BIP list or use fallback"""
        try:
            print("Loading BIP-39 word list...")
            response = requests.get(Config.BIP_WORDS_URL, timeout=10)
            if response.status_code == 200:
                self.words = [word.strip() for word in response.text.split('\n') if word.strip()]
                print(f"Loaded {len(self.words)} words from BIP list")
            else:
                raise Exception(f"HTTP {response.status_code}")
        except Exception as e:
            print(f"Failed to load BIP words ({e}), using fallback...")
            self.words = [
                "abstract", "ancient", "beautiful", "cosmic", "digital", "ethereal",
                "fantasy", "golden", "magical", "mysterious", "neon", "organic", 
                "surreal", "vibrant", "crystalline", "flowing", "luminous", "shadowy",
                "textured", "dynamic", "serene", "bold", "electric", "dreamy",
                "futuristic", "glowing", "infinite", "peaceful", "radiant", "smooth",
                "alien", "baroque", "celestial", "delicate", "enchanted", "fractal",
                "geometric", "holographic", "iridescent", "kaleidoscopic", "liquid",
                "metallic", "nebular", "opalescent", "prismatic", "quantum", "stellar",
                "translucent", "ultramarine", "velvet", "whimsical", "xenomorph", "zen"
            ]
            print(f"Using {len(self.words)} fallback words")
    
    def generate_prompt(self, seed=None):
        """Generate 7-word prompt"""
        if seed:
            random.seed(seed)
        
        if len(self.words) < 7:
            selected = [random.choice(self.words) for _ in range(7)]
        else:
            selected = random.sample(self.words, 7)
        
        prompt = " ".join(selected)
        print(f"Generated prompt: {prompt}")
        return prompt

class ImageGenerator:
    """Handles image generation using Stable Diffusion - FIXED VERSION"""
    
    def __init__(self):
        self.pipe = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")
        
        os.makedirs(Config.IMAGES_DIR, exist_ok=True)
        self.load_model()
    
    def load_model(self):
        """Load Stable Diffusion model with proper error handling"""
        try:
            print("Loading Stable Diffusion model...")
            print("This may take several minutes on first run...")
            
            # Use appropriate dtype based on device
            if self.device == "cuda":
                torch_dtype = torch.float16
            else:
                torch_dtype = torch.float32
            
            # Load the pipeline
            self.pipe = DiffusionPipeline.from_pretrained(
                "runwayml/stable-diffusion-v1-5",
                torch_dtype=torch_dtype,
                safety_checker=None,
                requires_safety_checker=False,
                use_safetensors=True
            )
            
            # Move to device
            self.pipe = self.pipe.to(self.device)
            
            # Enable memory optimizations - FIXED VERSION
            if hasattr(self.pipe, 'enable_attention_slicing'):
                self.pipe.enable_attention_slicing()
                print("Enabled attention slicing for memory optimization")
            
            # Only enable CPU offload for CUDA devices with limited memory
            if self.device == "cuda":
                try:
                    # Check available GPU memory
                    gpu_memory = torch.cuda.get_device_properties(0).total_memory
                    gpu_memory_gb = gpu_memory / (1024**3)
                    print(f"GPU Memory: {gpu_memory_gb:.1f} GB")
                    
                    # Enable CPU offload for GPUs with less than 8GB
                    if gpu_memory_gb < 8:
                        print("Enabling CPU offload for limited GPU memory...")
                        self.pipe.enable_model_cpu_offload()
                        
                except Exception as e:
                    print(f"Could not enable CPU offload: {e}")
            
            # Enable sequential CPU offload for CPU-only devices
            elif self.device == "cpu":
                try:
                    print("Enabling sequential CPU offload...")
                    self.pipe.enable_sequential_cpu_offload()
                except Exception as e:
                    print(f"Could not enable sequential CPU offload: {e}")
                    # Continue without offloading
            
            print("Model loaded successfully")
            
        except Exception as e:
            print(f"Failed to load model: {e}")
            print("Troubleshooting tips:")
            print("1. Make sure you have enough disk space (10GB+)")
            print("2. Check internet connection for model download")
            print("3. Try running with --no-cache-dir flag")
            raise
    
    def generate_image(self, prompt, seed=None):
        """Generate image and return file path"""
        try:
            print(f"Generating image for: {prompt}")
            
            # Set up generator with seed
            generator = None
            if seed:
                generator = torch.Generator(device=self.device).manual_seed(seed)
            
            # Generate image with proper memory management
            with torch.no_grad():
                result = self.pipe(
                    prompt,
                    height=Config.IMAGE_HEIGHT,
                    width=Config.IMAGE_WIDTH,
                    num_inference_steps=20,  # Reduced for faster generation
                    guidance_scale=7.5,
                    generator=generator
                )
            
            image = result.images[0]
            
            # Save image with timestamp and seed
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            if seed:
                filename = f"ai_art_{timestamp}_seed{seed}.png"
            else:
                filename = f"ai_art_{timestamp}.png"
            
            filepath = os.path.join(Config.IMAGES_DIR, filename)
            image.save(filepath)
            
            print(f"Image saved: {filepath}")
            
            # Clear cache to free memory
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            return filepath
            
        except Exception as e:
            print(f"Image generation failed: {e}")
            # Clear cache on error too
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            raise

class InstagramManager:
    """Manages Instagram posting with session persistence"""
    
    def __init__(self):
        self.client = Client()
        self.session_file = Config.SESSION_FILE
        self.logged_in = False
        self.load_session()
    
    def load_session(self):
        """Load existing Instagram session"""
        try:
            if os.path.exists(self.session_file):
                print("Loading Instagram session...")
                with open(self.session_file, 'rb') as f:
                    session = pickle.load(f)
                    self.client.set_settings(session)
                
                try:
                    self.client.account_info()
                    print("Instagram session is valid")
                    self.logged_in = True
                    return
                except LoginRequired:
                    print("Session expired, need to login again")
            
            self.login()
            
        except Exception as e:
            print(f"Session load failed: {e}")
            self.login()
    
    def login(self):
        """Login to Instagram"""
        try:
            if Config.INSTAGRAM_USERNAME == "your_username":
                print("Please set Instagram credentials in Config class")
                return False
            
            print("Logging into Instagram...")
            self.client.login(Config.INSTAGRAM_USERNAME, Config.INSTAGRAM_PASSWORD)
            
            with open(self.session_file, 'wb') as f:
                pickle.dump(self.client.get_settings(), f)
            
            print("Instagram login successful")
            self.logged_in = True
            return True
            
        except Exception as e:
            print(f"Instagram login failed: {e}")
            self.logged_in = False
            return False
    
    def post_image(self, image_path, caption):
        """Post image to Instagram"""
        try:
            if not self.logged_in:
                print("Not logged into Instagram")
                return False
            
            print(f"Posting to Instagram: {os.path.basename(image_path)}")
            
            media = self.client.photo_upload(
                path=image_path,
                caption=caption
            )
            
            print(f"Successfully posted to Instagram. Media ID: {media.pk}")
            return True
            
        except LoginRequired:
            print("Session expired, attempting re-login...")
            if self.login():
                return self.post_image(image_path, caption)
            return False
            
        except Exception as e:
            print(f"Instagram posting failed: {e}")
            return False

class TelegramBot:
    """Main Telegram bot class"""
    
    def __init__(self):
        print("Initializing bot components...")
        self.word_manager = WordManager()
        self.image_generator = ImageGenerator()
        self.instagram_manager = InstagramManager()
        print("Bot initialization complete")
    
    async def start_handler(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /start command"""
        keyboard = [
            [InlineKeyboardButton("Generate AI Art", callback_data="generate")],
            [InlineKeyboardButton("Check Status", callback_data="status")],
            [InlineKeyboardButton("Help", callback_data="help")]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        welcome_msg = (
            "AI Art Generator Bot\n\n"
            "I can create AI art and post it to Instagram automatically!\n\n"
            "Features:\n"
            "- Random prompts from BIP word list\n"
            "- Stable Diffusion image generation\n"
            "- Automatic Instagram posting\n"
            "- Session persistence\n\n"
            "Choose an option below:"
        )
        
        await update.message.reply_text(welcome_msg, reply_markup=reply_markup)
    
    async def button_handler(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle button callbacks"""
        query = update.callback_query
        await query.answer()
        
        if query.data == "generate":
            await self.handle_generate(query)
        elif query.data == "status":
            await self.handle_status(query)
        elif query.data == "help":
            await self.handle_help(query)
    
    async def handle_generate(self, query):
        """Generate and post image"""
        try:
            await query.edit_message_text("Starting AI art generation...")
            
            # Generate random prompt with seed
            seed = random.randint(1, 999999)
            prompt = self.word_manager.generate_prompt(seed)
            
            await query.edit_message_text(
                f"Generating AI art...\n\n"
                f"Prompt: {prompt}\n"
                f"Seed: {seed}\n\n"
                f"This may take 1-3 minutes..."
            )
            
            # Generate image
            image_path = self.image_generator.generate_image(prompt, seed)
            
            await query.edit_message_text(
                f"Image generated successfully!\n\n"
                f"Prompt: {prompt}\n"
                f"Seed: {seed}\n\n"
                f"Posting to Instagram..."
            )
            
            # Create Instagram caption
            caption = (
                f"AI Generated Art\n\n"
                f"Prompt: {prompt}\n"
                f"Seed: {seed}\n"
                f"Generated with Stable Diffusion\n\n"
                f"#ai #aiart #stablediffusion #generated #digitalart #art #creative #machinelearning"
            )
            
            # Post to Instagram
            success = self.instagram_manager.post_image(image_path, caption)
            
            if success:
                # Send the image to user as well
                with open(image_path, 'rb') as photo:
                    await query.message.reply_photo(
                        photo=photo,
                        caption=f"Success! Posted to @{Config.INSTAGRAM_USERNAME}\n\n"
                               f"Prompt: {prompt}\n"
                               f"Seed: {seed}"
                    )
                
                await query.edit_message_text(
                    f"Success!\n\n"
                    f"Prompt: {prompt}\n"
                    f"Seed: {seed}\n"
                    f"Posted to @{Config.INSTAGRAM_USERNAME}\n"
                    f"Saved as: {os.path.basename(image_path)}\n\n"
                    f"Generate another image anytime!"
                )
            else:
                await query.edit_message_text(
                    f"Image generated but Instagram posting failed\n\n"
                    f"Prompt: {prompt}\n"
                    f"Seed: {seed}\n"
                    f"Saved locally: {os.path.basename(image_path)}\n\n"
                    f"Please check Instagram credentials and try again."
                )
                
        except Exception as e:
            error_msg = f"Error occurred: {str(e)}\n\nPlease try again."
            await query.edit_message_text(error_msg)
            print(f"Generation error: {e}")
    
    async def handle_status(self, query):
        """Show system status"""
        try:
            model_status = "Loaded" if self.image_generator.pipe else "Not loaded"
            ig_status = "Connected" if self.instagram_manager.logged_in else "Disconnected"
            
            if os.path.exists(Config.IMAGES_DIR):
                image_count = len([f for f in os.listdir(Config.IMAGES_DIR) if f.endswith('.png')])
            else:
                image_count = 0
            
            # Check GPU memory if available
            gpu_info = ""
            if torch.cuda.is_available():
                gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                gpu_memory_used = torch.cuda.memory_allocated(0) / (1024**3)
                gpu_info = f"\nGPU Memory: {gpu_memory_used:.1f}/{gpu_memory:.1f} GB"
            
            status_text = (
                f"System Status\n\n"
                f"AI Model: {model_status}\n"
                f"Instagram: {ig_status}\n"
                f"Device: {self.image_generator.device.upper()}\n"
                f"Images Generated: {image_count}\n"
                f"Words Available: {len(self.word_manager.words)}"
                f"{gpu_info}\n\n"
                f"System ready for art generation!"
            )
            
            await query.edit_message_text(status_text)
            
        except Exception as e:
            await query.edit_message_text(f"Status check failed: {e}")
    
    async def handle_help(self, query):
        """Show help information"""
        help_text = (
            f"Help and Information\n\n"
            f"How it works:\n"
            f"1. Bot generates random 7-word prompts\n"
            f"2. Creates AI art using Stable Diffusion\n"
            f"3. Posts automatically to Instagram\n\n"
            f"Current Settings:\n"
            f"- Image Size: {Config.IMAGE_WIDTH}x{Config.IMAGE_HEIGHT}\n"
            f"- Model: Stable Diffusion v1.5\n"
            f"- Word Source: BIP-39 list\n\n"
            f"Commands:\n"
            f"/start - Show main menu\n\n"
            f"Tips:\n"
            f"- Generation takes 1-3 minutes\n"
            f"- Images are saved locally\n"
            f"- Each image has a unique seed\n"
            f"- Seeds can be used to reproduce images\n"
            f"- Bot sends you the generated image too"
        )
        
        await query.edit_message_text(help_text)
    
    def run(self):
        """Start the bot"""
        if Config.TELEGRAM_BOT_TOKEN == "YOUR_BOT_TOKEN_HERE":
            print("ERROR: Please set TELEGRAM_BOT_TOKEN in Config class")
            print("Get your token from @BotFather on Telegram")
            return
        
        print("Starting Telegram bot...")
        
        application = Application.builder().token(Config.TELEGRAM_BOT_TOKEN).build()
        
        application.add_handler(CommandHandler("start", self.start_handler))
        application.add_handler(CallbackQueryHandler(self.button_handler))
        
        print("Bot is running! Send /start to your bot to begin.")
        print("Press Ctrl+C to stop the bot.")
        
        try:
            application.run_polling(allowed_updates=Update.ALL_TYPES)
        except KeyboardInterrupt:
            print("\nBot stopped by user")

def main():
    """Main
