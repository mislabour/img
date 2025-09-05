#!/usr/bin/env python3
"""
AI Image Generator Bot with Instagram Integration
Generates images using Stable Diffusion, posts to Instagram, managed via Telegram bot
"""

import os
import random
import requests
import json
import asyncio
import logging
from pathlib import Path
from datetime import datetime
from typing import List, Optional

import torch
from diffusers import DiffusionPipeline
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Application, CommandHandler, CallbackQueryHandler, ContextTypes
from instagrapi import Client
from instagrapi.exceptions import LoginRequired
import pickle

# Configure logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

class Config:
    """Configuration class for all settings"""
    # Telegram Bot Token (get from @BotFather)
    TELEGRAM_BOT_TOKEN = "YOUR_TELEGRAM_BOT_TOKEN_HERE"
    
    # Instagram credentials
    INSTAGRAM_USERNAME = "your_instagram_username"
    INSTAGRAM_PASSWORD = "your_instagram_password"
    
    # File paths
    SESSION_FILE = "instagram_session.pkl"
    IMAGES_DIR = "generated_images"
    
    # Image generation settings
    IMAGE_HEIGHT = 512
    IMAGE_WIDTH = 512
    
    # BIP-39 word list URL
    BIP_WORDS_URL = "https://raw.githubusercontent.com/bitcoin/bips/master/bip-0039/english.txt"

class WordListManager:
    """Manages the BIP word list for generating prompts"""
    
    def __init__(self):
        self.words: List[str] = []
        self.load_words()
    
    def load_words(self):
        """Load BIP-39 word list from GitHub"""
        try:
            response = requests.get(Config.BIP_WORDS_URL)
            response.raise_for_status()
            self.words = response.text.strip().split('\n')
            logger.info(f"Loaded {len(self.words)} words from BIP list")
        except Exception as e:
            logger.error(f"Failed to load BIP words: {e}")
            # Fallback words if BIP list fails
            self.words = [
                "abstract", "ancient", "artistic", "beautiful", "cosmic", "digital", 
                "ethereal", "fantasy", "golden", "magical", "mysterious", "neon",
                "organic", "surreal", "vibrant", "whispered", "crystalline", "flowing",
                "luminous", "shadowy", "textured", "dynamic", "serene", "bold"
            ]
    
    def generate_prompt(self, seed: int = None) -> str:
        """Generate a 7-word prompt using random seed"""
        if seed is not None:
            random.seed(seed)
        else:
            random.seed()
        
        selected_words = random.sample(self.words, 7)
        prompt = " ".join(selected_words)
        logger.info(f"Generated prompt: {prompt}")
        return prompt

class ImageGenerator:
    """Handles Stable Diffusion image generation"""
    
    def __init__(self):
        self.pipe = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.load_model()
        
        # Ensure images directory exists
        Path(Config.IMAGES_DIR).mkdir(exist_ok=True)
    
    def load_model(self):
        """Load the Stable Diffusion model"""
        try:
            logger.info("Loading Stable Diffusion model...")
            self.pipe = DiffusionPipeline.from_pretrained(
                "runwayml/stable-diffusion-v1-5",
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
            )
            self.pipe = self.pipe.to(self.device)
            logger.info(f"Model loaded on {self.device}")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def generate_image(self, prompt: str, seed: int = None) -> str:
        """Generate image from prompt and return file path"""
        try:
            if seed is not None:
                generator = torch.Generator(device=self.device).manual_seed(seed)
            else:
                generator = None
            
            logger.info(f"Generating image for prompt: {prompt}")
            
            # Generate image
            result = self.pipe(
                prompt,
                height=Config.IMAGE_HEIGHT,
                width=Config.IMAGE_WIDTH,
                generator=generator,
                num_inference_steps=30,
                guidance_scale=7.5
            )
            
            image = result.images[0]
            
            # Save image with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"generated_{timestamp}.png"
            filepath = Path(Config.IMAGES_DIR) / filename
            
            image.save(filepath)
            logger.info(f"Image saved: {filepath}")
            
            return str(filepath)
            
        except Exception as e:
            logger.error(f"Failed to generate image: {e}")
            raise

class InstagramManager:
    """Handles Instagram posting with session persistence"""
    
    def __init__(self):
        self.client = Client()
        self.session_file = Path(Config.SESSION_FILE)
        self.load_session()
    
    def load_session(self):
        """Load existing Instagram session if available"""
        try:
            if self.session_file.exists():
                with open(self.session_file, 'rb') as f:
                    session_data = pickle.load(f)
                    self.client.set_settings(session_data)
                    logger.info("Loaded Instagram session from file")
            
            # Try to get account info to verify session
            try:
                self.client.account_info()
                logger.info("Instagram session is valid")
                return True
            except LoginRequired:
                logger.info("Session expired, need to login")
                return self.login()
                
        except Exception as e:
            logger.error(f"Failed to load session: {e}")
            return self.login()
    
    def login(self) -> bool:
        """Login to Instagram and save session"""
        try:
            logger.info("Logging into Instagram...")
            self.client.login(Config.INSTAGRAM_USERNAME, Config.INSTAGRAM_PASSWORD)
            
            # Save session
            with open(self.session_file, 'wb') as f:
                pickle.dump(self.client.get_settings(), f)
            
            logger.info("Instagram login successful, session saved")
            return True
            
        except Exception as e:
            logger.error(f"Instagram login failed: {e}")
            return False
    
    def post_image(self, image_path: str, caption: str) -> bool:
        """Post image to Instagram"""
        try:
            logger.info(f"Posting image to Instagram: {image_path}")
            
            media = self.client.photo_upload(
                path=image_path,
                caption=caption
            )
            
            logger.info(f"Successfully posted to Instagram: {media.pk}")
            return True
            
        except LoginRequired:
            logger.warning("Instagram session expired, attempting re-login")
            if self.login():
                return self.post_image(image_path, caption)
            return False
            
        except Exception as e:
            logger.error(f"Failed to post to Instagram: {e}")
            return False

class TelegramBot:
    """Telegram bot for managing the image generation process"""
    
    def __init__(self):
        self.word_manager = WordListManager()
        self.image_generator = ImageGenerator()
        self.instagram_manager = InstagramManager()
    
    async def start_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /start command"""
        keyboard = [
            [InlineKeyboardButton("🎨 Generate & Post Image", callback_data="generate")],
            [InlineKeyboardButton("📊 Status", callback_data="status")],
            [InlineKeyboardButton("🔧 Settings", callback_data="settings")]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        await update.message.reply_text(
            "🤖 AI Image Generator Bot\n\n"
            "I can generate AI images using random prompts and post them to Instagram!\n\n"
            "Features:\n"
            "• Random 7-word prompts from BIP word list\n"
            "• Stable Diffusion image generation\n"
            "• Automatic Instagram posting\n"
            "• Session persistence\n\n"
            "Choose an option below:",
            reply_markup=reply_markup
        )
    
    async def button_callback(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle button callbacks"""
        query = update.callback_query
        await query.answer()
        
        if query.data == "generate":
            await self.generate_and_post(query)
        elif query.data == "status":
            await self.show_status(query)
        elif query.data == "settings":
            await self.show_settings(query)
    
    async def generate_and_post(self, query):
        """Generate image and post to Instagram"""
        try:
            await query.edit_message_text("🎨 Generating image, please wait...")
            
            # Generate random prompt with seed
            seed = random.randint(1, 1000000)
            prompt = self.word_manager.generate_prompt(seed)
            
            # Generate image
            image_path = self.image_generator.generate_image(prompt, seed)
            
            await query.edit_message_text(
                f"✅ Image generated!\n"
                f"📝 Prompt: {prompt}\n"
                f"🌱 Seed: {seed}\n\n"
                f"📤 Posting to Instagram..."
            )
            
            # Create Instagram caption
            caption = f"🤖 AI Generated Art\n\n📝 Prompt: {prompt}\n🌱 Seed: {seed}\n\n#ai #aiart #stablediffusion #generated #art #digital"
            
            # Post to Instagram
            success = self.instagram_manager.post_image(image_path, caption)
            
            if success:
                await query.edit_message_text(
                    f"🎉 Success!\n\n"
                    f"📝 Prompt: {prompt}\n"
                    f"🌱 Seed: {seed}\n"
                    f"📱 Posted to Instagram: @{Config.INSTAGRAM_USERNAME}\n"
                    f"💾 Image saved: {Path(image_path).name}"
                )
            else:
                await query.edit_message_text(
                    f"⚠️ Image generated but Instagram posting failed\n\n"
                    f"📝 Prompt: {prompt}\n"
                    f"🌱 Seed: {seed}\n"
                    f"💾 Image saved: {Path(image_path).name}"
                )
            
        except Exception as e:
            logger.error(f"Generation process failed: {e}")
            await query.edit_message_text(f"❌ Error: {str(e)}")
    
    async def show_status(self, query):
        """Show system status"""
        try:
            # Check Instagram session
            try:
                self.instagram_manager.client.account_info()
                ig_status = "✅ Connected"
            except:
                ig_status = "❌ Disconnected"
            
            # Check model status
            model_status = "✅ Loaded" if self.image_generator.pipe else "❌ Not loaded"
            
            # Count generated images
            image_count = len(list(Path(Config.IMAGES_DIR).glob("*.png")))
            
            status_text = (
                f"📊 System Status\n\n"
                f"🤖 Model: {model_status}\n"
                f"📱 Instagram: {ig_status}\n"
                f"💽 Device: {self.image_generator.device.upper()}\n"
                f"🖼️ Generated Images: {image_count}\n"
                f"📝 BIP Words: {len(self.word_manager.words)}"
            )
            
            await query.edit_message_text(status_text)
            
        except Exception as e:
            await query.edit_message_text(f"❌ Error getting status: {str(e)}")
    
    async def show_settings(self, query):
        """Show current settings"""
        settings_text = (
            f"🔧 Current Settings\n\n"
            f"📐 Image Size: {Config.IMAGE_WIDTH}x{Config.IMAGE_HEIGHT}\n"
            f"📱 Instagram: @{Config.INSTAGRAM_USERNAME}\n"
            f"💾 Images Dir: {Config.IMAGES_DIR}\n"
            f"🔗 BIP Words Source: GitHub\n\n"
            f"ℹ️ Edit config.py to change settings"
        )
        
        await query.edit_message_text(settings_text)
    
    def run(self):
        """Start the Telegram bot"""
        logger.info("Starting Telegram bot...")
        
        application = Application.builder().token(Config.TELEGRAM_BOT_TOKEN).build()
        
        # Add handlers
        application.add_handler(CommandHandler("start", self.start_command))
        application.add_handler(CallbackQueryHandler(self.button_callback))
        
        # Start the bot
        application.run_polling(allowed_updates=Update.ALL_TYPES)

def main():
    """Main function to start the bot"""
    try:
        # Validate configuration
        if Config.TELEGRAM_BOT_TOKEN == "YOUR_TELEGRAM_BOT_TOKEN_HERE":
            print("❌ Please set your Telegram bot token in the Config class")
            return
        
        if Config.INSTAGRAM_USERNAME == "your_instagram_username":
            print("❌ Please set your Instagram credentials in the Config class")
            return
        
        print("🚀 Starting AI Image Generator Bot...")
        bot = TelegramBot()
        bot.run()
        
    except KeyboardInterrupt:
        print("\n👋 Bot stopped by user")
    except Exception as e:
        print(f"❌ Error starting bot: {e}")

if __name__ == "__main__":
    main()
