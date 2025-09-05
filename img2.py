#!/bin/bash

echo "🤖 AI Image Generator Bot - Setup Script"
echo "========================================"

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed. Please install Python 3.8 or higher."
    exit 1
fi

# Check Python version
python_version=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
echo "✅ Python $python_version detected"

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "⬆️ Upgrading pip..."
pip install --upgrade pip

# Install PyTorch first (with CUDA support if available)
echo "🔥 Installing PyTorch..."
if command -v nvidia-smi &> /dev/null; then
    echo "🎮 NVIDIA GPU detected, installing CUDA version..."
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
else
    echo "💻 Installing CPU version..."
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
fi

# Install other required packages
echo "📚 Installing required packages..."
pip install diffusers==0.24.0
pip install transformers==4.35.0
pip install accelerate==0.24.0
pip install python-telegram-bot==20.7
pip install instagrapi==1.19.3
pip install requests==2.31.0
pip install Pillow==10.1.0

# Check if all packages are installed correctly
echo "🔍 Checking installation..."
python3 -c "
import torch
import diffusers
import telegram
import instagrapi
import requests
from PIL import Image
print('✅ All packages imported successfully!')
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
"

# Create directories
echo "📁 Creating directories..."
mkdir -p generated_images

# Create requirements.txt for future reference
echo "📝 Creating requirements.txt..."
cat > requirements.txt << EOL
torch>=2.0.0
torchvision>=0.15.0
diffusers==0.24.0
transformers==4.35.0
accelerate==0.24.0
python-telegram-bot==20.7
instagrapi==1.19.3
requests==2.31.0
Pillow==10.1.0
EOL

echo ""
echo "🎉 Installation complete!"
echo ""
echo "📋 Next steps:"
echo "1. Get a Telegram bot token from @BotFather"
echo "2. Edit the Config class in img.py with your credentials:"
echo "   - TELEGRAM_BOT_TOKEN = '7933552663:AAGp2eQVrT5pZNgKRwWde07yW-CYcxlWyHo'"
echo "   - INSTAGRAM_USERNAME = 'your_username'"
echo "   - INSTAGRAM_PASSWORD = 'your_password'"
echo ""
echo "3. Run the bot:"
echo "   source venv/bin/activate"
echo "   python3 img.py"
echo ""
echo "🔧 Troubleshooting:"
echo "- If you get import errors, make sure virtual environment is activated"
echo "- For CUDA issues, check: python3 -c 'import torch; print(torch.cuda.is_available())'"
echo "- For Instagram issues, try logging in manually first"
