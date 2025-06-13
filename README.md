# 🎨 Automated Story Illustration

An AI-powered pipeline for creating consistent character illustrations for children's storybooks. This project automates the entire process from character description to trained AI models that can generate new images of your characters in different scenarios.

## ✨ Features

- **Automated Character Sheet Generation**: Generate character, action, and emotion sheets using OpenAI's GPT-4 with vision
- **Image Processing Pipeline**: Automatically upscale and segment character sheets into individual training images
- **AI Model Training**: Train custom SDXL models on your character images using Replicate
- **Character Consistency**: Maintain visual consistency across all generated images
- **Real-ESRGAN Integration**: High-quality image upscaling for better training data
- **Interactive Generation**: Generate new images of your trained characters
- **Comprehensive Logging**: Track the entire pipeline with detailed logs

## 🚀 Quick Start

### Prerequisites

- Python 3.11+
- OpenAI API key
- Replicate API key
- AWS S3 bucket (for training data upload)

### Installation

1. **Clone the repository:**
```bash
git clone https://github.com/yourusername/Automated-Story-Illustration.git
cd Automated-Story-Illustration
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

3. **Set up environment variables:**
Create a `.env` file in the root directory:
```env
OPENAI_API_KEY=your_openai_api_key
REPLICATE_API_TOKEN=your_replicate_token
AWS_ACCESS_KEY_ID=your_aws_access_key
AWS_SECRET_ACCESS_KEY=your_aws_secret_key
S3_BUCKET_NAME=your_s3_bucket_name
```

### Basic Usage

1. **Run the main pipeline:**
```bash
python src/main.py
```

2. **Follow the prompts to:**
   - Enter your character name and description
   - Wait for automatic sheet generation
   - Let the system process and train your model
   - Generate new images with your trained character

## 🛠️ Project Structure

```
Automated-Story-Illustration/
├── src/                          # Main source code
│   ├── main.py                   # Main pipeline orchestrator
│   ├── automated_sheet_handler.py # OpenAI API integration
│   ├── cut_an_image.py           # Image segmentation
│   ├── upscaler.py               # Real-ESRGAN upscaling
│   ├── model_trainer.py          # Replicate model training
│   ├── image_generator.py        # Image generation interface
│   └── pipeline_logger.py        # Logging system
├── z_GPT_Templates/              # GPT prompt templates
│   ├── GPT_Template_for_Sheet_Gen.txt
│   ├── GPT_Template_for_Action_Sheet_Gen.txt
│   └── GPT_Template_for_Emotion_Sheet_Gen.txt
├── Characters/                   # Generated character data
├── generated_images/             # Final generated images
├── Real-ESRGAN/                  # Image upscaling models
├── stable-diffusion-webui/       # Stable Diffusion integration
└── requirements.txt              # Python dependencies
```

## 🔄 Pipeline Workflow

### 1. Character Description Input
- User provides character name and detailed description
- System validates and processes the input

### 2. Automated Sheet Generation
- **Character Sheet**: 6 different poses (front, 3/4, profile, back, sitting, action)
- **Action Sheet**: 6 character-specific actions (movement, athletic, interaction, etc.)
- **Emotion Sheet**: 9 different facial expressions in a 3×3 grid

### 3. Image Processing
- **Upscaling**: Uses Real-ESRGAN (Anime6B model) for high-quality upscaling
- **Segmentation**: Automatically cuts sheets into individual character images

### 4. Model Training
- **Data Preparation**: Creates training dataset from processed images
- **Upload**: Automatically uploads training data to AWS S3
- **Training**: Uses Replicate to train custom SDXL model

### 5. Image Generation
- **Interactive Interface**: Generate new images with custom prompts
- **History Tracking**: Maintain generation history with metadata
- **Batch Generation**: Support for multiple image generation

## 🎯 Advanced Features

### Character Sheet Templates

The system uses specialized GPT templates for each sheet type:

- **Grid-based Layout**: Ensures consistent character positioning
- **Style Consistency**: Maintains visual coherence across all sheets
- **Children's Book Style**: Optimized for storybook illustrations

### Real-ESRGAN Integration

- **Anime6B Model**: Specialized for cartoon/anime-style upscaling
- **4x Upscaling**: Enhances image quality for better training
- **Batch Processing**: Handles multiple images efficiently

### CLIP Evaluation

- **Character Consistency**: Measures visual consistency across generated images
- **Quality Metrics**: Automated evaluation of character similarity
- **Feedback Loop**: Helps improve future generations

## 📝 Configuration

### Model Settings

Edit `src/model_trainer.py` to customize training parameters:

```python
# Training configuration
"max_train_steps": 1000,
"use_face_detection_instead": False,
"token_string": "TOK",
"caption_prefix": "a photo of TOK, "
```

### Generation Parameters

Customize image generation in `src/image_generator.py`:

```python
default_params = {
    "width": 1024,
    "height": 1024,
    "num_inference_steps": 50,
    "guidance_scale": 7.5,
    "num_outputs": 1
}
```

## 🗂️ File Organization

### Character Data Structure
```
Characters/
└── [character_name]/
    ├── uncut_images/           # Original generated sheets
    ├── upscaled_images/        # Upscaled sheets
    ├── cut_images/             # Individual character images
    ├── model_info.json         # Trained model metadata
    └── character_info.json     # Character description
```

### Logs and Monitoring
```
img_gen_logs/                   # Pipeline execution logs
generation_history.db          # Generation history database
characters.db                  # Character database
```

