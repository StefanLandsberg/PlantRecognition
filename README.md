# Plant Recognition System

A deep learning-based system for accurate plant species identification from images.

## Overview

This project implements a hierarchical plant recognition system using state-of-the-art deep learning techniques. The system can classify plant species from images with high accuracy by leveraging advanced neural network architectures, specialized data augmentation, and efficient training strategies.

## Features

- Support for hundreds of plant species
- Advanced data augmentation tailored specifically for plant images
- Hierarchical classification approach for improved accuracy
- Multiple model architectures (DenseNet121, MobileNetV3, EfficientNetV2)
- Web interface for easy interaction

## Directory Structure

```
PlantRecognition/
├── checkpoints/              # Model checkpoint files
├── data/
│   ├── plant_images/         # Original plant images sorted by species
│   ├── features/             # Extracted features for training
│   └── observations-561226.csv  # Plant observation data
├── logs/                     # Training logs
├── models/                   # Saved model files
├── PlantRgognotionV2/        # Version 2 of the recognition system
│   ├── checkpoints_v2/
│   ├── logs_v2/
│   ├── models_v2/
│   ├── scripts/              # V2 scripts
│   └── stats_v2/
├── Scripts/
│   ├── get_images_from_csv.py     # Data collection script
│   ├── offline_augmentation.py    # Offline data augmentation
│   ├── Plant_Reconition.py        # Main recognition script (V1)
│   └── plant_test.py              # Testing script
├── stats/                    # Statistics and metrics
└── webapp/                   # Web application for recognition
    ├── css/
    ├── js/
    ├── models/
    └── index.html
```

## Installation

### Prerequisites

- Python 3.8 or higher
- TensorFlow 2.x
- CUDA and cuDNN (for GPU acceleration, recommended)

### Setup

1. Clone this repository:
   ```
   git clone https://github.com/yourusername/PlantRecognition.git
   cd PlantRecognition
   ```

2. Create and activate a virtual environment (optional but recommended):
   ```
   python -m venv venv
   source venv/bin/activate   # On Windows: venv\Scripts\activate
   ```

3. Install required packages:
   ```
   pip install -r requirements.txt
   ```

## Complete Workflow Guide

This section provides a step-by-step guide for the complete workflow from data preparation to model deployment.

### 1. Data Collection and Preparation

First, download plant images from CSV observations:

```bash
python Scripts/get_images_from_csv.py
```

This script:
- Processes the observation CSV file
- Downloads plant images for each species
- Organizes them in the proper folder structure

### 2. Data Augmentation

The project uses two complementary approaches to data augmentation:

#### a. Offline Augmentation

Run the offline augmentation to save augmented images to disk:

```bash
python Scripts/offline_augmentation.py
```

This process:
- Generates additional training samples with controlled variations
- Balances classes with fewer examples
- Creates augmented versions with modified brightness, contrast, rotation, etc.
- Saves new images to disk to be used in training

The progress is tracked in `augmentation_checkpoint.json`, allowing you to resume if interrupted.

#### b. On-the-fly Augmentation

This is integrated directly in the training pipeline and happens during model training:
- No need to run a separate script for this
- The augmentation happens automatically when training models
- Applied dynamically during training without storing additional images
- Implemented in both V1 (basic) and V2 (advanced) pipelines, with V2 having more sophisticated transformations

### 3. Training Pipeline Options

You have two options for training models: V1 (basic) and V2 (advanced). Choose based on your computational resources and accuracy requirements.

#### Option A: Basic Training (V1)

For a simpler training pipeline:

```bash
python Scripts/Plant_Reconition.py
```

This script:
- Loads the dataset (including augmented images)
- Performs basic online augmentation during training
- Trains using transfer learning with pre-trained models
- Saves models to the `models/` directory

#### Option B: Advanced Training Pipeline (V2 - Recommended)

The V2 pipeline includes advanced augmentation and training techniques:

1. **Advanced Augmentation** (Run this first for V2):
   ```bash
   python PlantRgognotionV2/scripts/advanced_augmentation.py
   ```
   This performs specialized augmentations for plant imagery that goes beyond basic techniques:
   - Advanced color space transformations
   - Targeted background modifications
   - Plant-specific deformations and variations
   - Enhanced leaf structure preservation

2. **Main V2 Training Pipeline**:
   ```bash
   python PlantRgognotionV2/scripts/PlantRecognitionV2.py
   ```

The V2 pipeline follows these steps automatically:
1. **Parameter-Efficient Fine-Tuning (PEFT)** - Fine-tunes EfficientNetV2L and DenseNet121 models
2. **Feature Extraction** - Extracts rich features from fine-tuned models
3. **Chunked Training** - Processes features in manageable chunks with contrastive learning
4. **Model Distillation** - Distills knowledge from larger models to smaller, more efficient ones

### 4. Ensemble Model Creation (Optional)

For improved accuracy, create an ensemble of multiple models:

```bash
python Scripts/create_ensemble.py --models ./models/model1.h5 ./models/model2.h5 ./models/model3.h5 --output ./models/ensemble_model.h5
```

This combines predictions from multiple models for better performance.

### 5. Test-Time Augmentation (Evaluation)

To evaluate your models with test-time augmentation:

```bash
python Scripts/plant_test.py --model ./models/model_name.h5 --tta
```

The `--tta` flag enables test-time augmentation, which:
- Creates multiple versions of each test image
- Averages predictions across augmented versions
- Provides more robust classification results

Without the flag, standard evaluation is performed:

```bash
python Scripts/plant_test.py --model ./models/model_name.h5
```

### 6. Model Deployment

To use your trained model in the web application:

1. Convert the model to web format (if needed):
   ```bash
   python plant_model_convert.py --model ./models/final_model.h5 --output ./webapp/models/web_model
   ```

2. Open `webapp/index.html` in a web browser to use the interface

## When to Use Each Component

- **Offline Augmentation**: Always use this before training for better results
- **V2 Pipeline**: Use when maximum accuracy is needed and you have good GPU resources
- **Ensemble Models**: Use when you have several models with complementary strengths
- **Test-Time Augmentation**: Use during evaluation and deployment for improved robustness

## Performance Tuning

- For faster training but lower accuracy: Use V1 with MobileNetV3
- For highest accuracy: Use V2 with EfficientNetV2L and ensemble methods
- For limited GPU memory: Reduce batch sizes in the scripts
- For deployment on edge devices: Use the distilled models from V2

## Resources

- [TensorFlow Documentation](https://www.tensorflow.org/api_docs)
- [Plant Recognition Techniques](https://www.sciencedirect.com/topics/agricultural-and-biological-sciences/plant-identification)

## License

This project is licensed under the terms included in the LICENSE file.

## Citation

If you use this code in your research, please cite:

```
@software{plant_recognition,
  author = {Your Name},
  title = {Plant Recognition System},
  year = {2025},
  url = {https://github.com/yourusername/PlantRecognition}
}
```

## Acknowledgments

This project uses images and data from various botanical datasets and resources.