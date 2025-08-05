#!/usr/bin/env python3
"""
Setup script for plant recognition data directory
Creates the necessary directory structure for training
"""

import os
import shutil
from pathlib import Path
import sys

def create_data_structure():
    """Create the data directory structure for plant images"""
    
    # Define paths
    base_dir = Path(__file__).parent.parent
    data_dir = base_dir / "data" / "plant_images"
    
    print(f"Setting up data directory: {data_dir}")
    
    # Create directory if it doesn't exist
    data_dir.mkdir(parents=True, exist_ok=True)
    
    # Create example class directories
    example_classes = [
        "dandelion",
        "rose", 
        "sunflower",
        "tulip",
        "daisy"
    ]
    
    for class_name in example_classes:
        class_dir = data_dir / class_name
        class_dir.mkdir(exist_ok=True)
        print(f"  Created class directory: {class_name}")
    
    # Create README file with instructions
    readme_path = data_dir / "README.md"
    readme_content = """# Plant Images Directory

This directory should contain plant images organized by class.

## Directory Structure
```
data/plant_images/
├── class1/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
├── class2/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
└── ...
```

## Supported Image Formats
- .jpg / .jpeg
- .png
- .bmp

## Instructions
1. Create a subdirectory for each plant class
2. Place images of that plant class in the subdirectory
3. Use descriptive class names (e.g., "red_rose", "white_daisy")
4. Ensure each class has at least 10-20 images for training

## Example
```
data/plant_images/
├── red_rose/
│   ├── rose_001.jpg
│   ├── rose_002.jpg
│   └── ...
├── white_daisy/
│   ├── daisy_001.jpg
│   ├── daisy_002.jpg
│   └── ...
└── yellow_sunflower/
    ├── sunflower_001.jpg
    ├── sunflower_002.jpg
    └── ...
```

## Training
Once you have added your plant images, run:
```bash
cd modeltrainign
python training.py
```
"""
    
    with open(readme_path, 'w') as f:
        f.write(readme_content)
    
    print(f"  Created README with instructions: {readme_path}")
    
    # Create a sample script to download some test images
    sample_script_path = data_dir / "download_sample_images.py"
    sample_script_content = '''#!/usr/bin/env python3
"""
Sample script to download some test plant images
This is optional - you can replace with your own images
"""

import requests
from pathlib import Path
import os

def download_sample_images():
    """Download some sample plant images for testing"""
    
    # Sample image URLs (replace with actual URLs)
    sample_images = {
        "dandelion": [
            "https://example.com/dandelion1.jpg",
            "https://example.com/dandelion2.jpg"
        ],
        "rose": [
            "https://example.com/rose1.jpg", 
            "https://example.com/rose2.jpg"
        ]
    }
    
    print("Note: This is a template script.")
    print("Replace the URLs with actual image URLs or add your own images manually.")
    print("See README.md for instructions.")

if __name__ == "__main__":
    download_sample_images()
'''
    
    with open(sample_script_path, 'w') as f:
        f.write(sample_script_content)
    
    print(f"  Created sample download script: {sample_script_path}")
    
    print(f"\n✅ Data directory structure created successfully!")
    print(f"📁 Location: {data_dir}")
    print(f"📖 Instructions: {readme_path}")
    print(f"\nNext steps:")
    print(f"1. Add plant images to the class directories")
    print(f"2. Run: cd modeltrainign && python training.py")

def check_existing_data():
    """Check if there's already data in the directory"""
    
    base_dir = Path(__file__).parent.parent
    data_dir = base_dir / "data" / "plant_images"
    
    if data_dir.exists():
        classes = [d for d in data_dir.iterdir() if d.is_dir()]
        if classes:
            print(f"Found existing data:")
            for class_dir in classes:
                images = list(class_dir.glob("*.jpg")) + list(class_dir.glob("*.png"))
                print(f"  {class_dir.name}: {len(images)} images")
            return True
        else:
            print("Data directory exists but is empty")
            return False
    else:
        print("Data directory does not exist")
        return False

def main():
    """Main setup function"""
    
    print("=" * 50)
    print("PLANT RECOGNITION DATA SETUP")
    print("=" * 50)
    
    # Check if data already exists
    if check_existing_data():
        print("\n✅ Data directory already contains images!")
        print("You can proceed with training.")
        return
    
    # Create data structure
    create_data_structure()
    
    print("\n" + "=" * 50)
    print("SETUP COMPLETE")
    print("=" * 50)

if __name__ == "__main__":
    main() 