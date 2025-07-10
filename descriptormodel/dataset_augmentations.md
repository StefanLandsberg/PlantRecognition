# Dataset and Augmentation System Documentation

## Overview

The dataset and augmentation system supports the two-model plant recognition architecture with both traditional image augmentation and advanced 50k feature chunking methods. The system manages invasive vs non-invasive plant classification datasets whilst providing flexible augmentation strategies for different training scenarios.

The system handles three main dataset types: invasive plant species (with specific names), non-invasive plants (labelled as "unknown"), and unlabelled plant images for feature extraction training.

## Dataset Architecture

### Dataset Types and Structure

#### Invasive Species Dataset
```
invasive_plants/
├── Acacia_dealbata/          # Invasive species with specific names
│   ├── image_001.jpg
│   ├── image_002.jpg
│   └── metadata.json
├── Lantana_camara/
└── ...
```

#### Non-Invasive Dataset
```
non_invasive_plants/
├── unknown_native_001.jpg    # All labelled as "unknown"
├── unknown_native_002.jpg
├── unknown_garden_001.jpg
└── metadata.json             # Contains source information
```

#### Unlabelled Training Dataset
```
feature_training_plants/
├── batch_001/               # Large collection for feature extractor training, just very basic, lables dont matter 
│   ├── plant_001.jpg
│   ├── plant_002.jpg
│   └── ...
└── batch_002/
```

### Dataset Configuration

```python
DATASET_CONFIG = {
    'invasive_species': {
        'data_path': 'data/invasive_plants/',
        'label_type': 'species_name',
        'augmentation_factor': 50,
        'quality_threshold': 0.8
    },
    
    'non_invasive': {
        'data_path': 'data/non_invasive_plants/',
        'label_type': 'unknown',
        'augmentation_factor': 30,
        'quality_threshold': 0.7
    },
    
    'feature_training': {
        'data_path': 'data/feature_training_plants/',
        'label_type': 'none',
        'augmentation_factor': 10,
        'quality_threshold': 0.6
    }
}
```

## Traditional Image Augmentation

### Basic Augmentation Pipeline

The traditional augmentation system applies image-level transformations to increase dataset variety whilst preserving botanical accuracy.

#### Geometric Transformations
- **Rotation**: ±15 degrees to simulate different viewing angles
- **Scaling**: 0.8x to 1.2x to handle distance variations
- **Translation**: ±10% to account for framing differences
- **Horizontal Flip**: 50% probability for natural variation
- **Perspective**: Minor adjustments for realistic camera angles

#### Photometric Adjustments
- **Brightness**: 0.7x to 1.3x to simulate lighting conditions
- **Contrast**: 0.8x to 1.2x for exposure variations
- **Saturation**: 0.9x to 1.1x for colour balance changes
- **Hue Shift**: ±5 degrees for seasonal colour variations

#### Environmental Simulation
- **Gaussian Noise**: 30% probability with low intensity
- **Motion Blur**: 20% probability to simulate camera movement
- **Weather Effects**: Rain, mist, and shadow simulation
- **Seasonal Changes**: Colour temperature adjustments

### Botanical-Specific Augmentation

#### Growth Stage Simulation
- **Leaf Density Variation**: Simulate different maturity stages
- **Seasonal Appearance**: Spring growth vs autumn colours
- **Flowering State**: With and without flowers/fruits
- **Damage Simulation**: Minor leaf damage or browning (optional)

#### Habitat Context
- **Background Variation**: Different soil, rock, and vegetation backgrounds
- **Lighting Conditions**: Sunny, shaded, dawn, dusk variations
- **Scale Simulation**: Close-up details vs whole plant views

### Augmentation Implementation

```python
class TraditionalAugmentation:
    def __init__(self, config):
        self.geometric_transforms = self._setup_geometric()
        self.photometric_transforms = self._setup_photometric()
        self.environmental_effects = self._setup_environmental()
        self.botanical_specific = self._setup_botanical()
    
    def augment_image(self, image, augmentation_type='balanced'):
        """Apply traditional augmentation to plant image"""
        
    def create_augmented_dataset(self, source_path, target_path, 
                               augmentations_per_image=30):
        """Generate augmented dataset from source images"""
```

## 50k Feature Chunking Augmentation (THIS IS FOR FUTURE DONT DO THIS NOW)

### Feature Extraction Architecture

The feature chunking system operates on learned 50k feature vectors rather than raw images, providing analytical diversity without information loss.

#### Feature Extractor Model
```python
class PlantFeatureExtractor:
    """Neural network that extracts 50k features from plant images"""
    
    def __init__(self, backbone='efficientnet_b4'):
        self.backbone = self._build_backbone(backbone)
        self.feature_head = self._build_feature_head(output_dim=50000)
    
    def extract_features(self, image):
        """Extract 50k feature vector from plant image"""
        return features  # Shape: (50000,)
```

#### Chunking Strategy

The system creates multiple feature perspectives from single 50k feature vectors:

```python
CHUNKING_CONFIG = {
    'chunk_size': 1500,
    'chunks_per_sample': 33,
    'chunking_strategies': {
        'random': 0.3,           # 30% random chunks
        'semantic': 0.4,         # 40% semantic groupings
        'learned': 0.3           # 30% learned selection
    },
    'quality_threshold': 0.75
}
```

### Chunking Methods

#### Random Chunking
- Select random 1,500 features from 50k vector
- Multiple random chunks per image for variety
- Fast generation with good coverage

#### Semantic Chunking
- Group features by learned semantic meaning
- Create chunks representing different plant aspects
- Maintain feature relationships within chunks

#### Learned Chunking
- Train small network to select optimal feature subsets
- Adapt selection based on plant characteristics
- Dynamic chunking for different species types

### Feature Chunking Implementation

```python
class FeatureChunkingAugmentation:
    def __init__(self, feature_extractor, chunking_config):
        self.feature_extractor = feature_extractor
        self.chunk_size = chunking_config['chunk_size']
        self.chunks_per_sample = chunking_config['chunks_per_sample']
        
    def create_feature_chunks(self, image):
        """Create multiple feature chunks from single image"""
        # Extract 50k features
        features = self.feature_extractor.extract_features(image)
        
        # Generate chunks using different strategies
        chunks = []
        for strategy in self.chunking_strategies:
            chunk = self._apply_chunking_strategy(features, strategy)
            chunks.append(chunk)
            
        return chunks
    
    def cache_feature_chunks(self, dataset_path, cache_path):
        """Pre-compute and cache feature chunks for dataset"""
```

## Dataset Management Classes

### Invasive Species Dataset Loader

```python
class InvasiveSpeciesDataset:
    def __init__(self, data_path, augmentation_config):
        self.invasive_species = self._load_invasive_species(data_path)
        self.non_invasive_samples = self._load_non_invasive(data_path)
        self.augmentation = self._setup_augmentation(augmentation_config)
    
    def __getitem__(self, idx):
        """Return sample with appropriate label (species name or 'unknown')"""
        
    def get_balanced_batch(self, batch_size):
        """Return balanced batch of invasive and non-invasive samples"""
```

### Two-Model Training Dataset

```python
class TwoModelDataset:
    """Dataset for training the two-model architecture"""
    
    def __init__(self, feature_cache_path, classification_data_path):
        self.feature_cache = self._load_feature_cache(feature_cache_path)
        self.classification_data = self._load_classification_data()
        
    def get_feature_training_batch(self, batch_size):
        """Return batch for training feature extractor"""
        
    def get_classification_batch(self, batch_size, chunk_strategy='mixed'):
        """Return feature chunks with labels for classifier training"""
```

## Training Pipeline Integration

### Stage 1: Feature Extractor Training(THIS IS FOR FUTURE DONT DO IT NOW)

```python
def train_feature_extractor(dataset_path, model_config):
    """Train the 50k feature extraction model"""
    
    # Load unlabelled plant images
    training_dataset = UnlabelledPlantDataset(dataset_path)
    
    # Apply traditional augmentation
    augmented_dataset = apply_traditional_augmentation(training_dataset)
    
    # Train feature extractor (self-supervised or knowledge distillation)
    feature_extractor = train_extractor_model(augmented_dataset, model_config)
    
    return feature_extractor
```

### Stage 2: Classifier Training with Chunking

```python
def train_chunk_classifier(feature_extractor, invasive_dataset_path):
    """Train classifier using feature chunks"""
    
    # Load invasive/non-invasive datasets
    dataset = InvasiveSpeciesDataset(invasive_dataset_path)
    
    # Extract features for all images
    feature_cache = extract_all_features(dataset, feature_extractor)
    
    # Create chunked training data
    chunked_dataset = create_chunked_dataset(feature_cache, chunk_config)
    
    # Train lightweight classifier
    classifier = train_classifier_model(chunked_dataset)
    
    return classifier
```

## Data Quality and Validation

### Image Quality Assessment

yes

### Label Validation

yes

## Configuration Management

### Complete Configuration Example

```python
AUGMENTATION_SYSTEM_CONFIG = {
    'traditional_augmentation': {
        'geometric': {
            'rotation_range': (-15, 15),
            'scale_range': (0.8, 1.2),
            'translation_range': 0.1,
            'flip_probability': 0.5
        },
        'photometric': {
            'brightness_range': (0.7, 1.3),
            'contrast_range': (0.8, 1.2),
            'saturation_range': (0.9, 1.1),
            'hue_shift_range': (-5, 5)
        },
        'environmental': {
            'noise_probability': 0.3,
            'blur_probability': 0.2,
            'weather_effects': True
        }
    },
    
    'feature_chunking': {
        'feature_extractor': {
            'backbone': 'efficientnet_b4',
            'output_features': 50000,
            'training_method': 'knowledge_distillation'
        },
        'chunking': {
            'chunk_size': 1500,
            'chunks_per_image': 33,
            'strategies': ['random', 'semantic', 'learned'],
            'cache_enabled': True
        }
    },
    
    'dataset_management': {
        'invasive_species': {
            'augmentation_factor': 50,
            'validation_split': 0.2
        },
        'non_invasive': {
            'augmentation_factor': 30,
            'balance_ratio': 1.0
        },
        'quality_control': {
            'blur_threshold': 0.3,
            'size_minimum': (256, 256),
            'brightness_range': (20, 235)
        }
    }
}
```

## Performance Optimisation

### Caching Strategy

- **Feature Cache**: Store 50k features to avoid recomputation
- **Chunk Cache**: Pre-compute popular chunk combinations
- **Augmentation Cache**: Store frequently used augmented images
- **Metadata Cache**: Quick access to dataset statistics

### Memory Management

- **Batch Processing**: Process images in manageable batches
- **Lazy Loading**: Load data only when needed
- **Memory Mapping**: Use memory-mapped files for large datasets
- **Garbage Collection**: Explicit cleanup of temporary data

### Parallel Processing

- **Multi-Threading**: Parallel augmentation and feature extraction
- **GPU Acceleration**: Use GPU for feature extraction and chunking
- **Distributed Processing**: Scale across multiple machines if needed

## Deployment Considerations

### Production Dataset Pipeline

- **Automated Quality Control**: Real-time image quality assessment
- **Incremental Updates**: Add new species without full retraining
- **Version Control**: Track dataset versions and changes
- **Backup Systems**: Redundant storage for critical datasets

### Field Data Collection (THIS IS FOR FUTURE DONT DO IT NOW)

- **Mobile Integration**: Support for smartphone data collection
- **GPS Tagging**: Location metadata for ecological context
- **Citizen Science**: Integration with public data collection efforts
- **Quality Feedback**: User validation of classifications
