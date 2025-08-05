DATA_DIR = "data/plant_images"
IMAGE_SIZE = 224
BATCH_SIZE = 8
EPOCHS = 100
AUGMENT_COUNT = 30
VAL_SPLIT = 0.2
PATIENCE = 8
LR = 2e-3

# Augmentation split: percentage of realistic vs seasonal/edge
AUGMENT_REALISTIC_FRAC = 0.6  # 60% realistic (lighting, angles, basic, user_realistic)
AUGMENT_SEASONAL_FRAC = 0.4   # 40% seasonal/edge (seasonal, plant, weather)