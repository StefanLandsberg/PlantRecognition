<<<<<<< HEAD
import os
import pandas as pd
import time
import gc
import asyncio
import aiohttp
import signal
import json
import psutil
import threading
import gzip
import csv
import random
from tqdm.auto import tqdm
import multiprocessing
from concurrent.futures import ThreadPoolExecutor

# ===== CONFIGURATION =====
# Update paths to be more robust while still relative
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))  # Get directory where script is located
BASE_DIR = os.path.dirname(SCRIPT_DIR)  # Parent directory of Scripts folder
OUTPUT_DIR = os.path.join(BASE_DIR, "data", "plant_images")
PARQUET_FILE = os.path.join(BASE_DIR, "cc0_photos.parquet")
CSV_FILE = os.path.join(BASE_DIR, "data", "inaturalist_metadata", "photos.csv.gz")
METADATA_DIR = os.path.join(BASE_DIR, "data", "inaturalist_metadata")
# =========================
MAX_IMAGES_PER_OBSERVATION = 10         # Maximum images per observation
MAX_OBSERVATIONS = 7000                 # Maximum number of observations to download
IMAGE_QUALITY = "medium"                # Options: "original", "large", "medium", "small"
BATCH_SIZE = 500                        # Chunk size for processing
DOWNLOAD_THREADS = 150                  # Number of concurrent downloads
MEMORY_LIMIT_PERCENT = 75               # Memory usage limit, clear memory when exceeded
# =========================

# Terminal colors for better output readability
class Colors:
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BLUE = '\033[94m'
    ENDC = '\033[0m'

# Graceful termination handler
class GracefulTerminator:
    def __init__(self):
        self.terminate = False
        signal.signal(signal.SIGINT, self._signal_handler)
    
    def _signal_handler(self, sig, frame):
        """Handle Ctrl+C gracefully"""
        print(f"\n{Colors.YELLOW}Termination requested. Will stop after current batch.{Colors.ENDC}")
        self.terminate = True
    
    def should_terminate(self):
        return self.terminate

# Memory monitoring thread
def monitor_memory(limit_percent=MEMORY_LIMIT_PERCENT):
    """Background thread to monitor and manage memory usage"""
    def memory_monitor():
        print(f"{Colors.BLUE}Starting memory monitor (limit: {limit_percent}%){Colors.ENDC}")
        while True:
            try:
                mem = psutil.virtual_memory()
                if mem.percent > limit_percent:
                    print(f"{Colors.YELLOW}High memory usage: {mem.percent}% - Cleaning...{Colors.ENDC}")
                    gc.collect()
                time.sleep(10 if mem.percent > limit_percent-10 else 30)
            except Exception as e:
                print(f"{Colors.RED}Memory monitor error: {e}{Colors.ENDC}")
                time.sleep(60)
    
    thread = threading.Thread(target=memory_monitor, daemon=True)
    thread.start()
    return thread

# Clean memory function
def clean_memory():
    """Force garbage collection"""
    gc.collect()

# Checkpoint management for resumable downloads
class DownloadProgress:
    def __init__(self, checkpoint_file="download_checkpoint.json"):
        self.checkpoint_file = checkpoint_file
        self.completed_observations = []
        self.downloaded_images = {}
        self.current_observation = None
        self.load_checkpoint()
    
    def load_checkpoint(self):
        """Load previous download progress"""
        if os.path.exists(self.checkpoint_file):
            try:
                with open(self.checkpoint_file, 'r') as f:
                    data = json.load(f)
                    self.completed_observations = data.get('completed_observations', [])
                    self.downloaded_images = data.get('downloaded_images', {})
                    self.current_observation = data.get('current_observation')
                print(f"{Colors.GREEN}Resuming from previous download: {len(self.completed_observations)} observations completed{Colors.ENDC}")
            except Exception as e:
                print(f"{Colors.YELLOW}Error loading checkpoint: {e}{Colors.ENDC}")
    
    def save_checkpoint(self):
        """Save current download progress"""
        data = {
            'completed_observations': self.completed_observations,
            'downloaded_images': self.downloaded_images,
            'current_observation': self.current_observation,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        }
        try:
            with open(self.checkpoint_file, 'w') as f:
                json.dump(data, f)
        except Exception as e:
            print(f"{Colors.YELLOW}Error saving checkpoint: {e}{Colors.ENDC}")
    
    def observation_completed(self, observation_id):
        """Check if observation is already completed"""
        return observation_id in self.completed_observations
    
    def image_downloaded(self, observation_id, image_id):
        """Check if specific image is already downloaded"""
        return observation_id in self.downloaded_images and image_id in self.downloaded_images[observation_id]
    
    def mark_observation_complete(self, observation_id):
        """Mark an observation as completely processed"""
        if observation_id not in self.completed_observations:
            self.completed_observations.append(observation_id)
        self.current_observation = None
        self.save_checkpoint()
    
    def mark_observation_current(self, observation_id):
        """Mark which observation is currently being processed"""
        self.current_observation = observation_id
        self.save_checkpoint()
    
    def mark_image_downloaded(self, observation_id, image_id):
        """Record a successfully downloaded image"""
        if observation_id not in self.downloaded_images:
            self.downloaded_images[observation_id] = []
        
        if image_id not in self.downloaded_images[observation_id]:
            self.downloaded_images[observation_id].append(image_id)
        
        # Save checkpoint occasionally (not for every image)
        if len(self.downloaded_images[observation_id]) % 20 == 0:
            self.save_checkpoint()

# Function to extract CC0 photos from the original CSV 
def create_cc0_parquet(csv_file, parquet_file, chunk_size=100000):
    """Process the CSV file to extract CC0 licensed photos and save to parquet"""
    print(f"{Colors.BLUE}Creating CC0 photos parquet file from CSV...{Colors.ENDC}")
    
    cc0_photos = []
    
    try:
        # Open the gzipped CSV file
        with gzip.open(csv_file, 'rt') as f:
            # Get header
            header = f.readline().strip().split('\t')
            
            # Find license column
            license_col_idx = None
            for i, col in enumerate(header):
                if 'license' in col.lower():
                    license_col_idx = i
                    break
            
            if license_col_idx is None:
                print(f"{Colors.RED}Error: Cannot find license column in CSV{Colors.ENDC}")
                return False
            
            # Find photo ID column
            id_col_idx = None
            for i, col in enumerate(header):
                if col.lower() == 'id' or col.lower() == 'photo_id':
                    id_col_idx = i
                    break
            
            if id_col_idx is None:
                print(f"{Colors.RED}Error: Cannot find photo ID column in CSV{Colors.ENDC}")
                return False
            
            # Find observation ID column
            obs_col_idx = None
            for i, col in enumerate(header):
                if 'observation' in col.lower() and 'id' in col.lower():
                    obs_col_idx = i
                    break
            
            if obs_col_idx is None:
                print(f"{Colors.RED}Error: Cannot find observation ID column in CSV{Colors.ENDC}")
                return False
            
            # Process file in chunks
            processed = 0
            cc0_count = 0
            
            with tqdm(desc="Processing CSV", unit="lines") as pbar:
                while True:
                    chunk_data = []
                    
                    # Read chunk of lines
                    for _ in range(chunk_size):
                        line = f.readline()
                        if not line:
                            break
                        
                        parts = line.strip().split('\t')
                        if len(parts) <= max(id_col_idx, license_col_idx, obs_col_idx):
                            continue
                        
                        # Check if license contains CC0
                        if 'CC0' in parts[license_col_idx].upper():
                            chunk_data.append({
                                'photo_id': parts[id_col_idx],
                                'observation_id': parts[obs_col_idx]
                            })
                    
                    if not chunk_data:
                        break
                    
                    # Add to CC0 photos
                    cc0_photos.extend(chunk_data)
                    cc0_count += len(chunk_data)
                    processed += chunk_size
                    pbar.update(chunk_size)
                    pbar.set_description(f"Processing CSV ({cc0_count} CC0 photos found)")
                    
                    # Clean memory regularly
                    if len(cc0_photos) > 500000:  # Save in smaller chunks to avoid memory issues
                        temp_df = pd.DataFrame(cc0_photos)
                        temp_df.to_parquet(f"temp_{processed}.parquet")
                        cc0_photos = []
                        clean_memory()
        
        # Combine all chunks if needed
        if os.path.exists(f"temp_{processed}.parquet"):
            print(f"{Colors.BLUE}Combining temporary parquet files...{Colors.ENDC}")
            all_cc0 = []
            for temp_file in sorted([f for f in os.listdir('.') if f.startswith('temp_') and f.endswith('.parquet')]):
                chunk = pd.read_parquet(temp_file)
                all_cc0.append(chunk)
                os.remove(temp_file)
            
            cc0_df = pd.concat(all_cc0) if all_cc0 else pd.DataFrame(cc0_photos)
        else:
            cc0_df = pd.DataFrame(cc0_photos)
        
        # Save final parquet file
        cc0_df.to_parquet(parquet_file)
        print(f"{Colors.GREEN}Created {parquet_file} with {len(cc0_df)} CC0 photos{Colors.ENDC}")
        return True
    
    except Exception as e:
        print(f"{Colors.RED}Error creating CC0 parquet: {str(e)}{Colors.ENDC}")
        return False

# Optimized async download functions
async def download_image(url, path, session):
    """Download an image using async I/O"""
    try:
        async with session.get(url, timeout=15) as response:
            if response.status == 200:
                data = await response.read()
                with open(path, 'wb') as f:
                    f.write(data)
                return True
            return False
    except Exception:
        return False

async def download_batch(urls_paths, max_concurrent=DOWNLOAD_THREADS):
    """Download a batch of images concurrently"""
    connector = aiohttp.TCPConnector(limit=max_concurrent, ttl_dns_cache=300)
    timeout = aiohttp.ClientTimeout(total=30)
    async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
        tasks = []
        for url, path in urls_paths:
            tasks.append(asyncio.ensure_future(
                download_image(url, path, session)
            ))
        return await asyncio.gather(*tasks)

def download_images_async(urls_paths):
    """Run the async download with proper event loop handling"""
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    
    return loop.run_until_complete(download_batch(urls_paths))

def create_direct_download_list(cc0_photos_df):
    """Create a direct download list without species information"""
    print(f"{Colors.BLUE}Creating direct download list without species mapping...{Colors.ENDC}")
    
    # We'll organize photos by observation_id instead of species
    observation_groups = {}
    
    with tqdm(total=len(cc0_photos_df), desc="Grouping photos by observation") as pbar:
        # Process in chunks to avoid memory issues
        for start_idx in range(0, len(cc0_photos_df), BATCH_SIZE):
            end_idx = min(start_idx + BATCH_SIZE, len(cc0_photos_df))
            batch = cc0_photos_df.iloc[start_idx:end_idx]
            
            for _, row in batch.iterrows():
                observation_id = str(row['observation_id'])
                photo_id = str(row['photo_id'])
                
                # Group by observation
                if observation_id not in observation_groups:
                    observation_groups[observation_id] = []
                
                observation_groups[observation_id].append(photo_id)
            
            pbar.update(len(batch))
    
    # Select a maximum number of observations
    if len(observation_groups) > MAX_OBSERVATIONS:
        selected_observations = random.sample(list(observation_groups.keys()), MAX_OBSERVATIONS)
        filtered_groups = {obs_id: observation_groups[obs_id] for obs_id in selected_observations}
        observation_groups = filtered_groups
    
    print(f"{Colors.GREEN}Grouped photos by {len(observation_groups)} observations{Colors.ENDC}")
    
    # Create download list
    download_list = []
    
    for obs_id, photo_ids in observation_groups.items():
        # Create observation directory
        obs_dir = os.path.join(OUTPUT_DIR, f"observation_{obs_id}")
        os.makedirs(obs_dir, exist_ok=True)
        
        # Limit photos per observation
        if len(photo_ids) > MAX_IMAGES_PER_OBSERVATION:
            photo_ids = photo_ids[:MAX_IMAGES_PER_OBSERVATION]
        
        # Add to download list
        for photo_id in photo_ids:
            url = f"https://inaturalist-open-data.s3.amazonaws.com/photos/{photo_id}/{IMAGE_QUALITY}.jpg"
            output_path = os.path.join(obs_dir, f"{photo_id}.jpg")
            
            if not os.path.exists(output_path):
                download_list.append((url, output_path, photo_id, obs_id))
    
    print(f"{Colors.GREEN}Created download list with {len(download_list)} images{Colors.ENDC}")
    return download_list

def download_direct(download_list, progress, terminator):
    """Download images without species information"""
    print(f"{Colors.BLUE}Starting direct downloads for {len(download_list)} images{Colors.ENDC}")
    
    # Create progress bar
    with tqdm(total=len(download_list), desc="Downloading images") as pbar:
        # Download in batches to manage memory
        batch_size = 200  # Larger batch size for direct downloads
        download_batches = [download_list[i:i+batch_size] for i in range(0, len(download_list), batch_size)]
        
        for batch in download_batches:
            # Check for termination
            if terminator.should_terminate():
                break
            
            # Prepare URLs and paths for async download
            urls_paths = [(item[0], item[1]) for item in batch]
            
            # Download batch using async I/O
            results = download_images_async(urls_paths)
            
            # Process results and track successful downloads
            for i, success in enumerate(results):
                if success and i < len(batch):
                    photo_id = batch[i][2]
                    obs_id = batch[i][3]
                    progress.mark_image_downloaded(obs_id, photo_id)
            
            # Update progress
            pbar.update(len(batch))
            
            # Clean memory after each batch
            clean_memory()

def download_by_species(download_list, species_groups, progress, terminator):
    """Download images organized by species"""
    print(f"{Colors.BLUE}Starting species-organized downloads for {len(download_list)} images{Colors.ENDC}")
    
    # Group download tasks by species for better organization
    downloads_by_species = {}
    for url, path, photo_id, species in download_list:
        if species not in downloads_by_species:
            downloads_by_species[species] = []
        downloads_by_species[species].append((url, path, photo_id))
    
    # Process one species at a time
    species_bar = tqdm(total=len(downloads_by_species), desc="Processing species")
    
    for species, species_downloads in downloads_by_species.items():
        # Check for termination
        if terminator.should_terminate():
            break
        
        # Create a nested progress bar for this species
        with tqdm(total=len(species_downloads), desc=f"Downloading {species[:25]}", leave=False) as download_bar:
            # Download in batches to manage memory
            batch_size = 200
            download_batches = [species_downloads[i:i+batch_size] for i in range(0, len(species_downloads), batch_size)]
            
            for batch in download_batches:
                # Check for termination
                if terminator.should_terminate():
                    break
                
                # Prepare URLs and paths for async download
                urls_paths = [(item[0], item[1]) for item in batch]
                
                # Download batch using async I/O
                results = download_images_async(urls_paths)
                
                # Process results
                for i, success in enumerate(results):
                    if success and i < len(batch):
                        photo_id = batch[i][2]
                        progress.mark_image_downloaded(species, photo_id)
                
                download_bar.update(len(batch))
                
                # Clean memory after each batch
                clean_memory()
        
        # Mark species as complete
        progress.mark_observation_complete(species)
        species_bar.update(1)
    
    species_bar.close()

def download_cc0_plant_images():
    """Main function to download CC0 plant images - Plants only, organized by species"""
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Start memory monitoring
    monitor_thread = monitor_memory()
    
    # Setup terminator
    terminator = GracefulTerminator()
    
    # Initialize download progress tracker
    progress = DownloadProgress()
    
    # Check for parquet file
    if not os.path.exists(PARQUET_FILE):
        if os.path.exists(CSV_FILE):
            # Create parquet file from CSV
            success = create_cc0_parquet(CSV_FILE, PARQUET_FILE)
            if not success:
                print(f"{Colors.RED}Failed to create CC0 parquet file. Exiting.{Colors.ENDC}")
                return
        else:
            print(f"{Colors.RED}Error: Neither {PARQUET_FILE} nor {CSV_FILE} found{Colors.ENDC}")
            return
    
    # Load CC0 photos
    print(f"{Colors.BLUE}Loading CC0 photos from parquet...{Colors.ENDC}")
    cc0_photos_df = pd.read_parquet(PARQUET_FILE)
    print(f"{Colors.GREEN}Loaded {len(cc0_photos_df)} CC0 photos{Colors.ENDC}")
    
    observations_file = os.path.join(METADATA_DIR, "observations.csv.gz")
    taxa_file = os.path.join(METADATA_DIR, "taxa.csv.gz")
    
    if not os.path.exists(observations_file) or not os.path.exists(taxa_file):
        print(f"{Colors.RED}Error: Missing necessary CSV files for taxonomy data{Colors.ENDC}")
        print(f"{Colors.YELLOW}Falling back to downloading all CC0 photos without filtering for plants.{Colors.ENDC}")
        download_list = create_direct_download_list(cc0_photos_df)
        download_direct(download_list, progress, terminator)
        return
    
    # Extract taxa data - build a mapping from taxon_id to name
    print(f"{Colors.BLUE}Extracting taxa names from taxa.csv.gz...{Colors.ENDC}")
    taxon_name_map = {}
    
    try:
        with gzip.open(taxa_file, 'rt') as f:
            reader = csv.reader(f, delimiter='\t')
            header = next(reader)
            
            # Find required columns
            taxon_id_idx = header.index('taxon_id') if 'taxon_id' in header else None
            name_idx = header.index('name') if 'name' in header else None
            rank_idx = header.index('rank') if 'rank' in header else None
            ancestry_idx = header.index('ancestry') if 'ancestry' in header else None
            
            if None in (taxon_id_idx, name_idx):
                print(f"{Colors.RED}Missing required columns in taxa file{Colors.ENDC}")
                print(f"{Colors.YELLOW}Falling back to downloading all CC0 photos.{Colors.ENDC}")
                download_list = create_direct_download_list(cc0_photos_df)
                download_direct(download_list, progress, terminator)
                return
            
            # Read all taxa data
            for row in tqdm(reader, desc="Reading taxa data"):
                try:
                    if len(row) > max(taxon_id_idx, name_idx):
                        taxon_id = row[taxon_id_idx].strip()
                        name = row[name_idx].strip()
                        
                        # Get rank if available
                        rank = None
                        if rank_idx is not None and rank_idx < len(row):
                            rank = row[rank_idx].strip()
                        
                        # Store taxon name
                        taxon_name_map[taxon_id] = name
                except (IndexError, ValueError):
                    continue
        
        print(f"{Colors.GREEN}Found {len(taxon_name_map)} taxa entries{Colors.ENDC}")
        
        # Now map observations to taxa
        print(f"{Colors.BLUE}Mapping observations to taxa...{Colors.ENDC}")
        observation_to_species = {}
        
        with gzip.open(observations_file, 'rt') as f:
            reader = csv.reader(f, delimiter='\t')
            header = next(reader)
            
            # Find required columns
            obs_id_idx = header.index('observation_uuid') if 'observation_uuid' in header else None
            taxon_id_idx = header.index('taxon_id') if 'taxon_id' in header else None
            
            if None in (obs_id_idx, taxon_id_idx):
                print(f"{Colors.RED}Missing required columns in observations file{Colors.ENDC}")
                print(f"{Colors.YELLOW}Falling back to downloading all CC0 photos.{Colors.ENDC}")
                download_list = create_direct_download_list(cc0_photos_df)
                download_direct(download_list, progress, terminator)
                return
            
            # Map observations to taxa
            for row in tqdm(reader, desc="Mapping observations to taxa"):
                try:
                    # Ensure we have safe indices
                    safe_obs_id_idx = 0 if obs_id_idx is None else obs_id_idx
                    safe_taxon_id_idx = 0 if taxon_id_idx is None else taxon_id_idx
                    
                    if len(row) > max(safe_obs_id_idx, safe_taxon_id_idx):
                        obs_id = row[safe_obs_id_idx].strip()
                        taxon_id = row[safe_taxon_id_idx].strip()
                        
                        # Only include taxa we have names for
                        if taxon_id in taxon_name_map:
                            # Get species name from taxon
                            species_name = taxon_name_map[taxon_id]
                            
                            # Clean species name for filesystem
                            clean_species_name = species_name.replace(' ', '_').replace('/', '_').replace('\\', '_')
                            clean_species_name = ''.join(c for c in clean_species_name if c.isalnum() or c == '_')
                            
                            # Ensure we have a valid name
                            if not clean_species_name:
                                clean_species_name = f"Taxon_{taxon_id}"
                            
                            observation_to_species[obs_id] = clean_species_name
                except (IndexError, ValueError):
                    continue
        
        print(f"{Colors.GREEN}Mapped {len(observation_to_species)} observations to taxa{Colors.ENDC}")
        
        # Now group CC0 photos by species
        print(f"{Colors.BLUE}Grouping CC0 photos by species...{Colors.ENDC}")
        species_groups = {}
        photos_count = 0
        
        with tqdm(total=len(cc0_photos_df), desc="Grouping photos by species") as pbar:
            # Process in chunks to avoid memory issues
            for start_idx in range(0, len(cc0_photos_df), BATCH_SIZE):
                end_idx = min(start_idx + BATCH_SIZE, len(cc0_photos_df))
                batch = cc0_photos_df.iloc[start_idx:end_idx]
                
                for _, row in batch.iterrows():
                    observation_id = str(row['observation_id'])
                    photo_id = str(row['photo_id'])
                    
                    # Check if we have species info for this observation
                    if observation_id in observation_to_species:
                        species_name = observation_to_species[observation_id]
                        
                        # Group by species
                        if species_name not in species_groups:
                            species_groups[species_name] = []
                        
                        species_groups[species_name].append((photo_id, observation_id))
                        photos_count += 1
                
                pbar.update(len(batch))
        
        print(f"{Colors.GREEN}Grouped {photos_count} photos across {len(species_groups)} species{Colors.ENDC}")
        
        # Filter species with too few images and limit max images per species
        filtered_species_groups = {}
        MIN_IMAGES_PER_SPECIES = 10  # Minimum images per species
        MAX_IMAGES_PER_SPECIES = 250  # Maximum images per species
        
        for species, photos in species_groups.items():
            if len(photos) >= MIN_IMAGES_PER_SPECIES:
                if len(photos) > MAX_IMAGES_PER_SPECIES:
                    # Randomly select MAX_IMAGES_PER_SPECIES photos
                    random.seed(42)  # For reproducibility
                    photos = random.sample(photos, MAX_IMAGES_PER_SPECIES)
                
                filtered_species_groups[species] = photos
        
        print(f"{Colors.GREEN}Filtered to {len(filtered_species_groups)} species with at least {MIN_IMAGES_PER_SPECIES} images{Colors.ENDC}")
        
        # Create download list organized by species
        download_list = []
        
        for species_name, photos in filtered_species_groups.items():
            # Create species directory
            species_dir = os.path.join(OUTPUT_DIR, species_name)
            os.makedirs(species_dir, exist_ok=True)
            
            # Add to download list
            for photo_id, observation_id in photos:
                url = f"https://inaturalist-open-data.s3.amazonaws.com/photos/{photo_id}/{IMAGE_QUALITY}.jpg"
                output_path = os.path.join(species_dir, f"{photo_id}.jpg")
                
                if not os.path.exists(output_path):
                    download_list.append((url, output_path, photo_id, species_name))
        
        print(f"{Colors.GREEN}Created download list with {len(download_list)} images{Colors.ENDC}")
        
        # Download images by species
        print(f"{Colors.BLUE}Starting downloads for {len(filtered_species_groups)} species{Colors.ENDC}")
        download_by_species(download_list, filtered_species_groups, progress, terminator)
        
        # Create summary
        print(f"{Colors.BLUE}Creating dataset summary...{Colors.ENDC}")
        total_images = 0
        species_count = 0
        
        for species_dir in os.listdir(OUTPUT_DIR):
            full_path = os.path.join(OUTPUT_DIR, species_dir)
            if os.path.isdir(full_path) and not species_dir.startswith('observation_'):
                image_count = len([f for f in os.listdir(full_path) if f.endswith('.jpg')])
                if image_count > 0:
                    total_images += image_count
                    species_count += 1
        
        print(f"{Colors.GREEN}Download complete! Downloaded {total_images} images across {species_count} species{Colors.ENDC}")
        
    except Exception as e:
        print(f"{Colors.RED}Error processing taxonomy data: {e}{Colors.ENDC}")
        import traceback
        traceback.print_exc()
        print(f"{Colors.YELLOW}Falling back to downloading all CC0 photos without filtering.{Colors.ENDC}")
        download_list = create_direct_download_list(cc0_photos_df)
        download_direct(download_list, progress, terminator)

def download_from_csv(csv_file_path):
    """Download images from a CSV file with species information already included"""
    print(f"{Colors.BLUE}Loading observations from CSV file...{Colors.ENDC}")
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Start memory monitoring
    monitor_thread = monitor_memory()
    
    # Setup terminator
    terminator = GracefulTerminator()
    
    # Initialize download progress tracker
    progress = DownloadProgress()
    
    # Parse the CSV file
    try:
        # First, check if the file is gzipped
        is_gzipped = csv_file_path.endswith('.gz')
        if is_gzipped:
            open_func = gzip.open
            mode = 'rt'
        else:
            open_func = lambda file, mode: open(file, mode, encoding='utf-8', errors='replace')
            mode = 'r'
        
        # Read all data from CSV
        species_groups = {}
        photos_by_species = {}
        
        with open_func(csv_file_path, mode) as f:
            reader = csv.DictReader(f)
            
            # Process each row
            for row in tqdm(reader, desc="Processing observations"):
                try:
                    # Extract required fields
                    observation_id = row.get('id', '')
                    image_url = row.get('image_url', '')
                    scientific_name = row.get('scientific_name', '')
                    common_name = row.get('common_name', '')
                    iconic_taxon = row.get('iconic_taxon_name', '')
                    
                    # Skip if missing required data
                    if not observation_id or not image_url:
                        continue
                    
                    # Skip if not a plant (if you only want plants)
                    if iconic_taxon and iconic_taxon != 'Plantae':
                        continue
                    
                    # Create species name (prefer scientific, fall back to common)
                    species_name = scientific_name if scientific_name else common_name
                    if not species_name:
                        species_name = f"Unknown_Species_{observation_id}"
                    
                    # Clean species name for filesystem
                    clean_species_name = species_name.replace(' ', '_').replace('/', '_').replace('\\', '_')
                    clean_species_name = ''.join(c for c in clean_species_name if c.isalnum() or c == '_')
                    
                    # Extract photo ID from URL
                    # Format: https://inaturalist-open-data.s3.amazonaws.com/photos/169241/medium.JPG
                    # or: https://static.inaturalist.org/photos/60085/medium.jpg
                    photo_id = None
                    if 'inaturalist-open-data' in image_url:
                        parts = image_url.split('/')
                        if len(parts) >= 5:
                            photo_id = parts[-2]
                    elif 'static.inaturalist.org' in image_url:
                        parts = image_url.split('/')
                        if len(parts) >= 5:
                            photo_id = parts[-2]
                    
                    if not photo_id:
                        continue
                    
                    # Group by species
                    if clean_species_name not in photos_by_species:
                        photos_by_species[clean_species_name] = []
                    
                    # Store photo info
                    photos_by_species[clean_species_name].append((photo_id, observation_id))
                    
                except Exception as e:
                    print(f"{Colors.YELLOW}Error processing row: {e}{Colors.ENDC}")
                    continue
        
        print(f"{Colors.GREEN}Found {sum(len(photos) for photos in photos_by_species.values())} photos across {len(photos_by_species)} species{Colors.ENDC}")
        
        # Filter species with too few images and limit max images per species
        filtered_species_groups = {}
        MIN_IMAGES_PER_SPECIES = 5  # Lower threshold to include more species
        MAX_IMAGES_PER_SPECIES = 250  # Maximum images per species
        
        for species, photos in photos_by_species.items():
            if len(photos) >= MIN_IMAGES_PER_SPECIES:
                if len(photos) > MAX_IMAGES_PER_SPECIES:
                    # Randomly select MAX_IMAGES_PER_SPECIES photos
                    random.seed(42)  # For reproducibility
                    photos = random.sample(photos, MAX_IMAGES_PER_SPECIES)
                
                filtered_species_groups[species] = photos
        
        print(f"{Colors.GREEN}Filtered to {len(filtered_species_groups)} species with at least {MIN_IMAGES_PER_SPECIES} images{Colors.ENDC}")
        
        # Create download list organized by species
        download_list = []
        
        for species_name, photos in filtered_species_groups.items():
            # Create species directory
            species_dir = os.path.join(OUTPUT_DIR, species_name)
            os.makedirs(species_dir, exist_ok=True)
            
            # Add to download list
            for photo_id, observation_id in photos:
                url = f"https://inaturalist-open-data.s3.amazonaws.com/photos/{photo_id}/{IMAGE_QUALITY}.jpg"
                output_path = os.path.join(species_dir, f"{photo_id}.jpg")
                
                if not os.path.exists(output_path):
                    download_list.append((url, output_path, photo_id, species_name))
        
        print(f"{Colors.GREEN}Created download list with {len(download_list)} images{Colors.ENDC}")
        
        # Download images by species
        print(f"{Colors.BLUE}Starting downloads for {len(filtered_species_groups)} species{Colors.ENDC}")
        download_by_species(download_list, filtered_species_groups, progress, terminator)
        
        # Create summary
        print(f"{Colors.BLUE}Creating dataset summary...{Colors.ENDC}")
        total_images = 0
        species_count = 0
        
        for species_dir in os.listdir(OUTPUT_DIR):
            full_path = os.path.join(OUTPUT_DIR, species_dir)
            if os.path.isdir(full_path) and not species_dir.startswith('observation_'):
                image_count = len([f for f in os.listdir(full_path) if f.endswith('.jpg')])
                if image_count > 0:
                    total_images += image_count
                    species_count += 1
        
        print(f"{Colors.GREEN}Download complete! Downloaded {total_images} images across {species_count} species{Colors.ENDC}")
        
    except Exception as e:
        print(f"{Colors.RED}Error processing observations CSV: {e}{Colors.ENDC}")
        import traceback
        traceback.print_exc()

def debug_csv_columns(file_path):
    """Debug CSV file columns"""
    try:
        print(f"{Colors.BLUE}Debugging CSV file: {file_path}{Colors.ENDC}")
        import gzip
        with gzip.open(file_path, 'rt') as f:
            # Read the first line to get headers
            header_line = f.readline().strip()
            headers = header_line.split('\t')
            
            print(f"{Colors.BLUE}Found {len(headers)} columns:{Colors.ENDC}")
            for i, col in enumerate(headers):
                print(f"  {i}: '{col}'")
                
            # Check for known important columns
            kingdom_cols = [i for i, col in enumerate(headers) if 'kingdom' in col.lower()]
            if kingdom_cols:
                print(f"{Colors.GREEN}Potential kingdom columns: {kingdom_cols} - {[headers[i] for i in kingdom_cols]}{Colors.ENDC}")
            else:
                print(f"{Colors.RED}No kingdom column found!{Colors.ENDC}")
                
            # Check alternatives
            print(f"{Colors.BLUE}Looking for taxonomy columns:{Colors.ENDC}")
            for keyword in ['phylum', 'class', 'order', 'family', 'genus', 'taxon', 'taxonomy']:
                matching = [i for i, col in enumerate(headers) if keyword in col.lower()]
                if matching:
                    print(f"  '{keyword}' columns: {[headers[i] for i in matching]}")
    except Exception as e:
        print(f"{Colors.RED}Error debugging CSV: {e}{Colors.ENDC}")

def check_taxa_columns():
    """Simple function to check taxa file columns"""
    taxa_file = os.path.join(METADATA_DIR, "taxa.csv.gz")
    observations_file = os.path.join(METADATA_DIR, "observations.csv.gz")
    
    print(f"{Colors.BLUE}=================== TAXA FILE COLUMNS ==================={Colors.ENDC}")
    debug_csv_columns(taxa_file)
    
    print(f"\n{Colors.BLUE}=================== OBSERVATIONS FILE COLUMNS ==================={Colors.ENDC}")
    debug_csv_columns(observations_file)
    
    return

def analyze_and_recommend():
    """Analyze the downloaded dataset and provide recommendations for improvement"""
    print(f"{Colors.BLUE}=================== DATASET ANALYSIS AND RECOMMENDATIONS ==================={Colors.ENDC}")
    
    # Check if output directory exists
    if not os.path.exists(OUTPUT_DIR):
        print(f"{Colors.RED}No dataset found at {OUTPUT_DIR}. Please run the download first.{Colors.ENDC}")
        return
    
    # Collect statistics
    total_folders = 0
    total_images = 0
    observation_folders = 0
    species_folders = 0
    folder_sizes = {}
    small_folders = []
    large_folders = []
    
    print(f"{Colors.BLUE}Analyzing downloaded dataset structure...{Colors.ENDC}")
    
    # Analyze folder structure
    for folder_name in os.listdir(OUTPUT_DIR):
        folder_path = os.path.join(OUTPUT_DIR, folder_name)
        if os.path.isdir(folder_path):
            total_folders += 1
            
            # Count images in folder
            image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            image_count = len(image_files)
            total_images += image_count
            folder_sizes[folder_name] = image_count
            
            # Classify folder type
            if folder_name.startswith('observation_'):
                observation_folders += 1
            else:
                species_folders += 1
            
            # Track small and large folders
            if 0 < image_count < 10:
                small_folders.append((folder_name, image_count))
            elif image_count > 200:
                large_folders.append((folder_name, image_count))
    
    # Sort folders by size
    sorted_folders = sorted(folder_sizes.items(), key=lambda x: x[1], reverse=True)
    
    # Calculate balanced dataset metrics
    if total_folders > 0:
        avg_images_per_folder = total_images / total_folders
        if len(folder_sizes) > 0:
            median_size = list(sorted(folder_sizes.values()))[len(folder_sizes)//2]
            min_size = min(folder_sizes.values()) if folder_sizes else 0
            max_size = max(folder_sizes.values()) if folder_sizes else 0
        else:
            median_size = min_size = max_size = 0
    else:
        avg_images_per_folder = median_size = min_size = max_size = 0
    
    # Print analysis results
    print(f"{Colors.GREEN}Dataset Analysis Results:{Colors.ENDC}")
    print(f"  Total images: {total_images}")
    print(f"  Total folders: {total_folders}")
    print(f"  Observation-based folders: {observation_folders}")
    print(f"  Species-based folders: {species_folders}")
    print(f"  Average images per folder: {avg_images_per_folder:.1f}")
    print(f"  Median folder size: {median_size}")
    print(f"  Size range: {min_size} - {max_size} images per folder")
    print(f"  Small folders (<10 images): {len(small_folders)}")
    print(f"  Large folders (>200 images): {len(large_folders)}")
    
    # Generate recommendations
    print(f"\n{Colors.BLUE}Recommendations for Dataset Improvement:{Colors.ENDC}")
    
    if observation_folders > 0 and species_folders == 0:
        print(f"{Colors.YELLOW}1. Your dataset is organized by observation ID rather than species.{Colors.ENDC}")
        print("   This structure is suboptimal for plant recognition training.")
        print("   Recommendations:")
        print("   - Download a proper taxonomy dataset with species information")
        print("   - Use a pre-trained model to categorize images into species")
        print("   - Manually sort a subset of images into species folders")
        print("   - Try: https://www.inaturalist.org/pages/developers for full taxonomy data")
    
    if species_folders > 0:
        # If we have species data but dataset is imbalanced
        if len(small_folders) > 0.3 * total_folders:
            print(f"{Colors.YELLOW}2. Your dataset has many species with too few images.{Colors.ENDC}")
            print(f"   {len(small_folders)} species have fewer than 10 images.")
            print("   Recommendations:")
            print("   - Increase MIN_IMAGES_PER_SPECIES (currently 10)")
            print("   - Focus on species with adequate samples (delete or merge small classes)")
            print("   - Use data augmentation for underrepresented species")
        
        if len(large_folders) > 0.1 * total_folders:
            print(f"{Colors.YELLOW}3. Your dataset has some species with very many images.{Colors.ENDC}")
            print(f"   {len(large_folders)} species have more than 200 images.")
            print("   Recommendations:")
            print("   - Decrease MAX_IMAGES_PER_SPECIES (currently 250) for more balance")
            print("   - Use weighted sampling during training to account for class imbalance")
    
    # Check overall dataset size for training
    if total_images < 5000:
        print(f"{Colors.YELLOW}4. Your dataset may be too small for robust training.{Colors.ENDC}")
        print(f"   Current size: {total_images} images across {total_folders} folders")
        print("   Recommendations:")
        print("   - Increase MAX_OBSERVATIONS (currently 7000)")
        print("   - Download images from additional sources")
        print("   - Use transfer learning with pretrained weights")
        print("   - Apply extensive data augmentation")

if __name__ == "__main__":
    # Select which function to run
    # 1. Check CSV columns
    # check_taxa_columns()
    
    # 2. Download using taxonomy approach from CC0 iNaturalist dataset
    multiprocessing.set_start_method('spawn', force=True)
    download_cc0_plant_images()
    
    # 3. Download directly from observations CSV with species info
    multiprocessing.set_start_method('spawn', force=True)
    csv_file_path = os.path.join(BASE_DIR, "data", "observations-561226.csv")
    download_from_csv(csv_file_path)
    
    # 4. Analyze downloaded dataset and get recommendations
=======
import os
import pandas as pd
import time
import gc
import asyncio
import aiohttp
import signal
import json
import psutil
import threading
import gzip
import csv
import random
from tqdm.auto import tqdm
import multiprocessing
from concurrent.futures import ThreadPoolExecutor

# ===== CONFIGURATION =====
# Update paths to be more robust while still relative
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))  # Get directory where script is located
BASE_DIR = os.path.dirname(SCRIPT_DIR)  # Parent directory of Scripts folder
OUTPUT_DIR = os.path.join(BASE_DIR, "data", "plant_images")
PARQUET_FILE = os.path.join(BASE_DIR, "cc0_photos.parquet")
CSV_FILE = os.path.join(BASE_DIR, "data", "inaturalist_metadata", "photos.csv.gz")
METADATA_DIR = os.path.join(BASE_DIR, "data", "inaturalist_metadata")
# =========================
MAX_IMAGES_PER_OBSERVATION = 10         # Maximum images per observation
MAX_OBSERVATIONS = 7000                 # Maximum number of observations to download
IMAGE_QUALITY = "medium"                # Options: "original", "large", "medium", "small"
BATCH_SIZE = 500                        # Chunk size for processing
DOWNLOAD_THREADS = 150                  # Number of concurrent downloads
MEMORY_LIMIT_PERCENT = 75               # Memory usage limit, clear memory when exceeded
# =========================

# Terminal colors for better output readability
class Colors:
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BLUE = '\033[94m'
    ENDC = '\033[0m'

# Graceful termination handler
class GracefulTerminator:
    def __init__(self):
        self.terminate = False
        signal.signal(signal.SIGINT, self._signal_handler)
    
    def _signal_handler(self, sig, frame):
        """Handle Ctrl+C gracefully"""
        print(f"\n{Colors.YELLOW}Termination requested. Will stop after current batch.{Colors.ENDC}")
        self.terminate = True
    
    def should_terminate(self):
        return self.terminate

# Memory monitoring thread
def monitor_memory(limit_percent=MEMORY_LIMIT_PERCENT):
    """Background thread to monitor and manage memory usage"""
    def memory_monitor():
        print(f"{Colors.BLUE}Starting memory monitor (limit: {limit_percent}%){Colors.ENDC}")
        while True:
            try:
                mem = psutil.virtual_memory()
                if mem.percent > limit_percent:
                    print(f"{Colors.YELLOW}High memory usage: {mem.percent}% - Cleaning...{Colors.ENDC}")
                    gc.collect()
                time.sleep(10 if mem.percent > limit_percent-10 else 30)
            except Exception as e:
                print(f"{Colors.RED}Memory monitor error: {e}{Colors.ENDC}")
                time.sleep(60)
    
    thread = threading.Thread(target=memory_monitor, daemon=True)
    thread.start()
    return thread

# Clean memory function
def clean_memory():
    """Force garbage collection"""
    gc.collect()

# Checkpoint management for resumable downloads
class DownloadProgress:
    def __init__(self, checkpoint_file="download_checkpoint.json"):
        self.checkpoint_file = checkpoint_file
        self.completed_observations = []
        self.downloaded_images = {}
        self.current_observation = None
        self.load_checkpoint()
    
    def load_checkpoint(self):
        """Load previous download progress"""
        if os.path.exists(self.checkpoint_file):
            try:
                with open(self.checkpoint_file, 'r') as f:
                    data = json.load(f)
                    self.completed_observations = data.get('completed_observations', [])
                    self.downloaded_images = data.get('downloaded_images', {})
                    self.current_observation = data.get('current_observation')
                print(f"{Colors.GREEN}Resuming from previous download: {len(self.completed_observations)} observations completed{Colors.ENDC}")
            except Exception as e:
                print(f"{Colors.YELLOW}Error loading checkpoint: {e}{Colors.ENDC}")
    
    def save_checkpoint(self):
        """Save current download progress"""
        data = {
            'completed_observations': self.completed_observations,
            'downloaded_images': self.downloaded_images,
            'current_observation': self.current_observation,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        }
        try:
            with open(self.checkpoint_file, 'w') as f:
                json.dump(data, f)
        except Exception as e:
            print(f"{Colors.YELLOW}Error saving checkpoint: {e}{Colors.ENDC}")
    
    def observation_completed(self, observation_id):
        """Check if observation is already completed"""
        return observation_id in self.completed_observations
    
    def image_downloaded(self, observation_id, image_id):
        """Check if specific image is already downloaded"""
        return observation_id in self.downloaded_images and image_id in self.downloaded_images[observation_id]
    
    def mark_observation_complete(self, observation_id):
        """Mark an observation as completely processed"""
        if observation_id not in self.completed_observations:
            self.completed_observations.append(observation_id)
        self.current_observation = None
        self.save_checkpoint()
    
    def mark_observation_current(self, observation_id):
        """Mark which observation is currently being processed"""
        self.current_observation = observation_id
        self.save_checkpoint()
    
    def mark_image_downloaded(self, observation_id, image_id):
        """Record a successfully downloaded image"""
        if observation_id not in self.downloaded_images:
            self.downloaded_images[observation_id] = []
        
        if image_id not in self.downloaded_images[observation_id]:
            self.downloaded_images[observation_id].append(image_id)
        
        # Save checkpoint occasionally (not for every image)
        if len(self.downloaded_images[observation_id]) % 20 == 0:
            self.save_checkpoint()

# Function to extract CC0 photos from the original CSV 
def create_cc0_parquet(csv_file, parquet_file, chunk_size=100000):
    """Process the CSV file to extract CC0 licensed photos and save to parquet"""
    print(f"{Colors.BLUE}Creating CC0 photos parquet file from CSV...{Colors.ENDC}")
    
    cc0_photos = []
    
    try:
        # Open the gzipped CSV file
        with gzip.open(csv_file, 'rt') as f:
            # Get header
            header = f.readline().strip().split('\t')
            
            # Find license column
            license_col_idx = None
            for i, col in enumerate(header):
                if 'license' in col.lower():
                    license_col_idx = i
                    break
            
            if license_col_idx is None:
                print(f"{Colors.RED}Error: Cannot find license column in CSV{Colors.ENDC}")
                return False
            
            # Find photo ID column
            id_col_idx = None
            for i, col in enumerate(header):
                if col.lower() == 'id' or col.lower() == 'photo_id':
                    id_col_idx = i
                    break
            
            if id_col_idx is None:
                print(f"{Colors.RED}Error: Cannot find photo ID column in CSV{Colors.ENDC}")
                return False
            
            # Find observation ID column
            obs_col_idx = None
            for i, col in enumerate(header):
                if 'observation' in col.lower() and 'id' in col.lower():
                    obs_col_idx = i
                    break
            
            if obs_col_idx is None:
                print(f"{Colors.RED}Error: Cannot find observation ID column in CSV{Colors.ENDC}")
                return False
            
            # Process file in chunks
            processed = 0
            cc0_count = 0
            
            with tqdm(desc="Processing CSV", unit="lines") as pbar:
                while True:
                    chunk_data = []
                    
                    # Read chunk of lines
                    for _ in range(chunk_size):
                        line = f.readline()
                        if not line:
                            break
                        
                        parts = line.strip().split('\t')
                        if len(parts) <= max(id_col_idx, license_col_idx, obs_col_idx):
                            continue
                        
                        # Check if license contains CC0
                        if 'CC0' in parts[license_col_idx].upper():
                            chunk_data.append({
                                'photo_id': parts[id_col_idx],
                                'observation_id': parts[obs_col_idx]
                            })
                    
                    if not chunk_data:
                        break
                    
                    # Add to CC0 photos
                    cc0_photos.extend(chunk_data)
                    cc0_count += len(chunk_data)
                    processed += chunk_size
                    pbar.update(chunk_size)
                    pbar.set_description(f"Processing CSV ({cc0_count} CC0 photos found)")
                    
                    # Clean memory regularly
                    if len(cc0_photos) > 500000:  # Save in smaller chunks to avoid memory issues
                        temp_df = pd.DataFrame(cc0_photos)
                        temp_df.to_parquet(f"temp_{processed}.parquet")
                        cc0_photos = []
                        clean_memory()
        
        # Combine all chunks if needed
        if os.path.exists(f"temp_{processed}.parquet"):
            print(f"{Colors.BLUE}Combining temporary parquet files...{Colors.ENDC}")
            all_cc0 = []
            for temp_file in sorted([f for f in os.listdir('.') if f.startswith('temp_') and f.endswith('.parquet')]):
                chunk = pd.read_parquet(temp_file)
                all_cc0.append(chunk)
                os.remove(temp_file)
            
            cc0_df = pd.concat(all_cc0) if all_cc0 else pd.DataFrame(cc0_photos)
        else:
            cc0_df = pd.DataFrame(cc0_photos)
        
        # Save final parquet file
        cc0_df.to_parquet(parquet_file)
        print(f"{Colors.GREEN}Created {parquet_file} with {len(cc0_df)} CC0 photos{Colors.ENDC}")
        return True
    
    except Exception as e:
        print(f"{Colors.RED}Error creating CC0 parquet: {str(e)}{Colors.ENDC}")
        return False

# Optimized async download functions
async def download_image(url, path, session):
    """Download an image using async I/O"""
    try:
        async with session.get(url, timeout=15) as response:
            if response.status == 200:
                data = await response.read()
                with open(path, 'wb') as f:
                    f.write(data)
                return True
            return False
    except Exception:
        return False

async def download_batch(urls_paths, max_concurrent=DOWNLOAD_THREADS):
    """Download a batch of images concurrently"""
    connector = aiohttp.TCPConnector(limit=max_concurrent, ttl_dns_cache=300)
    timeout = aiohttp.ClientTimeout(total=30)
    async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
        tasks = []
        for url, path in urls_paths:
            tasks.append(asyncio.ensure_future(
                download_image(url, path, session)
            ))
        return await asyncio.gather(*tasks)

def download_images_async(urls_paths):
    """Run the async download with proper event loop handling"""
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    
    return loop.run_until_complete(download_batch(urls_paths))

def create_direct_download_list(cc0_photos_df):
    """Create a direct download list without species information"""
    print(f"{Colors.BLUE}Creating direct download list without species mapping...{Colors.ENDC}")
    
    # We'll organize photos by observation_id instead of species
    observation_groups = {}
    
    with tqdm(total=len(cc0_photos_df), desc="Grouping photos by observation") as pbar:
        # Process in chunks to avoid memory issues
        for start_idx in range(0, len(cc0_photos_df), BATCH_SIZE):
            end_idx = min(start_idx + BATCH_SIZE, len(cc0_photos_df))
            batch = cc0_photos_df.iloc[start_idx:end_idx]
            
            for _, row in batch.iterrows():
                observation_id = str(row['observation_id'])
                photo_id = str(row['photo_id'])
                
                # Group by observation
                if observation_id not in observation_groups:
                    observation_groups[observation_id] = []
                
                observation_groups[observation_id].append(photo_id)
            
            pbar.update(len(batch))
    
    # Select a maximum number of observations
    if len(observation_groups) > MAX_OBSERVATIONS:
        selected_observations = random.sample(list(observation_groups.keys()), MAX_OBSERVATIONS)
        filtered_groups = {obs_id: observation_groups[obs_id] for obs_id in selected_observations}
        observation_groups = filtered_groups
    
    print(f"{Colors.GREEN}Grouped photos by {len(observation_groups)} observations{Colors.ENDC}")
    
    # Create download list
    download_list = []
    
    for obs_id, photo_ids in observation_groups.items():
        # Create observation directory
        obs_dir = os.path.join(OUTPUT_DIR, f"observation_{obs_id}")
        os.makedirs(obs_dir, exist_ok=True)
        
        # Limit photos per observation
        if len(photo_ids) > MAX_IMAGES_PER_OBSERVATION:
            photo_ids = photo_ids[:MAX_IMAGES_PER_OBSERVATION]
        
        # Add to download list
        for photo_id in photo_ids:
            url = f"https://inaturalist-open-data.s3.amazonaws.com/photos/{photo_id}/{IMAGE_QUALITY}.jpg"
            output_path = os.path.join(obs_dir, f"{photo_id}.jpg")
            
            if not os.path.exists(output_path):
                download_list.append((url, output_path, photo_id, obs_id))
    
    print(f"{Colors.GREEN}Created download list with {len(download_list)} images{Colors.ENDC}")
    return download_list

def download_direct(download_list, progress, terminator):
    """Download images without species information"""
    print(f"{Colors.BLUE}Starting direct downloads for {len(download_list)} images{Colors.ENDC}")
    
    # Create progress bar
    with tqdm(total=len(download_list), desc="Downloading images") as pbar:
        # Download in batches to manage memory
        batch_size = 200  # Larger batch size for direct downloads
        download_batches = [download_list[i:i+batch_size] for i in range(0, len(download_list), batch_size)]
        
        for batch in download_batches:
            # Check for termination
            if terminator.should_terminate():
                break
            
            # Prepare URLs and paths for async download
            urls_paths = [(item[0], item[1]) for item in batch]
            
            # Download batch using async I/O
            results = download_images_async(urls_paths)
            
            # Process results and track successful downloads
            for i, success in enumerate(results):
                if success and i < len(batch):
                    photo_id = batch[i][2]
                    obs_id = batch[i][3]
                    progress.mark_image_downloaded(obs_id, photo_id)
            
            # Update progress
            pbar.update(len(batch))
            
            # Clean memory after each batch
            clean_memory()

def download_by_species(download_list, species_groups, progress, terminator):
    """Download images organized by species"""
    print(f"{Colors.BLUE}Starting species-organized downloads for {len(download_list)} images{Colors.ENDC}")
    
    # Group download tasks by species for better organization
    downloads_by_species = {}
    for url, path, photo_id, species in download_list:
        if species not in downloads_by_species:
            downloads_by_species[species] = []
        downloads_by_species[species].append((url, path, photo_id))
    
    # Process one species at a time
    species_bar = tqdm(total=len(downloads_by_species), desc="Processing species")
    
    for species, species_downloads in downloads_by_species.items():
        # Check for termination
        if terminator.should_terminate():
            break
        
        # Create a nested progress bar for this species
        with tqdm(total=len(species_downloads), desc=f"Downloading {species[:25]}", leave=False) as download_bar:
            # Download in batches to manage memory
            batch_size = 200
            download_batches = [species_downloads[i:i+batch_size] for i in range(0, len(species_downloads), batch_size)]
            
            for batch in download_batches:
                # Check for termination
                if terminator.should_terminate():
                    break
                
                # Prepare URLs and paths for async download
                urls_paths = [(item[0], item[1]) for item in batch]
                
                # Download batch using async I/O
                results = download_images_async(urls_paths)
                
                # Process results
                for i, success in enumerate(results):
                    if success and i < len(batch):
                        photo_id = batch[i][2]
                        progress.mark_image_downloaded(species, photo_id)
                
                download_bar.update(len(batch))
                
                # Clean memory after each batch
                clean_memory()
        
        # Mark species as complete
        progress.mark_observation_complete(species)
        species_bar.update(1)
    
    species_bar.close()

def download_cc0_plant_images():
    """Main function to download CC0 plant images - Plants only, organized by species"""
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Start memory monitoring
    monitor_thread = monitor_memory()
    
    # Setup terminator
    terminator = GracefulTerminator()
    
    # Initialize download progress tracker
    progress = DownloadProgress()
    
    # Check for parquet file
    if not os.path.exists(PARQUET_FILE):
        if os.path.exists(CSV_FILE):
            # Create parquet file from CSV
            success = create_cc0_parquet(CSV_FILE, PARQUET_FILE)
            if not success:
                print(f"{Colors.RED}Failed to create CC0 parquet file. Exiting.{Colors.ENDC}")
                return
        else:
            print(f"{Colors.RED}Error: Neither {PARQUET_FILE} nor {CSV_FILE} found{Colors.ENDC}")
            return
    
    # Load CC0 photos
    print(f"{Colors.BLUE}Loading CC0 photos from parquet...{Colors.ENDC}")
    cc0_photos_df = pd.read_parquet(PARQUET_FILE)
    print(f"{Colors.GREEN}Loaded {len(cc0_photos_df)} CC0 photos{Colors.ENDC}")
    
    observations_file = os.path.join(METADATA_DIR, "observations.csv.gz")
    taxa_file = os.path.join(METADATA_DIR, "taxa.csv.gz")
    
    if not os.path.exists(observations_file) or not os.path.exists(taxa_file):
        print(f"{Colors.RED}Error: Missing necessary CSV files for taxonomy data{Colors.ENDC}")
        print(f"{Colors.YELLOW}Falling back to downloading all CC0 photos without filtering for plants.{Colors.ENDC}")
        download_list = create_direct_download_list(cc0_photos_df)
        download_direct(download_list, progress, terminator)
        return
    
    # Extract taxa data - build a mapping from taxon_id to name
    print(f"{Colors.BLUE}Extracting taxa names from taxa.csv.gz...{Colors.ENDC}")
    taxon_name_map = {}
    
    try:
        with gzip.open(taxa_file, 'rt') as f:
            reader = csv.reader(f, delimiter='\t')
            header = next(reader)
            
            # Find required columns
            taxon_id_idx = header.index('taxon_id') if 'taxon_id' in header else None
            name_idx = header.index('name') if 'name' in header else None
            rank_idx = header.index('rank') if 'rank' in header else None
            ancestry_idx = header.index('ancestry') if 'ancestry' in header else None
            
            if None in (taxon_id_idx, name_idx):
                print(f"{Colors.RED}Missing required columns in taxa file{Colors.ENDC}")
                print(f"{Colors.YELLOW}Falling back to downloading all CC0 photos.{Colors.ENDC}")
                download_list = create_direct_download_list(cc0_photos_df)
                download_direct(download_list, progress, terminator)
                return
            
            # Read all taxa data
            for row in tqdm(reader, desc="Reading taxa data"):
                try:
                    if len(row) > max(taxon_id_idx, name_idx):
                        taxon_id = row[taxon_id_idx].strip()
                        name = row[name_idx].strip()
                        
                        # Get rank if available
                        rank = None
                        if rank_idx is not None and rank_idx < len(row):
                            rank = row[rank_idx].strip()
                        
                        # Store taxon name
                        taxon_name_map[taxon_id] = name
                except (IndexError, ValueError):
                    continue
        
        print(f"{Colors.GREEN}Found {len(taxon_name_map)} taxa entries{Colors.ENDC}")
        
        # Now map observations to taxa
        print(f"{Colors.BLUE}Mapping observations to taxa...{Colors.ENDC}")
        observation_to_species = {}
        
        with gzip.open(observations_file, 'rt') as f:
            reader = csv.reader(f, delimiter='\t')
            header = next(reader)
            
            # Find required columns
            obs_id_idx = header.index('observation_uuid') if 'observation_uuid' in header else None
            taxon_id_idx = header.index('taxon_id') if 'taxon_id' in header else None
            
            if None in (obs_id_idx, taxon_id_idx):
                print(f"{Colors.RED}Missing required columns in observations file{Colors.ENDC}")
                print(f"{Colors.YELLOW}Falling back to downloading all CC0 photos.{Colors.ENDC}")
                download_list = create_direct_download_list(cc0_photos_df)
                download_direct(download_list, progress, terminator)
                return
            
            # Map observations to taxa
            for row in tqdm(reader, desc="Mapping observations to taxa"):
                try:
                    # Ensure we have safe indices
                    safe_obs_id_idx = 0 if obs_id_idx is None else obs_id_idx
                    safe_taxon_id_idx = 0 if taxon_id_idx is None else taxon_id_idx
                    
                    if len(row) > max(safe_obs_id_idx, safe_taxon_id_idx):
                        obs_id = row[safe_obs_id_idx].strip()
                        taxon_id = row[safe_taxon_id_idx].strip()
                        
                        # Only include taxa we have names for
                        if taxon_id in taxon_name_map:
                            # Get species name from taxon
                            species_name = taxon_name_map[taxon_id]
                            
                            # Clean species name for filesystem
                            clean_species_name = species_name.replace(' ', '_').replace('/', '_').replace('\\', '_')
                            clean_species_name = ''.join(c for c in clean_species_name if c.isalnum() or c == '_')
                            
                            # Ensure we have a valid name
                            if not clean_species_name:
                                clean_species_name = f"Taxon_{taxon_id}"
                            
                            observation_to_species[obs_id] = clean_species_name
                except (IndexError, ValueError):
                    continue
        
        print(f"{Colors.GREEN}Mapped {len(observation_to_species)} observations to taxa{Colors.ENDC}")
        
        # Now group CC0 photos by species
        print(f"{Colors.BLUE}Grouping CC0 photos by species...{Colors.ENDC}")
        species_groups = {}
        photos_count = 0
        
        with tqdm(total=len(cc0_photos_df), desc="Grouping photos by species") as pbar:
            # Process in chunks to avoid memory issues
            for start_idx in range(0, len(cc0_photos_df), BATCH_SIZE):
                end_idx = min(start_idx + BATCH_SIZE, len(cc0_photos_df))
                batch = cc0_photos_df.iloc[start_idx:end_idx]
                
                for _, row in batch.iterrows():
                    observation_id = str(row['observation_id'])
                    photo_id = str(row['photo_id'])
                    
                    # Check if we have species info for this observation
                    if observation_id in observation_to_species:
                        species_name = observation_to_species[observation_id]
                        
                        # Group by species
                        if species_name not in species_groups:
                            species_groups[species_name] = []
                        
                        species_groups[species_name].append((photo_id, observation_id))
                        photos_count += 1
                
                pbar.update(len(batch))
        
        print(f"{Colors.GREEN}Grouped {photos_count} photos across {len(species_groups)} species{Colors.ENDC}")
        
        # Filter species with too few images and limit max images per species
        filtered_species_groups = {}
        MIN_IMAGES_PER_SPECIES = 10  # Minimum images per species
        MAX_IMAGES_PER_SPECIES = 250  # Maximum images per species
        
        for species, photos in species_groups.items():
            if len(photos) >= MIN_IMAGES_PER_SPECIES:
                if len(photos) > MAX_IMAGES_PER_SPECIES:
                    # Randomly select MAX_IMAGES_PER_SPECIES photos
                    random.seed(42)  # For reproducibility
                    photos = random.sample(photos, MAX_IMAGES_PER_SPECIES)
                
                filtered_species_groups[species] = photos
        
        print(f"{Colors.GREEN}Filtered to {len(filtered_species_groups)} species with at least {MIN_IMAGES_PER_SPECIES} images{Colors.ENDC}")
        
        # Create download list organized by species
        download_list = []
        
        for species_name, photos in filtered_species_groups.items():
            # Create species directory
            species_dir = os.path.join(OUTPUT_DIR, species_name)
            os.makedirs(species_dir, exist_ok=True)
            
            # Add to download list
            for photo_id, observation_id in photos:
                url = f"https://inaturalist-open-data.s3.amazonaws.com/photos/{photo_id}/{IMAGE_QUALITY}.jpg"
                output_path = os.path.join(species_dir, f"{photo_id}.jpg")
                
                if not os.path.exists(output_path):
                    download_list.append((url, output_path, photo_id, species_name))
        
        print(f"{Colors.GREEN}Created download list with {len(download_list)} images{Colors.ENDC}")
        
        # Download images by species
        print(f"{Colors.BLUE}Starting downloads for {len(filtered_species_groups)} species{Colors.ENDC}")
        download_by_species(download_list, filtered_species_groups, progress, terminator)
        
        # Create summary
        print(f"{Colors.BLUE}Creating dataset summary...{Colors.ENDC}")
        total_images = 0
        species_count = 0
        
        for species_dir in os.listdir(OUTPUT_DIR):
            full_path = os.path.join(OUTPUT_DIR, species_dir)
            if os.path.isdir(full_path) and not species_dir.startswith('observation_'):
                image_count = len([f for f in os.listdir(full_path) if f.endswith('.jpg')])
                if image_count > 0:
                    total_images += image_count
                    species_count += 1
        
        print(f"{Colors.GREEN}Download complete! Downloaded {total_images} images across {species_count} species{Colors.ENDC}")
        
    except Exception as e:
        print(f"{Colors.RED}Error processing taxonomy data: {e}{Colors.ENDC}")
        import traceback
        traceback.print_exc()
        print(f"{Colors.YELLOW}Falling back to downloading all CC0 photos without filtering.{Colors.ENDC}")
        download_list = create_direct_download_list(cc0_photos_df)
        download_direct(download_list, progress, terminator)

def download_from_csv(csv_file_path):
    """Download images from a CSV file with species information already included"""
    print(f"{Colors.BLUE}Loading observations from CSV file...{Colors.ENDC}")
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Start memory monitoring
    monitor_thread = monitor_memory()
    
    # Setup terminator
    terminator = GracefulTerminator()
    
    # Initialize download progress tracker
    progress = DownloadProgress()
    
    # Parse the CSV file
    try:
        # First, check if the file is gzipped
        is_gzipped = csv_file_path.endswith('.gz')
        if is_gzipped:
            open_func = gzip.open
            mode = 'rt'
        else:
            open_func = lambda file, mode: open(file, mode, encoding='utf-8', errors='replace')
            mode = 'r'
        
        # Read all data from CSV
        species_groups = {}
        photos_by_species = {}
        
        with open_func(csv_file_path, mode) as f:
            reader = csv.DictReader(f)
            
            # Process each row
            for row in tqdm(reader, desc="Processing observations"):
                try:
                    # Extract required fields
                    observation_id = row.get('id', '')
                    image_url = row.get('image_url', '')
                    scientific_name = row.get('scientific_name', '')
                    common_name = row.get('common_name', '')
                    iconic_taxon = row.get('iconic_taxon_name', '')
                    
                    # Skip if missing required data
                    if not observation_id or not image_url:
                        continue
                    
                    # Skip if not a plant (if you only want plants)
                    if iconic_taxon and iconic_taxon != 'Plantae':
                        continue
                    
                    # Create species name (prefer scientific, fall back to common)
                    species_name = scientific_name if scientific_name else common_name
                    if not species_name:
                        species_name = f"Unknown_Species_{observation_id}"
                    
                    # Clean species name for filesystem
                    clean_species_name = species_name.replace(' ', '_').replace('/', '_').replace('\\', '_')
                    clean_species_name = ''.join(c for c in clean_species_name if c.isalnum() or c == '_')
                    
                    # Extract photo ID from URL
                    # Format: https://inaturalist-open-data.s3.amazonaws.com/photos/169241/medium.JPG
                    # or: https://static.inaturalist.org/photos/60085/medium.jpg
                    photo_id = None
                    if 'inaturalist-open-data' in image_url:
                        parts = image_url.split('/')
                        if len(parts) >= 5:
                            photo_id = parts[-2]
                    elif 'static.inaturalist.org' in image_url:
                        parts = image_url.split('/')
                        if len(parts) >= 5:
                            photo_id = parts[-2]
                    
                    if not photo_id:
                        continue
                    
                    # Group by species
                    if clean_species_name not in photos_by_species:
                        photos_by_species[clean_species_name] = []
                    
                    # Store photo info
                    photos_by_species[clean_species_name].append((photo_id, observation_id))
                    
                except Exception as e:
                    print(f"{Colors.YELLOW}Error processing row: {e}{Colors.ENDC}")
                    continue
        
        print(f"{Colors.GREEN}Found {sum(len(photos) for photos in photos_by_species.values())} photos across {len(photos_by_species)} species{Colors.ENDC}")
        
        # Filter species with too few images and limit max images per species
        filtered_species_groups = {}
        MIN_IMAGES_PER_SPECIES = 5  # Lower threshold to include more species
        MAX_IMAGES_PER_SPECIES = 250  # Maximum images per species
        
        for species, photos in photos_by_species.items():
            if len(photos) >= MIN_IMAGES_PER_SPECIES:
                if len(photos) > MAX_IMAGES_PER_SPECIES:
                    # Randomly select MAX_IMAGES_PER_SPECIES photos
                    random.seed(42)  # For reproducibility
                    photos = random.sample(photos, MAX_IMAGES_PER_SPECIES)
                
                filtered_species_groups[species] = photos
        
        print(f"{Colors.GREEN}Filtered to {len(filtered_species_groups)} species with at least {MIN_IMAGES_PER_SPECIES} images{Colors.ENDC}")
        
        # Create download list organized by species
        download_list = []
        
        for species_name, photos in filtered_species_groups.items():
            # Create species directory
            species_dir = os.path.join(OUTPUT_DIR, species_name)
            os.makedirs(species_dir, exist_ok=True)
            
            # Add to download list
            for photo_id, observation_id in photos:
                url = f"https://inaturalist-open-data.s3.amazonaws.com/photos/{photo_id}/{IMAGE_QUALITY}.jpg"
                output_path = os.path.join(species_dir, f"{photo_id}.jpg")
                
                if not os.path.exists(output_path):
                    download_list.append((url, output_path, photo_id, species_name))
        
        print(f"{Colors.GREEN}Created download list with {len(download_list)} images{Colors.ENDC}")
        
        # Download images by species
        print(f"{Colors.BLUE}Starting downloads for {len(filtered_species_groups)} species{Colors.ENDC}")
        download_by_species(download_list, filtered_species_groups, progress, terminator)
        
        # Create summary
        print(f"{Colors.BLUE}Creating dataset summary...{Colors.ENDC}")
        total_images = 0
        species_count = 0
        
        for species_dir in os.listdir(OUTPUT_DIR):
            full_path = os.path.join(OUTPUT_DIR, species_dir)
            if os.path.isdir(full_path) and not species_dir.startswith('observation_'):
                image_count = len([f for f in os.listdir(full_path) if f.endswith('.jpg')])
                if image_count > 0:
                    total_images += image_count
                    species_count += 1
        
        print(f"{Colors.GREEN}Download complete! Downloaded {total_images} images across {species_count} species{Colors.ENDC}")
        
    except Exception as e:
        print(f"{Colors.RED}Error processing observations CSV: {e}{Colors.ENDC}")
        import traceback
        traceback.print_exc()

def debug_csv_columns(file_path):
    """Debug CSV file columns"""
    try:
        print(f"{Colors.BLUE}Debugging CSV file: {file_path}{Colors.ENDC}")
        import gzip
        with gzip.open(file_path, 'rt') as f:
            # Read the first line to get headers
            header_line = f.readline().strip()
            headers = header_line.split('\t')
            
            print(f"{Colors.BLUE}Found {len(headers)} columns:{Colors.ENDC}")
            for i, col in enumerate(headers):
                print(f"  {i}: '{col}'")
                
            # Check for known important columns
            kingdom_cols = [i for i, col in enumerate(headers) if 'kingdom' in col.lower()]
            if kingdom_cols:
                print(f"{Colors.GREEN}Potential kingdom columns: {kingdom_cols} - {[headers[i] for i in kingdom_cols]}{Colors.ENDC}")
            else:
                print(f"{Colors.RED}No kingdom column found!{Colors.ENDC}")
                
            # Check alternatives
            print(f"{Colors.BLUE}Looking for taxonomy columns:{Colors.ENDC}")
            for keyword in ['phylum', 'class', 'order', 'family', 'genus', 'taxon', 'taxonomy']:
                matching = [i for i, col in enumerate(headers) if keyword in col.lower()]
                if matching:
                    print(f"  '{keyword}' columns: {[headers[i] for i in matching]}")
    except Exception as e:
        print(f"{Colors.RED}Error debugging CSV: {e}{Colors.ENDC}")

def check_taxa_columns():
    """Simple function to check taxa file columns"""
    taxa_file = os.path.join(METADATA_DIR, "taxa.csv.gz")
    observations_file = os.path.join(METADATA_DIR, "observations.csv.gz")
    
    print(f"{Colors.BLUE}=================== TAXA FILE COLUMNS ==================={Colors.ENDC}")
    debug_csv_columns(taxa_file)
    
    print(f"\n{Colors.BLUE}=================== OBSERVATIONS FILE COLUMNS ==================={Colors.ENDC}")
    debug_csv_columns(observations_file)
    
    return

def analyze_and_recommend():
    """Analyze the downloaded dataset and provide recommendations for improvement"""
    print(f"{Colors.BLUE}=================== DATASET ANALYSIS AND RECOMMENDATIONS ==================={Colors.ENDC}")
    
    # Check if output directory exists
    if not os.path.exists(OUTPUT_DIR):
        print(f"{Colors.RED}No dataset found at {OUTPUT_DIR}. Please run the download first.{Colors.ENDC}")
        return
    
    # Collect statistics
    total_folders = 0
    total_images = 0
    observation_folders = 0
    species_folders = 0
    folder_sizes = {}
    small_folders = []
    large_folders = []
    
    print(f"{Colors.BLUE}Analyzing downloaded dataset structure...{Colors.ENDC}")
    
    # Analyze folder structure
    for folder_name in os.listdir(OUTPUT_DIR):
        folder_path = os.path.join(OUTPUT_DIR, folder_name)
        if os.path.isdir(folder_path):
            total_folders += 1
            
            # Count images in folder
            image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            image_count = len(image_files)
            total_images += image_count
            folder_sizes[folder_name] = image_count
            
            # Classify folder type
            if folder_name.startswith('observation_'):
                observation_folders += 1
            else:
                species_folders += 1
            
            # Track small and large folders
            if 0 < image_count < 10:
                small_folders.append((folder_name, image_count))
            elif image_count > 200:
                large_folders.append((folder_name, image_count))
    
    # Sort folders by size
    sorted_folders = sorted(folder_sizes.items(), key=lambda x: x[1], reverse=True)
    
    # Calculate balanced dataset metrics
    if total_folders > 0:
        avg_images_per_folder = total_images / total_folders
        if len(folder_sizes) > 0:
            median_size = list(sorted(folder_sizes.values()))[len(folder_sizes)//2]
            min_size = min(folder_sizes.values()) if folder_sizes else 0
            max_size = max(folder_sizes.values()) if folder_sizes else 0
        else:
            median_size = min_size = max_size = 0
    else:
        avg_images_per_folder = median_size = min_size = max_size = 0
    
    # Print analysis results
    print(f"{Colors.GREEN}Dataset Analysis Results:{Colors.ENDC}")
    print(f"  Total images: {total_images}")
    print(f"  Total folders: {total_folders}")
    print(f"  Observation-based folders: {observation_folders}")
    print(f"  Species-based folders: {species_folders}")
    print(f"  Average images per folder: {avg_images_per_folder:.1f}")
    print(f"  Median folder size: {median_size}")
    print(f"  Size range: {min_size} - {max_size} images per folder")
    print(f"  Small folders (<10 images): {len(small_folders)}")
    print(f"  Large folders (>200 images): {len(large_folders)}")
    
    # Generate recommendations
    print(f"\n{Colors.BLUE}Recommendations for Dataset Improvement:{Colors.ENDC}")
    
    if observation_folders > 0 and species_folders == 0:
        print(f"{Colors.YELLOW}1. Your dataset is organized by observation ID rather than species.{Colors.ENDC}")
        print("   This structure is suboptimal for plant recognition training.")
        print("   Recommendations:")
        print("   - Download a proper taxonomy dataset with species information")
        print("   - Use a pre-trained model to categorize images into species")
        print("   - Manually sort a subset of images into species folders")
        print("   - Try: https://www.inaturalist.org/pages/developers for full taxonomy data")
    
    if species_folders > 0:
        # If we have species data but dataset is imbalanced
        if len(small_folders) > 0.3 * total_folders:
            print(f"{Colors.YELLOW}2. Your dataset has many species with too few images.{Colors.ENDC}")
            print(f"   {len(small_folders)} species have fewer than 10 images.")
            print("   Recommendations:")
            print("   - Increase MIN_IMAGES_PER_SPECIES (currently 10)")
            print("   - Focus on species with adequate samples (delete or merge small classes)")
            print("   - Use data augmentation for underrepresented species")
        
        if len(large_folders) > 0.1 * total_folders:
            print(f"{Colors.YELLOW}3. Your dataset has some species with very many images.{Colors.ENDC}")
            print(f"   {len(large_folders)} species have more than 200 images.")
            print("   Recommendations:")
            print("   - Decrease MAX_IMAGES_PER_SPECIES (currently 250) for more balance")
            print("   - Use weighted sampling during training to account for class imbalance")
    
    # Check overall dataset size for training
    if total_images < 5000:
        print(f"{Colors.YELLOW}4. Your dataset may be too small for robust training.{Colors.ENDC}")
        print(f"   Current size: {total_images} images across {total_folders} folders")
        print("   Recommendations:")
        print("   - Increase MAX_OBSERVATIONS (currently 7000)")
        print("   - Download images from additional sources")
        print("   - Use transfer learning with pretrained weights")
        print("   - Apply extensive data augmentation")

if __name__ == "__main__":
    # Select which function to run
    # 1. Check CSV columns
    # check_taxa_columns()
    
    # 2. Download using taxonomy approach from CC0 iNaturalist dataset
    multiprocessing.set_start_method('spawn', force=True)
    download_cc0_plant_images()
    
    # 3. Download directly from observations CSV with species info
    multiprocessing.set_start_method('spawn', force=True)
    csv_file_path = os.path.join(BASE_DIR, "data", "observations-561226.csv")
    download_from_csv(csv_file_path)
    
    # 4. Analyze downloaded dataset and get recommendations
>>>>>>> ba6742206bc2a45205ad6d1b60d1894da4fd3dd6
    # analyze_and_recommend()