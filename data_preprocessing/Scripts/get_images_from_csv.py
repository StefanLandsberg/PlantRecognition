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
import re
from tqdm.auto import tqdm
import multiprocessing
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict
import numpy as np

# ===== CONFIGURATION =====
# Update paths to be more robust while still relative
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))  # Get directory where script is located
BASE_DIR = os.path.dirname(SCRIPT_DIR)  # Parent directory of Scripts folder
OUTPUT_DIR = os.path.join(BASE_DIR, "data", "plant_images")
PARQUET_FILE = os.path.join(BASE_DIR, "cc0_photos.parquet")
METADATA_DIR = os.path.join(BASE_DIR, "data", "inaturalist_metadata")
# =========================
MAX_IMAGES_PER_OBSERVATION = 20        # Maximum images per observation
MAX_OBSERVATIONS = 20         # Maximum number of observations to download
IMAGE_QUALITY = "medium"                # Options: "original", "large", "medium", "small"
BATCH_SIZE = 5000                       # INCREASED chunk size for processing (was 2000)
DOWNLOAD_THREADS = 400                  # INCREASED concurrent downloads (was 300)
SPECIES_PROCESS_THREADS = 4             # NEW: Number of parallel species to process
MEMORY_LIMIT_PERCENT = 85               # Memory usage limit
CHECKPOINT_FREQUENCY = 1000             # Save checkpoint every N downloads
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

# Memory monitoring thread - OPTIMIZED
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
                time.sleep(30 if mem.percent > limit_percent-10 else 60)  # INCREASED sleep intervals
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

# OPTIMIZED Checkpoint management for resumable downloads
class DownloadProgress:
    def __init__(self, checkpoint_file="download_checkpoint.json"):
        self.checkpoint_file = checkpoint_file
        self.completed_observations = set()  # CHANGED to set for O(1) lookup
        self.downloaded_images = defaultdict(set)  # CHANGED to defaultdict with sets
        self.current_observation = None
        self.download_count = 0  # NEW: Track downloads for checkpoint frequency
        self.load_checkpoint()
    
    def load_checkpoint(self):
        """Load previous download progress"""
        if os.path.exists(self.checkpoint_file):
            try:
                with open(self.checkpoint_file, 'r') as f:
                    data = json.load(f)
                    self.completed_observations = set(data.get('completed_observations', []))
                    # Convert downloaded_images back to defaultdict of sets
                    downloaded_dict = data.get('downloaded_images', {})
                    for obs_id, img_list in downloaded_dict.items():
                        self.downloaded_images[obs_id] = set(img_list)
                    self.current_observation = data.get('current_observation')
                print(f"{Colors.GREEN}Resuming from previous download: {len(self.completed_observations)} observations completed{Colors.ENDC}")
            except Exception as e:
                print(f"{Colors.YELLOW}Error loading checkpoint: {e}{Colors.ENDC}")
    
    def save_checkpoint(self):
        """Save current download progress"""
        # Convert sets back to lists for JSON serialization
        downloaded_dict = {obs_id: list(img_set) for obs_id, img_set in self.downloaded_images.items()}
        data = {
            'completed_observations': list(self.completed_observations),
            'downloaded_images': downloaded_dict,
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
        return image_id in self.downloaded_images[observation_id]
    
    def mark_observation_complete(self, observation_id):
        """Mark an observation as completely processed"""
        self.completed_observations.add(observation_id)
        self.current_observation = None
        self.save_checkpoint()
    
    def mark_observation_current(self, observation_id):
        """Mark which observation is currently being processed"""
        self.current_observation = observation_id
        # Don't save checkpoint here - too frequent
    
    def mark_image_downloaded(self, observation_id, image_id):
        """Record a successfully downloaded image"""
        self.downloaded_images[observation_id].add(image_id)
        self.download_count += 1
        
        # Save checkpoint less frequently for better performance
        if self.download_count % CHECKPOINT_FREQUENCY == 0:
            self.save_checkpoint()

# Function to extract CC0 photos from the original CSV 
def create_cc0_parquet(csv_file, parquet_file, chunk_size=200000):  # INCREASED chunk size
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
                    
                    # Clean memory regularly - LESS FREQUENT
                    if len(cc0_photos) > 1000000:  # INCREASED threshold (was 500000)
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

# HIGHLY OPTIMIZED async download functions
async def download_image(url, path, session, retries=2):  # REDUCED retries for speed
    """Download an image using async I/O with fallback URLs and retries"""
    original_url = url
    photo_id = os.path.basename(path).replace('.jpg', '')
    
    # List of URLs to try - OPTIMIZED order
    urls_to_try = [
        original_url,
        f"https://inaturalist-open-data.s3.amazonaws.com/photos/{photo_id}/{IMAGE_QUALITY}.jpg",
        f"https://static.inaturalist.org/photos/{photo_id}/{IMAGE_QUALITY}.jpg"
    ]
    
    # Remove duplicates while preserving order
    urls_to_try = list(dict.fromkeys(urls_to_try))
    
    # Try each URL with retries
    last_error = None
    for url in urls_to_try:
        for attempt in range(retries):
            try:
                async with session.get(url, timeout=20) as response:  # REDUCED timeout for speed
                    if response.status == 200:
                        data = await response.read()
                        with open(path, 'wb') as f:
                            f.write(data)
                        return True, None
                    else:
                        last_error = f"HTTP {response.status}"
                        if attempt < retries - 1 and response.status in [500, 502, 503, 504]:
                             await asyncio.sleep(1)  # REDUCED backoff time
                             continue
                        break
            except Exception as e:
                last_error = str(e)
                if attempt < retries - 1:
                    await asyncio.sleep(1)  # REDUCED backoff time
                    continue
                break
        
        # If download succeeded for this URL after retries, no need to try other URLs
        if last_error is None:
             return True, None
    
    return False, last_error

async def download_batch(urls_paths, max_concurrent=DOWNLOAD_THREADS):
    """Download a batch of images concurrently - OPTIMIZED"""
    connector = aiohttp.TCPConnector(
        limit=max_concurrent, 
        ttl_dns_cache=600,  # INCREASED DNS cache
        use_dns_cache=True,
        keepalive_timeout=30,
        enable_cleanup_closed=True
    )
    timeout = aiohttp.ClientTimeout(total=40, connect=10)  # OPTIMIZED timeouts
    async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
        # Create semaphore to limit concurrent downloads
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def download_with_semaphore(url, path):
            async with semaphore:
                return await download_image(url, path, session)
        
        tasks = [download_with_semaphore(url, path) for url, path in urls_paths]
        return await asyncio.gather(*tasks, return_exceptions=True)

def download_images_async(urls_paths):
    """Run the async download with proper event loop handling"""
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    
    return loop.run_until_complete(download_batch(urls_paths))

def create_direct_download_list(cc0_photos_df):
    """Create a direct download list without species information - OPTIMIZED"""
    print(f"{Colors.BLUE}Creating direct download list without species mapping...{Colors.ENDC}")
    
    # Use pandas groupby for faster processing
    observation_groups = cc0_photos_df.groupby('observation_id')['photo_id'].apply(list).to_dict()
    
    # Select a maximum number of observations
    if len(observation_groups) > MAX_OBSERVATIONS:
        selected_observations = random.sample(list(observation_groups.keys()), MAX_OBSERVATIONS)
        observation_groups = {obs_id: observation_groups[obs_id] for obs_id in selected_observations}
    
    print(f"{Colors.GREEN}Grouped photos by {len(observation_groups)} observations{Colors.ENDC}")
    
    # Create download list
    download_list = []
    
    # Batch create directories
    obs_dirs = []
    for obs_id in observation_groups.keys():
        obs_dir = os.path.join(OUTPUT_DIR, f"observation_{obs_id}")
        obs_dirs.append(obs_dir)
    
    # Create all directories at once
    for obs_dir in obs_dirs:
        os.makedirs(obs_dir, exist_ok=True)
    
    for obs_id, photo_ids in observation_groups.items():
        obs_dir = os.path.join(OUTPUT_DIR, f"observation_{obs_id}")
        
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
    """Download images without species information - OPTIMIZED"""
    print(f"{Colors.BLUE}Starting direct downloads for {len(download_list)} images{Colors.ENDC}")
    
    # Create progress bar
    with tqdm(total=len(download_list), desc="Downloading images") as pbar:
        # Download in larger batches for better performance
        batch_size = 500  # INCREASED batch size
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
            for i, result in enumerate(results):
                if i < len(batch) and isinstance(result, tuple) and result[0]:
                    photo_id = batch[i][2]
                    obs_id = batch[i][3]
                    progress.mark_image_downloaded(obs_id, photo_id)
            
            # Update progress
            pbar.update(len(batch))
            
            # Clean memory less frequently
            if len(download_batches) > 10 and download_batches.index(batch) % 5 == 0:
                clean_memory()

def download_by_species(download_list, species_groups, progress, terminator):
    """Download images organized by species - PARALLELIZED"""
    print(f"{Colors.BLUE}Starting species-organized downloads for {len(download_list)} images{Colors.ENDC}")

    from concurrent.futures import ThreadPoolExecutor, as_completed
    # Group download tasks by species for better organization - OPTIMIZED
    downloads_by_species = defaultdict(list)
    for url, path, photo_id, species in download_list:
        downloads_by_species[species].append((url, path, photo_id))

    # Track download statistics
    download_stats = {
        'total': len(download_list),
        'success': 0,
        'failed': 0,
        'error_types': defaultdict(int)
    }

    def process_species(species_tuple):
        species, species_downloads = species_tuple
        if terminator.should_terminate():
            return (species, 0, 0, {})
        batch_size = 300
        download_batches = [species_downloads[i:i+batch_size] for i in range(0, len(species_downloads), batch_size)]
        species_success = 0
        species_failed = 0
        error_types = defaultdict(int)
        for batch in download_batches:
            if terminator.should_terminate():
                break
            urls_paths = [(item[0], item[1]) for item in batch]
            results = download_images_async(urls_paths)
            for i, result in enumerate(results):
                if i < len(batch):
                    if isinstance(result, tuple) and result[0]:
                        photo_id = batch[i][2]
                        progress.mark_image_downloaded(species, photo_id)
                        species_success += 1
                    else:
                        species_failed += 1
                        error = result[1] if isinstance(result, tuple) else str(result)
                        error_types[error] += 1
            # Clean memory less frequently
            if len(download_batches) > 5 and download_batches.index(batch) % 3 == 0:
                clean_memory()
        return (species, species_success, species_failed, error_types)

    species_items = list(downloads_by_species.items())
    species_bar = tqdm(total=len(species_items), desc="Processing species (parallel)")
    results = []
    with ThreadPoolExecutor(max_workers=SPECIES_PROCESS_THREADS) as executor:
        future_to_species = {executor.submit(process_species, item): item[0] for item in species_items}
        for future in as_completed(future_to_species):
            species = future_to_species[future]
            try:
                species, species_success, species_failed, error_types = future.result()
                download_stats['success'] += species_success
                download_stats['failed'] += species_failed
                for err, count in error_types.items():
                    download_stats['error_types'][err] += count
                if species_failed > species_success * 0.1:
                    print(f"{Colors.YELLOW}Species {species}: {species_success} successful, {species_failed} failed{Colors.ENDC}")
                progress.mark_observation_complete(species)
            except Exception as exc:
                print(f"{Colors.RED}Species {species} generated an exception: {exc}{Colors.ENDC}")
            species_bar.update(1)
    species_bar.close()

    # Report on overall download statistics
    print(f"\n{Colors.BLUE}Download Statistics:{Colors.ENDC}")
    print(f"  Total attempted: {download_stats['total']}")
    print(f"  Successful: {download_stats['success']} ({download_stats['success']/download_stats['total']*100:.1f}%)")
    print(f"  Failed: {download_stats['failed']} ({download_stats['failed']/download_stats['total']*100:.1f}%)")
    if download_stats['failed'] > 0:
        print(f"\n{Colors.YELLOW}Error Types:{Colors.ENDC}")
        for error, count in sorted(download_stats['error_types'].items(), key=lambda x: x[1], reverse=True)[:5]:
            print(f"  {error}: {count}")

# OPTIMIZED URL pattern matching with regex
URL_PATTERNS = [
    re.compile(r'inaturalist-open-data\.s3\.amazonaws\.com/photos/(\d+)/'),
    re.compile(r'static\.inaturalist\.org/photos/(\d+)/')
]

def extract_photo_id_fast(image_url, observation_id):
    """Fast photo ID extraction using compiled regex patterns"""
    for pattern in URL_PATTERNS:
        match = pattern.search(image_url)
        if match:
            return match.group(1)
    
    # Fallback to observation ID if it's numeric
    if observation_id.isdigit():
        return observation_id
    
    return None

def download_from_csv(csv_file_path):
    """Download images from a CSV file with species information already included - HIGHLY OPTIMIZED"""
    print(f"{Colors.BLUE}Loading observations from CSV file...{Colors.ENDC}")
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Start memory monitoring
    monitor_thread = monitor_memory()
    
    # Setup terminator
    terminator = GracefulTerminator()
    
    # Initialize download progress tracker
    progress = DownloadProgress()
    
    # Parse the CSV file using pandas for MUCH faster processing
    try:
        print(f"{Colors.BLUE}Reading CSV file with pandas...{Colors.ENDC}")
        
        # Read CSV in chunks for memory efficiency
        chunk_size = 50000  # LARGE chunks for speed
        chunks = []
        
        for chunk in pd.read_csv(csv_file_path, chunksize=chunk_size, low_memory=False):
            # Filter out rows with missing required data immediately
            chunk = chunk.dropna(subset=['id', 'image_url'])
            chunks.append(chunk)
        
        # Combine all chunks
        df = pd.concat(chunks, ignore_index=True)
        print(f"{Colors.GREEN}Loaded {len(df)} observations from CSV{Colors.ENDC}")
        
        # Vectorized operations for MUCH faster processing
        print(f"{Colors.BLUE}Processing species names...{Colors.ENDC}")
        
        # Create species names vectorized
        df['species_name'] = df['scientific_name'].fillna(df['common_name'])
        df['species_name'] = df['species_name'].fillna('Unknown_Species_' + df['iconic_taxon_name'].fillna('unnamed'))
        
        # Clean species names vectorized - MATCH MLP SCRIPT EXACTLY
        df['clean_species_name'] = (df['species_name']
                                   .str.replace(' ', '_', regex=False)
                                   .str.replace('/', '_', regex=False)
                                   .str.replace('\\', '_', regex=False)
                                   .str.replace(r'[^a-zA-Z0-9_]', '', regex=True))
        
        # DO NOT add taxon prefix - MLP script expects just scientific names
        # Removed add_taxon_prefix function call to match MLP expectations
        
        # Extract photo IDs ONLY for URL construction
        df['photo_id'] = df.apply(lambda row: extract_photo_id_fast(row['image_url'], str(row['id'])), axis=1)
        
        # Convert observation IDs to strings to ensure consistency
        df['observation_id'] = df['id'].astype(str)
        
        # Filter out rows where photo_id extraction failed
        original_count = len(df)
        df = df.dropna(subset=['photo_id'])
        filtered_count = len(df)
        if original_count != filtered_count:
            print(f"{Colors.YELLOW}Filtered out {original_count - filtered_count} observations without valid photo IDs{Colors.ENDC}")
        
        print(f"{Colors.GREEN}Successfully processed {len(df)} observations with valid photo IDs{Colors.ENDC}")
        print(f"{Colors.BLUE}Sample data - Observation ID: {df['observation_id'].iloc[0]}, Photo ID: {df['photo_id'].iloc[0]}{Colors.ENDC}")
        
        # Group by species - IMPORTANT: Store observation_id separately from photo_id
        species_groups = df.groupby('clean_species_name').apply(
            lambda x: list(zip(x['observation_id'], x['photo_id'], x['image_url']))
        ).to_dict()
        
        print(f"{Colors.GREEN}Found {len(species_groups)} species{Colors.ENDC}")
        
        # Filter species with too few images and enforce absolute max images per species
        MIN_IMAGES_PER_SPECIES = 20  # Only this min is used
        ABSOLUTE_MAX_IMAGES_PER_SPECIES = 20  # Use this as the only max

        filtered_species_groups = {}
        for species, photos in species_groups.items():
            # Always enforce absolute max
            if len(photos) > ABSOLUTE_MAX_IMAGES_PER_SPECIES:
                random.seed(42)
                photos = random.sample(photos, ABSOLUTE_MAX_IMAGES_PER_SPECIES)
            # Only include if meets min images requirement
            if len(photos) >= MIN_IMAGES_PER_SPECIES:
                filtered_species_groups[species] = photos
        
        print(f"{Colors.GREEN}Proceeding with {len(filtered_species_groups)} species{Colors.ENDC}")
        
        # Create download list - FIXED: observation_id comes FIRST in tuple
        download_list = []
        species_dirs = []
        for species_name, photos in filtered_species_groups.items():
            # Only create directory if there will be images in it
            species_dir = os.path.join(OUTPUT_DIR, species_name)
            # Build download list for this species
            species_downloads = []
            # Enforce absolute max here as well
            if len(photos) > ABSOLUTE_MAX_IMAGES_PER_SPECIES:
                photos = photos[:ABSOLUTE_MAX_IMAGES_PER_SPECIES]
            for observation_id, photo_id, original_url in photos:
                # Use photo_id to construct download URL
                if original_url.lower().endswith(('.jpg', '.jpeg', '.png')):
                    url = original_url
                else:
                    url = f"https://inaturalist-open-data.s3.amazonaws.com/photos/{photo_id}/{IMAGE_QUALITY}.jpg"
                output_path = os.path.join(species_dir, f"{species_name}_{observation_id}.jpg")
                if not os.path.exists(output_path):
                    species_downloads.append((url, output_path, observation_id, species_name))
            # Only create dir and add to download_list if there are images to download
            if species_downloads:
                os.makedirs(species_dir, exist_ok=True)
                download_list.extend(species_downloads)
        
        print(f"{Colors.GREEN}Created download list with {len(download_list)} images{Colors.ENDC}")
        
        # Download images by species
        print(f"{Colors.BLUE}Starting downloads for {len(filtered_species_groups)} species{Colors.ENDC}")
        download_by_species(download_list, filtered_species_groups, progress, terminator)
        
        # Create summary - OPTIMIZED
        print(f"{Colors.BLUE}Creating dataset summary...{Colors.ENDC}")
        total_images = 0
        species_count = 0
        
        with os.scandir(OUTPUT_DIR) as entries:
            for entry in entries:
                if entry.is_dir() and not entry.name.startswith('observation_'):
                    jpg_count = sum(1 for f in os.listdir(entry.path) if f.endswith('.jpg'))
                    if jpg_count > 0:
                        total_images += jpg_count
                        species_count += 1
        
        print(f"{Colors.GREEN}Download complete! Downloaded {total_images} images across {species_count} species{Colors.ENDC}")
        
    except Exception as e:
        print(f"{Colors.RED}Error processing observations CSV: {e}{Colors.ENDC}")
        import traceback
        traceback.print_exc()

def debug_csv_columns(file_path):
    """Debug CSV file columns - OPTIMIZED"""
    try:
        print(f"{Colors.BLUE}Debugging CSV file: {file_path}{Colors.ENDC}")
        
        # Use pandas for faster column detection
        if file_path.endswith('.gz'):
            df_sample = pd.read_csv(file_path, sep='\t', compression='gzip', nrows=0)  # Just get headers
        else:
            df_sample = pd.read_csv(file_path, nrows=0)  # Just get headers
        
        headers = df_sample.columns.tolist()
        
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
    """Analyze the downloaded dataset and provide recommendations for improvement - OPTIMIZED"""
    print(f"{Colors.BLUE}=================== DATASET ANALYSIS AND RECOMMENDATIONS ==================={Colors.ENDC}")
    
    # Check if output directory exists
    if not os.path.exists(OUTPUT_DIR):
        print(f"{Colors.RED}No dataset found at {OUTPUT_DIR}. Please run the download first.{Colors.ENDC}")
        return
    
    # Collect statistics using faster methods
    total_folders = 0
    total_images = 0
    observation_folders = 0
    species_folders = 0
    folder_sizes = {}
    small_folders = []
    large_folders = []
    
    print(f"{Colors.BLUE}Analyzing downloaded dataset structure...{Colors.ENDC}")
    
    # Use os.scandir for faster directory analysis
    with os.scandir(OUTPUT_DIR) as entries:
        for entry in entries:
            if entry.is_dir():
                total_folders += 1
                
                # Count images in folder using faster method
                jpg_count = sum(1 for f in os.listdir(entry.path) if f.lower().endswith(('.jpg', '.jpeg', '.png')))
                total_images += jpg_count
                folder_sizes[entry.name] = jpg_count
                
                # Classify folder type
                if entry.name.startswith('observation_'):
                    observation_folders += 1
                else:
                    species_folders += 1
                
                # Track small and large folders
                if 0 < jpg_count < 10:
                    small_folders.append((entry.name, jpg_count))
                elif jpg_count > 200:
                    large_folders.append((entry.name, jpg_count))
    
    # Calculate balanced dataset metrics
    if total_folders > 0:
        avg_images_per_folder = total_images / total_folders
        if len(folder_sizes) > 0:
            sorted_sizes = sorted(folder_sizes.values())
            median_size = sorted_sizes[len(sorted_sizes)//2]
            min_size = min(folder_sizes.values())
            max_size = max(folder_sizes.values())
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

def download_cc0_plant_images():
    """Main function to download CC0 plant images - Plants only, organized by species - OPTIMIZED"""
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
        print(f"{Colors.RED}Error: {PARQUET_FILE} not found. This function requires the CC0 parquet file.{Colors.ENDC}")
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
    
    # OPTIMIZED taxa data extraction using pandas
    print(f"{Colors.BLUE}Extracting taxa names from taxa.csv.gz...{Colors.ENDC}")
    try:
        # Read taxa file with pandas for much faster processing
        taxa_df = pd.read_csv(taxa_file, sep='\t', compression='gzip', low_memory=False)
        
        # Create taxon name mapping
        if 'taxon_id' in taxa_df.columns and 'name' in taxa_df.columns:
            taxon_name_map = dict(zip(taxa_df['taxon_id'].astype(str), taxa_df['name']))
            print(f"{Colors.GREEN}Found {len(taxon_name_map)} taxa entries{Colors.ENDC}")
        else:
            print(f"{Colors.RED}Missing required columns in taxa file{Colors.ENDC}")
            download_list = create_direct_download_list(cc0_photos_df)
            download_direct(download_list, progress, terminator)
            return
        
        # OPTIMIZED observations mapping using pandas
        print(f"{Colors.BLUE}Mapping observations to taxa...{Colors.ENDC}")
        obs_df = pd.read_csv(observations_file, sep='\t', compression='gzip', low_memory=False)
        
        if 'observation_uuid' in obs_df.columns and 'taxon_id' in obs_df.columns:
            # Filter observations that have taxa names
            obs_df = obs_df[obs_df['taxon_id'].astype(str).isin(taxon_name_map.keys())]
            
            # Map taxon_id to species names
            obs_df['species_name'] = obs_df['taxon_id'].astype(str).map(taxon_name_map)
            
            # Clean species names vectorized
            obs_df['clean_species_name'] = (obs_df['species_name']
                                          .str.replace(' ', '_', regex=False)
                                          .str.replace('/', '_', regex=False)
                                          .str.replace('\\', '_', regex=False)
                                          .str.replace(r'[^a-zA-Z0-9_]', '', regex=True))
            
            # Create observation to species mapping
            observation_to_species = dict(zip(obs_df['observation_uuid'], obs_df['clean_species_name']))
            print(f"{Colors.GREEN}Mapped {len(observation_to_species)} observations to taxa{Colors.ENDC}")
        else:
            print(f"{Colors.RED}Missing required columns in observations file{Colors.ENDC}")
            download_list = create_direct_download_list(cc0_photos_df)
            download_direct(download_list, progress, terminator)
            return
        
        # OPTIMIZED CC0 photos grouping using pandas
        print(f"{Colors.BLUE}Grouping CC0 photos by species...{Colors.ENDC}")
        
        # Filter CC0 photos that have species mapping
        cc0_photos_df['observation_id_str'] = cc0_photos_df['observation_id'].astype(str)
        cc0_photos_df = cc0_photos_df[cc0_photos_df['observation_id_str'].isin(observation_to_species.keys())]
        
        # Map species names
        cc0_photos_df['species_name'] = cc0_photos_df['observation_id_str'].map(observation_to_species)
        
        # Group by species
        species_groups = cc0_photos_df.groupby('species_name').apply(
            lambda x: list(zip(x['photo_id'], x['observation_id']))
        ).to_dict()
        
        print(f"{Colors.GREEN}Grouped {len(cc0_photos_df)} photos across {len(species_groups)} species{Colors.ENDC}")
        
        # Filter species with appropriate image counts
        filtered_species_groups = {}
        MIN_IMAGES_PER_SPECIES = 10
        MAX_IMAGES_PER_SPECIES = 250
        
        for species, photos in species_groups.items():
            if len(photos) >= MIN_IMAGES_PER_SPECIES:
                if len(photos) > MAX_IMAGES_PER_SPECIES:
                    random.seed(42)
                    photos = random.sample(photos, MAX_IMAGES_PER_SPECIES)
                filtered_species_groups[species] = photos
        
        print(f"{Colors.GREEN}Filtered to {len(filtered_species_groups)} species with at least {MIN_IMAGES_PER_SPECIES} images{Colors.ENDC}")
        
        # Create download list - OPTIMIZED
        download_list = []
        
        # Batch create directories
        species_dirs = []
        for species_name in filtered_species_groups.keys():
            species_dir = os.path.join(OUTPUT_DIR, species_name)
            species_dirs.append(species_dir)
        
        for species_dir in species_dirs:
            os.makedirs(species_dir, exist_ok=True)
        
        # Build download list
        for species_name, photos in filtered_species_groups.items():
            species_dir = os.path.join(OUTPUT_DIR, species_name)
            
            for photo_id, observation_id in photos:
                url = f"https://inaturalist-open-data.s3.amazonaws.com/photos/{photo_id}/{IMAGE_QUALITY}.jpg"
                output_path = os.path.join(species_dir, f"{species_name}_{observation_id}.jpg")
                
                if not os.path.exists(output_path):
                    download_list.append((url, output_path, photo_id, species_name))
        
        print(f"{Colors.GREEN}Created download list with {len(download_list)} images{Colors.ENDC}")
        
        # Download images by species
        print(f"{Colors.BLUE}Starting downloads for {len(filtered_species_groups)} species{Colors.ENDC}")
        download_by_species(download_list, filtered_species_groups, progress, terminator)
        
        # Create summary - OPTIMIZED
        print(f"{Colors.BLUE}Creating dataset summary...{Colors.ENDC}")
        total_images = 0
        species_count = 0
        
        with os.scandir(OUTPUT_DIR) as entries:
            for entry in entries:
                if entry.is_dir() and not entry.name.startswith('observation_'):
                    jpg_count = sum(1 for f in os.listdir(entry.path) if f.endswith('.jpg'))
                    if jpg_count > 0:
                        total_images += jpg_count
                        species_count += 1
        
        print(f"{Colors.GREEN}Download complete! Downloaded {total_images} images across {species_count} species{Colors.ENDC}")
        
    except Exception as e:
        print(f"{Colors.RED}Error processing taxonomy data: {e}{Colors.ENDC}")
        import traceback
        traceback.print_exc()
        print(f"{Colors.YELLOW}Falling back to downloading all CC0 photos without filtering.{Colors.ENDC}")
        download_list = create_direct_download_list(cc0_photos_df)
        download_direct(download_list, progress, terminator)

def remove_folders_below_min_images(min_images=2):
    """Remove species folders with fewer than min_images images."""
    print(f"{Colors.BLUE}Checking for species folders with fewer than {min_images} images...{Colors.ENDC}")
    removed = 0
    for entry in os.scandir(OUTPUT_DIR):
        if entry.is_dir() and not entry.name.startswith('observation_'):
            jpg_count = sum(1 for f in os.listdir(entry.path) if f.lower().endswith(('.jpg', '.jpeg', '.png')))
            if jpg_count < min_images:
                print(f"{Colors.YELLOW}Removing folder {entry.name} (only {jpg_count} images){Colors.ENDC}")
                for file in os.listdir(entry.path):
                    file_path = os.path.join(entry.path, file)
                    try:
                        os.remove(file_path)
                    except Exception as e:
                        print(f"{Colors.RED}Could not delete file {file_path}: {e}{Colors.ENDC}")
                try:
                    os.rmdir(entry.path)
                    removed += 1
                except Exception as e:
                    print(f"{Colors.RED}Could not delete folder {entry.path}: {e}{Colors.ENDC}")
    print(f"{Colors.GREEN}Removed {removed} folders with fewer than {min_images} images.{Colors.ENDC}")

if __name__ == "__main__":
    # Set multiprocessing start method once at the beginning
    multiprocessing.set_start_method('spawn', force=True)
    
    # Ensure necessary directories exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Get the path to the observations CSV file
    csv_file_path = os.path.join(BASE_DIR, "data", "observations-582302.csv")
    
    # Check if the CSV file exists
    if not os.path.exists(csv_file_path):
        print(f"{Colors.RED}Error: Observations CSV file not found at {csv_file_path}{Colors.ENDC}")
        print(f"{Colors.YELLOW}Please make sure the file exists before running this script.{Colors.ENDC}")
        exit(1)
        
    # Uncomment only ONE of the following functions to run:
    
    # 1. Check CSV columns
    # check_taxa_columns()
    
    # 2. Download using taxonomy approach from CC0 iNaturalist dataset
    # download_cc0_plant_images()
    
    # 3. Download directly from observations CSV with species info - OPTIMIZED
    download_from_csv(csv_file_path)
    
    # 4. Analyze downloaded dataset and get recommendations
    # analyze_and_recommend()
    # Optionally clean up folders below min images
    remove_folders_below_min_images(min_images=5)