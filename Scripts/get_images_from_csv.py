import os
import time
import gc
import asyncio
import signal
import json
import psutil
import threading
import csv
import random
import traceback
import atexit
import sys
import queue
from tqdm.auto import tqdm

# Set policy for Windows to use SelectEventLoopPolicy - this is a fix for Windows asyncio issues
if sys.platform == 'win32':
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# ===== CONFIGURATION =====
# These settings can be modified to change the behavior of the script
# ----------------------------------------------------------------------------
# Paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))    # Script directory
BASE_DIR = os.path.dirname(SCRIPT_DIR)                     # Project root directory
OUTPUT_DIR = os.path.join(BASE_DIR, "data", "plant_images") # Where images are saved
CSV_FILE = os.path.join(BASE_DIR, "data", "observations-561226.csv") # Input file

# Download parameters
MAX_IMAGES_PER_OBSERVATION = 250
MAX_IMAGES_PER_CLASS = 400
MAX_OBSERVATIONS = 99000                # Maximum number of observations to download
IMAGE_QUALITY = "medium"                # Options: "original", "large", "medium", "small"
DOWNLOAD_RETRY_ATTEMPTS = 3             # Number of download retry attempts
DOWNLOAD_TIMEOUT = 60                   # Timeout for download requests in seconds

# Pipeline configuration
PREBATCHER_COUNT = 32                   # Number of prebatching threads
QUEUE_SIZE_PER_PREBATCHER = 32          # Queue size for each prebatcher
MAX_PARALLEL_DOWNLOADS = 32             # Max number of batches to download simultaneously

# Filtering options
FILTER_BY_LICENSE = False               # Whether to filter by license type
LICENSE_TYPE = "CC0"                    # License to filter by if FILTER_BY_LICENSE is True
                                        # Common values: "CC0", "CC-BY", "CC-BY-NC"

# Resource management
MEMORY_LIMIT_PERCENT = 90              # Memory usage limit
PREBATCH_SIZE = 40000                  # Number of images to prepare ahead of time
# ----------------------------------------------------------------------------
# =========================

# Terminal colors for better output readability
class TermColors:
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BLUE = '\033[94m'
    MAGENTA = '\033[95m'
    CYAN = '\033[96m'
    BOLD = '\033[1m'
    ENDC = '\033[0m'

# Graceful termination handler
class GracefulTerminator:
    def __init__(self):
        self.terminate = False
        signal.signal(signal.SIGINT, self._signal_handler)
    
    def _signal_handler(self, sig, frame):
        """Handle Ctrl+C gracefully"""
        print(f"\n{TermColors.YELLOW}Termination requested. Will stop after current batch.{TermColors.ENDC}")
        self.terminate = True
    
    def should_terminate(self):
        return self.terminate

# Memory monitoring thread
def monitor_memory(limit_percent=MEMORY_LIMIT_PERCENT):
    """Background thread to monitor and manage memory usage"""
    def memory_monitor():
        print(f"{TermColors.BLUE}Starting memory monitor (limit: {limit_percent}%){TermColors.ENDC}")
        while True:
            try:
                mem = psutil.virtual_memory()
                if mem.percent > limit_percent:
                    print(f"{TermColors.YELLOW}High memory usage: {mem.percent}% - Cleaning...{TermColors.ENDC}")
                    gc.collect()
                time.sleep(10 if mem.percent > limit_percent-10 else 30)
            except Exception as e:
                print(f"{TermColors.RED}Memory monitor error: {e}{TermColors.ENDC}")
                time.sleep(60)
    
    thread = threading.Thread(target=memory_monitor, daemon=True)
    thread.start()
    return thread

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
                print(f"{TermColors.GREEN}Resuming from previous download: {len(self.completed_observations)} observations completed{TermColors.ENDC}")
            except Exception as e:
                print(f"{TermColors.YELLOW}Error loading checkpoint: {e}{TermColors.ENDC}")
    
    def save_checkpoint(self):
        """Save current download progress with proper locking"""
        try:
            # Use file locking to prevent corruption when multiple processes write
            import portalocker
            
            # Create a deep copy of the dictionaries to avoid modification during iteration
            import copy
            completed_observations_copy = copy.deepcopy(self.completed_observations)
            downloaded_images_copy = copy.deepcopy(self.downloaded_images)
            
            data = {
                'completed_observations': completed_observations_copy,
                'downloaded_images': downloaded_images_copy,
                'current_observation': self.current_observation,
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
            }
            
            # Use exclusive lock when writing
            with open(self.checkpoint_file + '.tmp', 'w') as f:
                portalocker.lock(f, portalocker.LOCK_EX)
                json.dump(data, f)
                f.flush()
                os.fsync(f.fileno())
                
            # Atomic rename for safer file replacement
            if sys.platform == 'win32':
                # Windows needs special handling for atomic rename
                import os
                if os.path.exists(self.checkpoint_file):
                    os.replace(self.checkpoint_file + '.tmp', self.checkpoint_file)
                else:
                    os.rename(self.checkpoint_file + '.tmp', self.checkpoint_file)
            else:
                # Unix supports atomic rename
                os.rename(self.checkpoint_file + '.tmp', self.checkpoint_file)
        except Exception as e:
            # Log but continue - checkpoint saving shouldn't halt the process
            pass
    
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

# Multi-stage pipeline architecture for continuous processing
class DownloadPipeline:
    """Advanced pipeline that keeps multiple batches in different stages of processing
    
    This creates a true pipeline with multiple stages:
    1. CSV processing stage - reads observations and creates download items
    2. Multiple prebatching stages - prepare batches in parallel
    3. Multiple download stages - download multiple batches simultaneously
    """

    def __init__(self, prebatcher_count=PREBATCHER_COUNT, queue_size=QUEUE_SIZE_PER_PREBATCHER):
        self.terminator = GracefulTerminator()
        self.progress = DownloadProgress()
        
        # Register exit handlers
        register_exit_handler(self.progress)
        
        # Create multiple prebatchers for a multi-stage pipeline
        self.prebatchers = []
        for i in range(prebatcher_count):
            prebatcher = {
                'id': i,
                'queue': queue.Queue(maxsize=queue_size),
                'current_batch': [],
                'thread': None,
                'is_running': True,
                'lock': threading.Lock()
            }
            self.prebatchers.append(prebatcher)
        
        # Create a queue for download-ready batches
        self.download_queue = queue.Queue(maxsize=MAX_PARALLEL_DOWNLOADS * 2)
        
        # Status tracking
        self.current_obs_index = 0
        self.observation_ids = []
        self.obs_data = {}
        self.total_observations = 0
        
        # Metrics
        self.prebatch_stats = {}
        self.download_stats = {
            'batches_processed': 0,
            'total_downloads': 0,
            'successful_downloads': 0,
            'start_time': time.time()
        }
        
        # Memory monitor
        self.memory_monitor = monitor_memory()
    
    def start(self, observation_ids, obs_data, output_dir, quality):
        """Start the pipeline with the given observation data"""
        self.observation_ids = observation_ids
        self.obs_data = obs_data
        self.total_observations = len(observation_ids)
        self.output_dir = output_dir
        self.quality = quality
        
        # Start prebatcher threads
        for prebatcher in self.prebatchers:
            thread = threading.Thread(
                target=self._prebatch_worker, 
                args=(prebatcher,)
            )
            thread.daemon = True
            thread.start()
            prebatcher['thread'] = thread
        
        # Start downloader threads
        for i in range(MAX_PARALLEL_DOWNLOADS):
            thread = threading.Thread(
                target=self._download_worker,
                args=(i,)
            )
            thread.daemon = True
            thread.start()
        
        # Start supervisor thread that manages observation flow
        supervisor = threading.Thread(target=self._supervisor_worker)
        supervisor.daemon = True
        supervisor.start()
        
        return supervisor
    
    def _supervisor_worker(self):
        """Supervisor thread that keeps the pipeline fed with observations"""
        try:
            print(f"{TermColors.BLUE}Starting pipeline supervisor{TermColors.ENDC}")
            
            # Add a counter to track consecutive small batches
            small_batch_count = 0
            small_batch_threshold = 10  # Number of consecutive small batches before terminating
            
            # Track images per class to enforce MAX_IMAGES_PER_CLASS
            class_image_counts = {}
            
            while self.current_obs_index < self.total_observations and not self.terminator.should_terminate():
                # Process a batch of observations
                batch_end = min(self.current_obs_index + 50, self.total_observations)
                
                # Track how many images were added in this batch
                batch_images_added = 0
                
                for idx in range(self.current_obs_index, batch_end):
                    if self.terminator.should_terminate():
                        break
                    
                    obs_id = self.observation_ids[idx]
                    
                    # Skip if already completed
                    if self.progress.observation_completed(obs_id):
                        continue
                    
                    # Get class name and image URLs for this observation
                    obs_info = self.obs_data.get(obs_id, {})
                    class_name = obs_info.get('class_name', f"unknown_{obs_id}")
                    image_urls = obs_info.get('image_urls', [])
                    
                    # Skip if no URLs available
                    if not image_urls:
                        self.progress.mark_observation_complete(obs_id)
                        continue
                    
                    # Check if we've already hit MAX_IMAGES_PER_CLASS for this class
                    if class_name in class_image_counts and class_image_counts[class_name] >= MAX_IMAGES_PER_CLASS:
                        # Skip this observation as we already have enough images for this class
                        self.progress.mark_observation_complete(obs_id)
                        continue
                    
                    # Create class directory
                    class_dir = os.path.join(self.output_dir, class_name)
                    if not os.path.exists(class_dir):
                        try:
                            os.makedirs(class_dir)
                        except FileExistsError:
                            pass  # Race condition handling
                    
                    # Mark this observation as current
                    self.progress.mark_observation_current(obs_id)
                    
                    # Calculate how many more images we can add for this class
                    if class_name not in class_image_counts:
                        class_image_counts[class_name] = 0
                    
                    remaining_class_capacity = MAX_IMAGES_PER_CLASS - class_image_counts[class_name]
                    
                    # Process each image URL (limit to MAX_IMAGES_PER_OBSERVATION and remaining class capacity)
                    images_added_for_obs = 0
                    max_images_for_obs = min(MAX_IMAGES_PER_OBSERVATION, remaining_class_capacity)
                    
                    for i, photo_url in enumerate(image_urls[:max_images_for_obs]):
                        # Generate a unique ID for this image
                        photo_id = f"{obs_id}_{i}"
                        
                        # Skip if already downloaded
                        if self.progress.image_downloaded(obs_id, photo_id):
                            continue
                        
                        # Path for saving this image
                        image_path = os.path.join(class_dir, f"{class_name}_{obs_id}_{i}.jpg")
                        
                        # Add to a prebatcher (round-robin distribution)
                        item = (photo_url, image_path, photo_id, obs_id)
                        prebatcher_idx = idx % len(self.prebatchers)
                        self._add_to_prebatcher(prebatcher_idx, item)
                        batch_images_added += 1
                        images_added_for_obs += 1
                        
                        # Update the class image count
                        class_image_counts[class_name] += 1
                        
                        # Stop if we've reached the class limit
                        if class_image_counts[class_name] >= MAX_IMAGES_PER_CLASS:
                            break
                    
                    # Mark observation complete
                    self.progress.mark_observation_complete(obs_id)
                
                # Check if we're getting small batches consistently
                if batch_images_added <= 10:  # If we only added a few images
                    small_batch_count += 1
                    
                    if small_batch_count >= small_batch_threshold:
                        print(f"{TermColors.GREEN}Detected end of significant data - processing only small batches. Gracefully finishing...{TermColors.ENDC}")
                        break  # Exit the main loop to finish processing
                else:
                    # Reset the counter if we had a substantial batch
                    small_batch_count = 0
                
                # Update index
                self.current_obs_index = batch_end
                
                # Show progress
                self._print_pipeline_status()
                
                # Brief pause to avoid hogging CPU
                if self.current_obs_index < self.total_observations:
                    time.sleep(0.1)
            
            # Wait for all prebatchers to finish
            print(f"{TermColors.BLUE}All observations processed. Waiting for downloads to complete...{TermColors.ENDC}")
            self._print_pipeline_status()
            
            # Signal all prebatchers to finish
            for prebatcher in self.prebatchers:
                with prebatcher['lock']:
                    if prebatcher['current_batch']:
                        try:
                            prebatcher['queue'].put(prebatcher['current_batch'], block=False)
                            prebatcher['current_batch'] = []
                        except queue.Full:
                            pass
            
            # Wait until download queue is empty and all prebatchers are done
            max_wait_time = 60  # Maximum time to wait in seconds
            start_wait_time = time.time()
            
            while not self.download_queue.empty():
                if time.time() - start_wait_time > max_wait_time:
                    print(f"{TermColors.YELLOW}Maximum wait time reached. Some small downloads may be incomplete.{TermColors.ENDC}")
                    break
                    
                time.sleep(1)
                if self.terminator.should_terminate():
                    break
                self._print_pipeline_status()
            
            # One final status update
            self._print_pipeline_status(final=True)
            
        except Exception as e:
            print(f"{TermColors.RED}Error in supervisor: {e}{TermColors.ENDC}")
            traceback.print_exc()
            # Save progress even on error
            self.progress.save_checkpoint()
    
    def _add_to_prebatcher(self, prebatcher_idx, item):
        """Add an item to a specific prebatcher"""
        prebatcher = self.prebatchers[prebatcher_idx]
        
        with prebatcher['lock']:
            prebatcher['current_batch'].append(item)
            
            # If the batch is full enough, try to queue it
            if len(prebatcher['current_batch']) >= PREBATCH_SIZE // len(self.prebatchers):
                self._try_queue_prebatcher_batch(prebatcher)
    
    def _try_queue_prebatcher_batch(self, prebatcher):
        """Try to queue a prebatcher's current batch"""
        with prebatcher['lock']:
            if not prebatcher['current_batch']:
                return
            
            try:
                # Try to queue the batch without blocking
                prebatcher['queue'].put(prebatcher['current_batch'], block=False)
                prebatcher['current_batch'] = []
            except queue.Full:
                # Queue is full, we'll try again later
                pass
    
    def _prebatch_worker(self, prebatcher):
        """Worker thread for a prebatch stage in the pipeline"""
        prebatcher_id = prebatcher['id']
        self.prebatch_stats[prebatcher_id] = {
            'batches_processed': 0,
            'items_processed': 0
        }
        
        try:
            print(f"{TermColors.BLUE}Starting prebatcher {prebatcher_id}{TermColors.ENDC}")
            
            while prebatcher['is_running'] and not self.terminator.should_terminate():
                try:
                    # Get a batch from this prebatcher's queue
                    batch = prebatcher['queue'].get(timeout=1)
                    
                    if batch:
                        # Queue this batch for download
                        try:
                            self.download_queue.put(batch, timeout=30)
                            
                            # Update stats
                            self.prebatch_stats[prebatcher_id]['batches_processed'] += 1
                            self.prebatch_stats[prebatcher_id]['items_processed'] += len(batch)
                            
                        except queue.Full:
                            # Download queue is full - put the batch back in our queue
                            prebatcher['queue'].put(batch, timeout=5)
                            time.sleep(1)
                
                except queue.Empty:
                    # Check if we need to queue the current batch
                    self._try_queue_prebatcher_batch(prebatcher)
                    
                    # No batch available, wait a bit
                    time.sleep(0.1)
                
                except Exception as e:
                    print(f"{TermColors.RED}Prebatcher {prebatcher_id} error: {e}{TermColors.ENDC}")
                    time.sleep(1)
        
        except Exception as e:
            print(f"{TermColors.RED}Error in prebatcher {prebatcher_id}: {e}{TermColors.ENDC}")
            traceback.print_exc()
    
    def _process_download_batch(self, batch, batch_id):
        """Process a batch of downloads directly within the thread"""
        try:
            # Filter already downloaded images
            filtered_list = []
            for url, path, photo_id, obs_id in batch:
                if not os.path.exists(path) and not self.progress.image_downloaded(obs_id, photo_id):
                    filtered_list.append((url, path, photo_id, obs_id))
            
            if not filtered_list:
                return 0  # All images already downloaded
            
            # Process downloads
            total_success = 0
            
            # Split into smaller batches for more efficient processing
            sub_batch_size = min(100, len(filtered_list))  # Increased from 10 to 100 for better throughput
            sub_batches = [filtered_list[i:i+sub_batch_size] for i in range(0, len(filtered_list), sub_batch_size)]
            
            # Get terminal width for progress bar
            terminal_width = 50  # Default width
            try:
                import shutil
                terminal_width = shutil.get_terminal_size().columns - 40  # Leave space for text
                terminal_width = max(20, min(terminal_width, 60))  # Keep between 20 and 60 chars
            except:
                pass
            
            # Create a simple progress tracking system
            total_items = len(filtered_list)
            processed_items = 0
            
            # Initial progress bar
            self._update_progress_bar(batch_id, processed_items, total_items, terminal_width)
            
            # Process batch with standard requests library
            for sub_batch in sub_batches:
                if self.terminator.should_terminate():
                    break
                
                import requests
                from concurrent.futures import ThreadPoolExecutor
                
                def download_one(item):
                    nonlocal processed_items, total_success
                    url, path, photo_id, obs_id = item
                    for attempt in range(DOWNLOAD_RETRY_ATTEMPTS):
                        try:
                            response = requests.get(url, timeout=DOWNLOAD_TIMEOUT)
                            if response.status_code == 200:
                                # Create directory if needed
                                os.makedirs(os.path.dirname(path), exist_ok=True)
                                
                                # Write to file
                                with open(path, 'wb') as f:
                                    f.write(response.content)
                                
                                # Explicitly track this successful download immediately
                                self.progress.mark_image_downloaded(obs_id, photo_id)
                                
                                # Update stats immediately for real-time tracking
                                with threading.Lock():
                                    self.download_stats['successful_downloads'] += 1
                                    total_success += 1
                                    processed_items += 1
                                
                                # Update progress bar
                                self._update_progress_bar(batch_id, processed_items, total_items, terminal_width)
                                return True
                            elif response.status_code == 404:
                                # Don't retry if the image doesn't exist
                                with threading.Lock():
                                    processed_items += 1
                                self._update_progress_bar(batch_id, processed_items, total_items, terminal_width)
                                return False
                            time.sleep(1 * (attempt + 1))
                        except requests.Timeout:
                            time.sleep(2 * (attempt + 1))
                        except Exception as e:
                            print(f"{TermColors.YELLOW}Download error: {str(e)}{TermColors.ENDC}")
                            time.sleep(0.5 * (attempt + 1))
                    
                    # If we get here, all attempts failed
                    with threading.Lock():
                        processed_items += 1
                    self._update_progress_bar(batch_id, processed_items, total_items, terminal_width)
                    return False
                
                # Use ThreadPoolExecutor with appropriate number of workers
                sub_batch_success = 0
                with ThreadPoolExecutor(max_workers=min(32, len(sub_batch))) as executor:
                    futures = {executor.submit(download_one, item): i for i, item in enumerate(sub_batch)}
                    
                    # Process futures as they complete
                    for future in futures:
                        try:
                            future.result()
                        except Exception as e:
                            with threading.Lock():
                                processed_items += 1
                            self._update_progress_bar(batch_id, processed_items, total_items, terminal_width)
            
            # Final update to ensure we show 100%
            self._update_progress_bar(batch_id, total_items, total_items, terminal_width)
            
            # Save progress after batch
            self.progress.save_checkpoint()
            
            return total_success
            
        except Exception as e:
            print(f"{TermColors.RED}Error processing download batch: {e}{TermColors.ENDC}")
            traceback.print_exc()
            return 0
            
    def _update_progress_bar(self, batch_id, current, total, width=50):
        """Update progress tracking without showing a progress bar - just completion status"""
        # Lock for thread-safe terminal output
        if not hasattr(self, '_print_lock'):
            self._print_lock = threading.Lock()
        
        # Only show completion when the batch is done
        if current >= total:
            with self._print_lock:
                # Calculate download speed for the completion message
                if not hasattr(self, '_batch_start_times'):
                    self._batch_start_times = {}
                
                if batch_id not in self._batch_start_times:
                    self._batch_start_times[batch_id] = time.time()
                
                elapsed = time.time() - self._batch_start_times[batch_id]
                download_rate = current / max(0.1, elapsed) if current > 0 else 0
                
                # Print a completion message with a check mark
                print(f"{TermColors.GREEN}✓{TermColors.ENDC} Batch {batch_id} completed: {current}/{total} images [{download_rate:.1f} img/s]")
    
    def _download_worker(self, worker_id):
        """Downloader thread that processes ready-to-download batches"""
        try:
            print(f"{TermColors.BLUE}Starting downloader {worker_id}{TermColors.ENDC}")
            # Track batch counter per worker to ensure unique batch IDs
            batch_counter = 0
            
            while not self.terminator.should_terminate():
                try:
                    # Get a batch from the download queue
                    batch = self.download_queue.get(timeout=5)
                    
                    if batch:
                        # Generate a unique batch ID using worker ID and counter
                        batch_id = f"{worker_id}_{batch_counter}"
                        batch_counter += 1
                        
                        # Simple one-line status update when starting a batch
                        print(f"{TermColors.CYAN}Loading batch {batch_id}: {len(batch)} images{TermColors.ENDC}", flush=True)
                        
                        # Delay slightly to ensure the message is displayed
                        time.sleep(0.1)
                        
                        # Process this batch
                        start_time = time.time()
                        success_count = self._process_download_batch(batch, batch_id)
                        end_time = time.time()
                        
                        # Update stats
                        with threading.Lock():
                            self.download_stats['batches_processed'] += 1
                            self.download_stats['total_downloads'] += len(batch)
                        
                        # Brief pause to let other threads run
                        time.sleep(0.1)
                
                except queue.Empty:
                    # No batch available, wait a bit
                    time.sleep(0.1)
                
                except Exception as e:
                    print(f"{TermColors.RED}Downloader {worker_id} error: {e}{TermColors.ENDC}")
                    traceback.print_exc()
                    time.sleep(1)
        
        except Exception as e:
            print(f"{TermColors.RED}Error in downloader {worker_id}: {e}{TermColors.ENDC}")
            traceback.print_exc()
    
    def _print_pipeline_status(self, final=False):
        """Print current pipeline status"""
        # Only print status every 2 minutes, unless it's the final status
        current_time = time.time()
        if not hasattr(self, '_last_status_time'):
            self._last_status_time = 0
        
        # Skip status update if it hasn't been 2 minutes yet, unless it's the final status
        if not final and (current_time - self._last_status_time < 120):  # 120 seconds = 2 minutes
            return
            
        # Update the last status time
        self._last_status_time = current_time
        
        completed_obs = len(self.progress.completed_observations)
        percentage = completed_obs / self.total_observations * 100
        
        # Calculate download rate
        elapsed = time.time() - self.download_stats['start_time']
        download_rate = self.download_stats['successful_downloads'] / max(1, elapsed)
        
        # Queue sizes
        prebatcher_queue_sizes = [len(p['current_batch']) + p['queue'].qsize() for p in self.prebatchers]
        total_queued = sum(prebatcher_queue_sizes) + self.download_queue.qsize()
        
        # Calculate ETA
        remaining_obs = self.total_observations - completed_obs
        avg_imgs_per_obs = max(1, self.download_stats['successful_downloads']) / max(1, completed_obs)
        remaining_imgs = remaining_obs * avg_imgs_per_obs
        eta_seconds = remaining_imgs / max(1, download_rate)
        eta_str = f"{int(eta_seconds // 3600)}h {int((eta_seconds % 3600) // 60)}m {int(eta_seconds % 60)}s"
        
        # Format the progress bar
        bar_width = 50
        bar_fill = int(percentage / 100 * bar_width)
        progress_bar = "█" * bar_fill + "░" * (bar_width - bar_fill)
        
        # Only show pipeline status (no final status box, even when finished)
        status = f"""
{TermColors.CYAN}╔══════════════════════ Pipeline Status ══════════════════════╗{TermColors.ENDC}
{TermColors.CYAN}║{TermColors.ENDC} Progress: {percentage:6.2f}% {progress_bar} {TermColors.CYAN}║{TermColors.ENDC}
{TermColors.CYAN}║{TermColors.ENDC} Observations: {completed_obs}/{self.total_observations} processed               {TermColors.CYAN}║{TermColors.ENDC}
{TermColors.CYAN}║{TermColors.ENDC} Images: {self.download_stats['successful_downloads']}/{self.download_stats['total_downloads']} downloaded                    {TermColors.CYAN}║{TermColors.ENDC}
{TermColors.CYAN}║{TermColors.ENDC} Download rate: {download_rate:.1f} images/second                {TermColors.CYAN}║{TermColors.ENDC}
{TermColors.CYAN}║{TermColors.ENDC} Pipeline queue: {total_queued} images waiting                  {TermColors.CYAN}║{TermColors.ENDC}
{TermColors.CYAN}║{TermColors.ENDC} Estimated time to completion: {eta_str}                {TermColors.CYAN}║{TermColors.ENDC}
{TermColors.CYAN}║{TermColors.ENDC} Memory usage: {psutil.virtual_memory().percent}%                                 {TermColors.CYAN}║{TermColors.ENDC}
{TermColors.CYAN}╚═════════════════════════════════════════════════════════════╝{TermColors.ENDC}
"""
        print(status)

# Register global exit handler to save progress on unexpected exit
def register_exit_handler(progress):
    """Register handlers to save progress on various exit scenarios"""
    def on_exit():
        print(f"\n{TermColors.YELLOW}Process ending: Saving download progress...{TermColors.ENDC}")
        progress.save_checkpoint()
    
    # Register for normal exit
    atexit.register(on_exit)
    
    # For Windows Ctrl+Break and other signals
    signal.signal(signal.SIGTERM, lambda sig, frame: on_exit())
    
    # For system shutdown (Windows only)
    if sys.platform == 'win32':
        try:
            # Special Windows exit handler 
            import win32api
            win32api.SetConsoleCtrlHandler(lambda sig: on_exit() or 1, True)
        except ImportError:
            pass

def extract_observation_ids_from_csv(csv_file, target_count=MAX_OBSERVATIONS):
    """Extract observation IDs and scientific names directly from the CSV"""
    print(f"{TermColors.BLUE}Loading observations from {csv_file}...{TermColors.ENDC}")
    
    # Data structures to store results
    observation_data = {}
    species_counts = {}  # Track counts per species
    
    try:
        # Try different encodings if the default fails
        encodings_to_try = ['utf-8', 'latin-1', 'ISO-8859-1', 'cp1252']
        
        for encoding in encodings_to_try:
            try:
                print(f"{TermColors.BLUE}Trying to open CSV with {encoding} encoding...{TermColors.ENDC}")
                
                with open(csv_file, 'r', encoding=encoding) as f:
                    # Read a small portion to test if encoding works
                    sample = f.read(1024)
                    f.seek(0)  # Reset file pointer
                    
                    # Check if file has header
                    reader = csv.DictReader(f)
                    
                    # Check required columns exist
                    headers = reader.fieldnames
                    if not headers:
                        continue
                        
                    required_columns = ['id', 'scientific_name', 'image_url', 'iconic_taxon_name']
                    missing_columns = [col for col in required_columns if col not in headers]
                    
                    if missing_columns:
                        print(f"{TermColors.YELLOW}Missing required columns: {missing_columns}{TermColors.ENDC}")
                        print(f"{TermColors.YELLOW}Available columns: {headers}{TermColors.ENDC}")
                        # Print first few rows for debugging
                        print(f"{TermColors.YELLOW}First row sample: {next(reader, None)}{TermColors.ENDC}")
                        continue
                    
                    # Process all rows
                    print(f"{TermColors.GREEN}Processing observations with {encoding} encoding...{TermColors.ENDC}")
                    row_count = 0
                    plant_count = 0
                    non_plant_count = 0
                    empty_url_count = 0
                    empty_taxon_count = 0
                    
                    with tqdm(desc="Processing observations", unit="obs") as pbar:
                        for row in reader:
                            try:
                                row_count += 1
                                obs_id = row.get('id', '').strip()
                                if not obs_id:
                                    continue
                                
                                # Check license if filtering is enabled
                                if FILTER_BY_LICENSE and row.get('license') != LICENSE_TYPE:
                                    continue
                                
                                # Verify this is a plant observation
                                iconic_taxon = row.get('iconic_taxon_name', '').strip()
                                if not iconic_taxon:
                                    empty_taxon_count += 1
                                    continue
                                    
                                # Standardize taxon case to avoid issues
                                iconic_taxon_lower = iconic_taxon.lower()
                                if iconic_taxon_lower != 'plantae':
                                    non_plant_count += 1
                                    continue
                                else:
                                    plant_count += 1
                                
                                # Get scientific name - crucial for organizing by species
                                taxa_name = row.get('scientific_name', '').strip()
                                if not taxa_name:
                                    # If scientific name is missing, try species_guess or common_name as fallback
                                    taxa_name = row.get('species_guess', '').strip()
                                    if not taxa_name:
                                        taxa_name = row.get('common_name', '').strip()
                                        if not taxa_name:
                                            taxa_name = f"Unknown_Plant_{obs_id}"
                                
                                # Get the image URL
                                image_url = row.get('image_url', '').strip()
                                if not image_url:
                                    empty_url_count += 1
                                    continue
                                
                                # Validate and fix URL if needed
                                if not image_url.startswith(('http://', 'https://')):
                                    if image_url.startswith('//'):
                                        image_url = 'https:' + image_url
                                    else:
                                        # Skip invalid URLs
                                        print(f"{TermColors.YELLOW}Invalid URL format: {image_url} for {obs_id}{TermColors.ENDC}")
                                        continue
                                
                                # Clean up name for folder name - consistent naming for species folders
                                class_name = ''.join(c for c in taxa_name.replace(' ', '_') 
                                                    if c.isalnum() or c in ['_', '-'])
                                
                                # Store the image URL with the observation data
                                if obs_id not in observation_data:
                                    observation_data[obs_id] = {
                                        'class_name': class_name,
                                        'scientific_name': taxa_name,
                                        'image_urls': [image_url]
                                    }
                                else:
                                    # Just in case there are multiple images for one observation in the CSV
                                    observation_data[obs_id]['image_urls'].append(image_url)
                                
                                # Track how many observations we have for each species
                                if class_name not in species_counts:
                                    species_counts[class_name] = 0
                                species_counts[class_name] += 1
                                
                                pbar.update(1)
                                
                                # Check if we've reached the limit
                                if len(observation_data) >= target_count:
                                    break
                                    
                            except Exception as e:
                                print(f"{TermColors.YELLOW}Error processing observation row {row_count}: {e}{TermColors.ENDC}")
                                continue
                        
                    # Print summary stats 
                    print(f"{TermColors.GREEN}CSV Processing Summary:{TermColors.ENDC}")
                    print(f"- Total rows processed: {row_count}")
                    print(f"- Plant observations found: {plant_count}")
                    print(f"- Non-plant observations skipped: {non_plant_count}")
                    print(f"- Observations with missing URLs skipped: {empty_url_count}")
                    print(f"- Observations with missing taxonomy skipped: {empty_taxon_count}")
                        
                    # If we get here with data, we've successfully processed the file
                    if observation_data:
                        break
                        
            except Exception as e:
                print(f"{TermColors.YELLOW}Failed with {encoding} encoding: {str(e)}{TermColors.ENDC}")
                continue
        
        # Add debug log to help identify if we're getting any data
        print(f"{TermColors.BLUE}After CSV processing: found {len(observation_data)} observations{TermColors.ENDC}")
        
        if not observation_data:
            # Try one more approach with pandas if all above fail
            try:
                print(f"{TermColors.BLUE}Trying to open CSV with pandas...{TermColors.ENDC}")
                import pandas as pd
                df = pd.read_csv(csv_file)
                print(f"{TermColors.GREEN}Successfully read CSV with pandas. Columns: {df.columns.tolist()}{TermColors.ENDC}")
                
                # Process with pandas
                plant_df = df[df['iconic_taxon_name'].str.lower() == 'plantae'] if 'iconic_taxon_name' in df.columns else df
                
                for _, row in tqdm(plant_df.iterrows(), total=len(plant_df), desc="Processing with pandas"):
                    try:
                        obs_id = str(row.get('id', '')).strip()
                        if not obs_id:
                            continue
                            
                        taxa_name = str(row.get('scientific_name', '')).strip()
                        if not taxa_name:
                            taxa_name = str(row.get('species_guess', '')).strip() 
                            if not taxa_name:
                                taxa_name = str(row.get('common_name', '')).strip()
                                if not taxa_name:
                                    taxa_name = f"Unknown_Plant_{obs_id}"
                                    
                        image_url = str(row.get('image_url', '')).strip()
                        
                        if not image_url:
                            continue
                            
                        # Validate URL
                        if not image_url.startswith(('http://', 'https://')):
                            if image_url.startswith('//'):
                                image_url = 'https:' + image_url
                            else:
                                continue
                            
                        class_name = ''.join(c for c in taxa_name.replace(' ', '_') 
                                            if c.isalnum() or c in ['_', '-'])
                        
                        if obs_id not in observation_data:
                            observation_data[obs_id] = {
                                'class_name': class_name,
                                'scientific_name': taxa_name,
                                'image_urls': [image_url]
                            }
                            
                            # Track how many observations we have for each species
                            if class_name not in species_counts:
                                species_counts[class_name] = 0
                            species_counts[class_name] += 1
                            
                            # Check if we've reached the limit
                            if len(observation_data) >= target_count:
                                break
                                
                    except Exception as e:
                        print(f"{TermColors.YELLOW}Error processing pandas row: {e}{TermColors.ENDC}")
                        continue
            except Exception as e:
                print(f"{TermColors.RED}Failed with pandas approach: {str(e)}{TermColors.ENDC}")
        
        # Add fallback to create dummy data if no data was extracted
        if not observation_data:
            print(f"{TermColors.YELLOW}No data found in CSV - Creating dummy test data{TermColors.ENDC}")
            # Create some dummy data for testing
            observation_data["test_1"] = {
                "class_name": "test_plant",
                "scientific_name": "Testus plantus",
                "image_urls": ["https://inaturalist-open-data.s3.amazonaws.com/photos/1/medium.jpg"]
            }
            observation_ids = ["test_1"]
            return observation_ids, observation_data
        
        # Report on species distribution
        print(f"{TermColors.GREEN}Found {len(species_counts)} unique plant species{TermColors.ENDC}")
        
        # Show distribution of species
        top_species = sorted(species_counts.items(), key=lambda x: x[1], reverse=True)[:10]
        print(f"{TermColors.GREEN}Top 10 species by observation count:{TermColors.ENDC}")
        for species, count in top_species:
            print(f"- {species}: {count} observations")
        
        # If we have too many, randomly sample
        observation_ids = list(observation_data.keys())
        if len(observation_ids) > target_count:
            print(f"{TermColors.YELLOW}Randomly sampling {target_count} observations from {len(observation_ids)}{TermColors.ENDC}")
            sampled_ids = random.sample(observation_ids, target_count)
            observation_data = {k: observation_data[k] for k in sampled_ids}
            observation_ids = sampled_ids
        
        print(f"{TermColors.GREEN}Extracted {len(observation_ids)} observations{TermColors.ENDC}")
        return observation_ids, observation_data
    
    except Exception as e:
        print(f"{TermColors.RED}Error loading CSV: {str(e)}{TermColors.ENDC}")
        traceback.print_exc()
        # Return dummy data as fallback
        print(f"{TermColors.YELLOW}Error occurred - Creating dummy test data{TermColors.ENDC}")
        observation_data = {"test_1": {
            "class_name": "test_plant",
            "scientific_name": "Testus plantus",
            "image_urls": ["https://inaturalist-open-data.s3.amazonaws.com/photos/1/medium.jpg"]
        }}
        return ["test_1"], observation_data

def init_worker():
        """Initialize worker process safely"""
        # Set policy for Windows to use SelectEventLoopPolicy in each worker process
        if sys.platform == 'win32':
            asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
        
        # Create a new event loop for this process
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        # Set up process-specific signal handlers
        signal.signal(signal.SIGINT, signal.SIG_IGN)  # Parent process will handle this

def process_observation_chunk(chunk_id, observation_ids, obs_data, output_dir, quality):
    """Process a chunk of observations in a separate process"""
    try:
        print(f"{TermColors.BLUE}Process {chunk_id}: Starting with {len(observation_ids)} observations{TermColors.ENDC}")
        
        # Create a pipeline specific to this process
        pipeline = DownloadPipeline(
            prebatcher_count=max(2, PREBATCHER_COUNT // 2),
            queue_size=QUEUE_SIZE_PER_PREBATCHER
        )
        
        # Start the pipeline for this chunk
        supervisor = pipeline.start(observation_ids, obs_data, output_dir, quality)
        
        # Wait for completion
        while supervisor.is_alive():
            try:
                supervisor.join(1.0)
            except KeyboardInterrupt:
                # This shouldn't happen in child processes, but just in case
                print(f"{TermColors.YELLOW}Process {chunk_id}: Interrupt received. Stopping...{TermColors.ENDC}")
                pipeline.terminator.terminate = True
                supervisor.join(10.0)
                break
        
        print(f"{TermColors.GREEN}Process {chunk_id}: Completed processing {len(observation_ids)} observations{TermColors.ENDC}")
        return True
        
    except Exception as e:
        print(f"{TermColors.RED}Process {chunk_id} error: {e}{TermColors.ENDC}")
        traceback.print_exc()
        return False

def process_observations_from_csv(csv_file, output_dir=OUTPUT_DIR, quality=IMAGE_QUALITY):
    """Process observations with the advanced multi-stage pipeline"""
    print(f"{TermColors.BLUE}Processing observations from {csv_file}{TermColors.ENDC}")
    print(f"{TermColors.BLUE}Target directory: {output_dir}{TermColors.ENDC}")
    
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Extract observation IDs and data
    observation_ids, obs_data = extract_observation_ids_from_csv(csv_file)
    
    if not observation_ids:
        print(f"{TermColors.RED}No observations found to process{TermColors.ENDC}")
        return False
    
    pipeline = DownloadPipeline(
        prebatcher_count=PREBATCHER_COUNT,
        queue_size=QUEUE_SIZE_PER_PREBATCHER
    )
    
    # Start the pipeline
    supervisor = pipeline.start(observation_ids, obs_data, output_dir, quality)
    
    # Wait for completion
    try:
        while supervisor.is_alive():
            try:
                supervisor.join(1.0)
            except KeyboardInterrupt:
                print(f"{TermColors.YELLOW}Keyboard interrupt received. Gracefully stopping...{TermColors.ENDC}")
                pipeline.terminator.terminate = True
                supervisor.join(30.0)
                break
    except Exception as e:
        print(f"{TermColors.RED}Error: {e}{TermColors.ENDC}")
        return False
    
    return True

def main():
    """Main entry point"""
    print(f"{TermColors.GREEN}===== iNaturalist Image Downloader ====={TermColors.ENDC}")
    print(f"{TermColors.GREEN}Pipeline configuration:{TermColors.ENDC}")
    print(f"  {TermColors.CYAN}Max memory usage: {MEMORY_LIMIT_PERCENT}%{TermColors.ENDC}")
    print(f"  {TermColors.CYAN}Parallel downloads: {MAX_PARALLEL_DOWNLOADS}{TermColors.ENDC}")
    print(f"  {TermColors.CYAN}Prebatchers: {PREBATCHER_COUNT}{TermColors.ENDC}")
    print(f"  {TermColors.CYAN}Max observations: {MAX_OBSERVATIONS}{TermColors.ENDC}")
    
    if os.path.exists(CSV_FILE):
        print(f"{TermColors.GREEN}Found target CSV: {CSV_FILE}{TermColors.ENDC}")
        process_observations_from_csv(CSV_FILE)
    else:
        print(f"{TermColors.RED}Target CSV file not found: {CSV_FILE}{TermColors.ENDC}")

if __name__ == "__main__":
    main()
