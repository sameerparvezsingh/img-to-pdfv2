import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk, ExifTags, ImageOps
import os
import threading
import time
from reportlab.lib.pagesizes import letter, A4
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader
import io
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
import logging
from datetime import datetime
import psutil
import gc
import traceback
import mimetypes
from pathlib import Path
import numpy as np
import sys 

# Optional imports with fallback
try:
    import pillow_heif
    pillow_heif.register_heif_opener()
    HEIF_SUPPORT = True
except ImportError:
    HEIF_SUPPORT = False
    
try:
    from PIL import ImageCms
    COLOR_MANAGEMENT = True
except ImportError:
    COLOR_MANAGEMENT = False

# Production logging setup
log_dir = Path("logs")
log_dir.mkdir(exist_ok=True)
log_filename = log_dir / f"image_to_pdf_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_filename),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Log system info
logger.info(f"Starting Image to PDF Converter")
logger.info(f"Python version: {sys.version}")
logger.info(f"PIL version: {Image.__version__}")
logger.info(f"HEIF support: {HEIF_SUPPORT}")
logger.info(f"Color management: {COLOR_MANAGEMENT}")

class Settings:
    """Application settings with production defaults"""
    # Image processing
    DEFAULT_QUALITY = 85
    MAX_IMAGE_SIZE = (1920, 1080)
    THUMBNAIL_SIZE = (150, 200)
    MAX_IMAGE_DIMENSION = 10000  # Maximum dimension for safety
    
    # Resource management
    MAX_CPU_PERCENT = 70
    MAX_MEMORY_MB = 1024
    BATCH_SIZE = 5
    MAX_WORKERS = max(1, min(multiprocessing.cpu_count() // 2, 4))
    PROCESS_DELAY = 0.1
    
    # File handling
    MAX_FILE_SIZE_MB = 100  # Maximum individual file size
    SUPPORTED_FORMATS = {
        '.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff', '.tif', 
        '.webp', '.ico', '.heic', '.heif', '.avif'
    }
    
    # PDF settings
    DEFAULT_DPI = 72
    MAX_PDF_SIZE_MB = 500  # Warning threshold

class ImageProcessor:
    """Enhanced image processor with EXIF and alpha channel handling"""
    
    def __init__(self):
        self.max_workers = Settings.MAX_WORKERS
        self._init_color_profiles()
        
    def _init_color_profiles(self):
        """Initialize color profiles for color management"""
        if COLOR_MANAGEMENT:
            try:
                # Standard sRGB profile
                self.srgb_profile = ImageCms.createProfile("sRGB")
            except:
                self.srgb_profile = None
        else:
            self.srgb_profile = None
    
    @staticmethod
    def get_image_info(path):
        """Get detailed image information"""
        try:
            with Image.open(path) as img:
                info = {
                    'path': path,
                    'format': img.format,
                    'mode': img.mode,
                    'size': img.size,
                    'has_alpha': img.mode in ('RGBA', 'LA', 'P'),
                    'has_transparency': 'transparency' in img.info,
                    'exif': {}
                }
                
                # Extract EXIF data
                if hasattr(img, '_getexif') and img._getexif():
                    exif = img._getexif()
                    for tag, value in exif.items():
                        tag_name = ExifTags.TAGS.get(tag, tag)
                        info['exif'][tag_name] = value
                
                return info
        except Exception as e:
            logger.error(f"Error getting image info for {path}: {e}")
            return None
    
    @staticmethod
    def handle_exif_orientation(img):
        """Handle EXIF orientation to correct image rotation"""
        try:
            if hasattr(img, '_getexif') and img._getexif():
                exif = dict(img._getexif().items())
                
                # Check for orientation tag
                orientation_tag = 274  # Standard EXIF orientation tag
                if orientation_tag in exif:
                    orientation = exif[orientation_tag]
                    
                    # Apply rotation based on orientation
                    if orientation == 2:
                        img = img.transpose(Image.FLIP_LEFT_RIGHT)
                    elif orientation == 3:
                        img = img.rotate(180, expand=True)
                    elif orientation == 4:
                        img = img.transpose(Image.FLIP_TOP_BOTTOM)
                    elif orientation == 5:
                        img = img.rotate(-90, expand=True).transpose(Image.FLIP_LEFT_RIGHT)
                    elif orientation == 6:
                        img = img.rotate(-90, expand=True)
                    elif orientation == 7:
                        img = img.rotate(90, expand=True).transpose(Image.FLIP_LEFT_RIGHT)
                    elif orientation == 8:
                        img = img.rotate(90, expand=True)
                        
            return img
        except Exception as e:
            logger.debug(f"Error handling EXIF orientation: {e}")
            return img
    
    @staticmethod
    def handle_alpha_channel(img, background_color=(255, 255, 255)):
        """Handle alpha channel and transparency"""
        if img.mode in ('RGBA', 'LA'):
            # Create background
            background = Image.new('RGB', img.size, background_color)
            
            # Paste image with alpha channel as mask
            if img.mode == 'RGBA':
                background.paste(img, mask=img.split()[3])
            else:  # LA mode
                background.paste(img, mask=img.split()[1])
            
            return background
        elif img.mode == 'P':
            # Handle palette mode with potential transparency
            img = img.convert('RGBA')
            return ImageProcessor.handle_alpha_channel(img, background_color)
        elif img.mode not in ('RGB', 'L'):
            # Convert other modes to RGB
            return img.convert('RGB')
        
        return img
    
    @staticmethod
    def apply_color_profile(img):
        """Apply color profile management"""
        if COLOR_MANAGEMENT and img.mode == 'RGB':
            try:
                # Check if image has embedded profile
                if 'icc_profile' in img.info:
                    input_profile = ImageCms.ImageCmsProfile(io.BytesIO(img.info['icc_profile']))
                    output_profile = ImageCms.createProfile("sRGB")
                    
                    # Convert to sRGB
                    img = ImageCms.profileToProfile(img, input_profile, output_profile)
                    
            except Exception as e:
                logger.debug(f"Color profile conversion error: {e}")
        
        return img
    
    def preprocess_images_batch(self, file_paths, target_size=None, quality=85, 
                               progress_callback=None, cancel_check=None):
        """Batch process images with comprehensive handling"""
        if target_size is None:
            target_size = Settings.MAX_IMAGE_SIZE
            
        total = len(file_paths)
        processed = []
        errors = []
        
        logger.info(f"Starting batch processing of {total} images with {self.max_workers} workers")
        
        try:
            # Process images in batches
            for batch_start in range(0, total, Settings.BATCH_SIZE):
                if cancel_check and cancel_check():
                    logger.info("Processing cancelled by user")
                    break
                
                # Resource throttling
                while ResourceMonitor.should_throttle():
                    logger.debug("System under load, throttling...")
                    time.sleep(0.5)
                    if cancel_check and cancel_check():
                        break
                
                batch_end = min(batch_start + Settings.BATCH_SIZE, total)
                batch_files = file_paths[batch_start:batch_end]
                
                logger.info(f"Processing batch {batch_start}-{batch_end} of {total}")
                
                # Process batch
                with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
                    futures = {}
                    
                    for i, path in enumerate(batch_files, batch_start):
                        future = executor.submit(
                            self._process_single_image_safe, 
                            path, target_size, quality
                        )
                        futures[future] = (i, path)
                    
                    # Process completed futures
                    for future in as_completed(futures):
                        i, path = futures[future]
                        
                        try:
                            result, error = future.result(timeout=30)
                            
                            if result:
                                processed.append(result)
                            else:
                                errors.append((path, error))
                                
                            if progress_callback:
                                progress_callback(
                                    len(processed) + len(errors), 
                                    total, 
                                    f"Processing {os.path.basename(path)}",
                                    f"CPU: {ResourceMonitor.get_cpu_usage():.1f}% | Memory: {ResourceMonitor.get_memory_usage():.0f}MB"
                                )
                                
                        except Exception as e:
                            error_msg = f"Error processing {path}: {str(e)}"
                            logger.error(error_msg)
                            errors.append((path, str(e)))
                
                # Delay and cleanup
                time.sleep(Settings.PROCESS_DELAY)
                gc.collect()
                
        except Exception as e:
            error_msg = f"Batch processing error: {str(e)}"
            logger.error(error_msg)
            raise Exception(error_msg)
        
        logger.info(f"Processed {len(processed)} images successfully, {len(errors)} errors")
        
        return processed, errors
    
    @staticmethod
    def _process_single_image_safe(path, target_size, quality):
        """Safely process single image with comprehensive error handling"""
        try:
            return ImageProcessor._process_single_image(path, target_size, quality), None
        except Exception as e:
            error_msg = f"Failed to process {path}: {str(e)}"
            logger.error(error_msg)
            logger.debug(traceback.format_exc())
            return None, str(e)
    
    @staticmethod
    def _process_single_image(path, target_size, quality):
        """Process single image with all enhancements"""
        # Validate file
        if not os.path.exists(path):
            raise FileNotFoundError(f"File not found: {path}")
        
        if not os.access(path, os.R_OK):
            raise PermissionError(f"Cannot read file: {path}")
        
        # Check file size
        file_size_mb = os.path.getsize(path) / 1024 / 1024
        if file_size_mb > Settings.MAX_FILE_SIZE_MB:
            raise ValueError(f"File too large ({file_size_mb:.1f}MB > {Settings.MAX_FILE_SIZE_MB}MB)")
        
        # Validate file type
        file_ext = os.path.splitext(path)[1].lower()
        if file_ext not in Settings.SUPPORTED_FORMATS:
            # Try to detect by content
            mime_type = mimetypes.guess_type(path)[0]
            if not mime_type or not mime_type.startswith('image/'):
                raise ValueError(f"Unsupported file format: {file_ext}")
        
        with Image.open(path) as img:
            # Validate image
            img.verify()
            img = Image.open(path)  # Reopen after verify
            
            # Check image dimensions
            if max(img.size) > Settings.MAX_IMAGE_DIMENSION:
                raise ValueError(f"Image too large: {img.size}")
            
            # Handle EXIF orientation
            img = ImageProcessor.handle_exif_orientation(img)
            
            # Apply color profile
            img = ImageProcessor.apply_color_profile(img)
            
            # Handle alpha channel and transparency
            img = ImageProcessor.handle_alpha_channel(img)
            
            # Resize if needed
            img_width, img_height = img.size
            max_width, max_height = target_size
            
            if img_width > max_width or img_height > max_height:
                # Calculate new size maintaining aspect ratio
                ratio = min(max_width/img_width, max_height/img_height)
                new_size = (int(img_width * ratio), int(img_height * ratio))
                
                # Use high-quality resampling
                img = img.resize(new_size, Image.Resampling.LANCZOS)
                
                # Sharpen slightly after resize
                try:
                    from PIL import ImageEnhance
                    enhancer = ImageEnhance.Sharpness(img)
                    img = enhancer.enhance(1.1)
                except:
                    pass
            
            # Convert to RGB if not already
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            # Save to bytes with optimization
            buffer = io.BytesIO()
            
            # Determine best format
            save_kwargs = {
                'format': 'JPEG',
                'quality': quality,
                'optimize': True,
                'progressive': True
            }
            
            # Add DPI info
            save_kwargs['dpi'] = (Settings.DEFAULT_DPI, Settings.DEFAULT_DPI)
            img.save(buffer, **save_kwargs)
            buffer.seek(0)
            
            return {
                'data': buffer.getvalue(),
                'size': img.size,
                'format': 'JPEG',
                'path': path,
                'original_format': img.format,
                'file_size': len(buffer.getvalue())
            }

class ResourceMonitor:
    """Monitor system resources"""
    
    @staticmethod
    def get_cpu_usage():
        """Get current CPU usage percentage"""
        return psutil.cpu_percent(interval=0.1)
    
    @staticmethod
    def get_memory_usage():
        """Get current memory usage in MB"""
        process = psutil.Process()
        return process.memory_info().rss / 1024 / 1024
    
    @staticmethod
    def should_throttle():
        """Check if we should throttle processing"""
        cpu_usage = ResourceMonitor.get_cpu_usage()
        memory_usage = ResourceMonitor.get_memory_usage()
        
        return (cpu_usage > Settings.MAX_CPU_PERCENT or 
                memory_usage > Settings.MAX_MEMORY_MB)

class UIStateMixin:
    """Mixin class for UI state management"""
    
    def __init__(self):
        self.ui_elements = []
        self.operation_in_progress = False
        self._lock = threading.Lock()
        self.drag_drop_enabled = True
        self._cancel_requested = False
        
    def register_ui_element(self, element):
        """Register UI element for state management"""
        if element and element not in self.ui_elements:
            self.ui_elements.append(element)
            
    def set_ui_state(self, enabled):
        """Enable/disable all registered UI elements"""
        state = 'normal' if enabled else 'disabled'
        
        # Update drag and drop state
        self.drag_drop_enabled = enabled
        
        for element in self.ui_elements:
            try:
                if hasattr(element, 'config'):
                    element.config(state=state)
            except Exception as e:
                logger.debug(f"Could not set state for element: {e}")
                
        # Special handling for listbox
        if hasattr(self, 'image_listbox'):
            if enabled:
                self.image_listbox.config(state='normal')
                self.bind_drag_drop_events()
            else:
                self.image_listbox.config(state='disabled')
                self.unbind_drag_drop_events()
                
        # Update cancel button
        if hasattr(self, 'cancel_button'):
            self.cancel_button.config(state='normal' if not enabled else 'disabled')
                
    def bind_drag_drop_events(self):
        """Bind drag and drop events to listbox"""
        self.image_listbox.bind('<Button-1>', self.on_image_click)
        self.image_listbox.bind('<B1-Motion>', self.on_image_drag)
        self.image_listbox.bind('<ButtonRelease-1>', self.on_image_drop)
        
    def unbind_drag_drop_events(self):
        """Unbind drag and drop events from listbox"""
        self.image_listbox.unbind('<Button-1>')
        self.image_listbox.unbind('<B1-Motion>')
        self.image_listbox.unbind('<ButtonRelease-1>')
                
    def operation_started(self):
        """Call when operation starts"""
        with self._lock:
            self.operation_in_progress = True
            self._cancel_requested = False
            
        self.root.after(0, lambda: self.set_ui_state(False))
        self.root.after(0, lambda: self.root.config(cursor="watch"))
            
    def operation_completed(self):
        """Call when operation completes"""
        with self._lock:
            self.operation_in_progress = False
            self._cancel_requested = False
                
        self.root.after(0, lambda: self.set_ui_state(True))
        self.root.after(0, lambda: self.root.config(cursor=""))
                
    def check_operation_in_progress(self):
        """Check if any operation is in progress"""
        if self.operation_in_progress:
            messagebox.showwarning("Warning", "Another operation is in progress. Please wait.")
            return True
        return False
    
    def request_cancel(self):
        """Request cancellation of current operation"""
        with self._lock:
            self._cancel_requested = True
        logger.info("Cancellation requested by user")

class EnhancedProgressBar:
    """Enhanced progress bar with ETA calculation and detailed status"""
    
    def __init__(self, parent, root):
        self.root = root
        self.frame = ttk.Frame(parent)
        self.progress = ttk.Progressbar(self.frame, mode='determinate')
        self.label = tk.Label(self.frame, text="Ready")
        self.detail_label = tk.Label(self.frame, text="", font=('Helvetica', 9))
        self.start_time = None
        
        self.label.pack(fill='x', padx=5, pady=(0, 2))
        self.detail_label.pack(fill='x', padx=5, pady=(0, 5))
        self.progress.pack(fill='x', padx=5)
        
    def start(self, total, message="Processing..."):
        self.progress['maximum'] = total
        self.progress['value'] = 0
        self.start_time = time.time()
        self.label.config(text=message)
        self.detail_label.config(text="")
        self.root.update_idletasks()
        
    def update(self, current, message="", detail=""):
        try:
            self.progress['value'] = current
            
            if self.start_time and current > 0 and current < self.progress['maximum']:
                elapsed = time.time() - self.start_time
                rate = current / elapsed
                remaining = (self.progress['maximum'] - current) / rate
                eta = time.strftime('%M:%S', time.gmtime(remaining))
                status_text = f"{message} - ETA: {eta}" if message else f"Processing... ETA: {eta}"
            else:
                status_text = message if message else "Processing..."
                
            self.label.config(text=status_text)
            
            if detail:
                self.detail_label.config(text=detail)
                
            self.root.update_idletasks()
        except Exception as e:
            logger.error(f"Progress update error: {e}")
            
    def complete(self, message="Complete"):
        self.progress['value'] = self.progress['maximum']
        self.label.config(text=message)
        self.detail_label.config(text="")
        self.root.update_idletasks()
        
    def error(self, message="Error occurred"):
        self.label.config(text=message, fg='red')
        self.detail_label.config(text="")
        self.root.update_idletasks()
        
    def reset(self):
        self.progress['value'] = 0
        self.label.config(text="Ready", fg='black')
        self.detail_label.config(text="")
        self.start_time = None
        
    def pack(self, **kwargs):
        self.frame.pack(**kwargs)

class ImageToPDFTool(UIStateMixin):
    def __init__(self, root):
        super().__init__()
        self.root = root
        self.root.title("Image to PDF Converter Pro")
        self.root.geometry("950x900")
        
        # Configure grid weights
        self.root.grid_rowconfigure(0, weight=1)
        self.root.grid_columnconfigure(0, weight=1)
        
        # Initialize components
        self.image_processor = ImageProcessor()
        
        # File list
        self.image_files = []
        
        # Setup UI
        self.setup_ui()
        self.register_all_ui_elements()
        
        # Drag and drop variables
        self.drag_start_index = None
        
        # Apply styles
        self.apply_styles()
        
        # Show supported formats
        self.show_format_support()
        
        # Log startup
        logger.info(f"Image to PDF Converter Pro started - Version 2.0")
        logger.info(f"Max workers: {Settings.MAX_WORKERS}, Batch size: {Settings.BATCH_SIZE}")
        
    def apply_styles(self):
        style = ttk.Style()
        style.theme_use('clam')
        
        bg_color = '#f0f0f0'
        self.root.configure(bg=bg_color)
        
        style.configure('Title.TLabel', font=('Helvetica', 16, 'bold'))
        style.configure('Success.TButton', font=('Helvetica', 10, 'bold'))
        style.configure('Cancel.TButton', font=('Helvetica', 10))
        style.configure('Warning.TLabel', foreground='orange')
        style.configure('Error.TLabel', foreground='red')

    def setup_ui(self):
        # Main container
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky="nsew")
        
        # Configure main frame grid
        main_frame.grid_rowconfigure(1, weight=1)
        main_frame.grid_columnconfigure(0, weight=1)
        
        # Title and format info
        title_frame = ttk.Frame(main_frame)
        title_frame.grid(row=0, column=0, sticky="ew", pady=(0, 10))
        
        ttk.Label(title_frame, text="Image to PDF Converter Pro", style='Title.TLabel').pack(side='left')
        
        self.format_label = ttk.Label(title_frame, text="", font=('Helvetica', 9))
        self.format_label.pack(side='right', padx=20)
        
        # File selection frame
        file_frame = ttk.Frame(main_frame)
        file_frame.grid(row=1, column=0, sticky="nsew", pady=10)
        
        # Configure file frame grid
        file_frame.grid_rowconfigure(0, weight=1)
        file_frame.grid_columnconfigure(1, weight=1)
        
        # Left panel - buttons
        button_frame = ttk.Frame(file_frame)
        button_frame.grid(row=0, column=0, sticky="ns", padx=(0, 10))
        
        # File operations
        file_ops_frame = ttk.LabelFrame(button_frame, text="File Operations", padding=5)
        file_ops_frame.pack(fill='x', pady=(0, 5))
        
        self.select_images_button = ttk.Button(file_ops_frame, text="Add Images", 
                                              command=self.select_image_files)
        self.select_images_button.pack(fill='x', pady=2)
        
        self.add_folder_button = ttk.Button(file_ops_frame, text="Add Folder", 
                                           command=self.add_folder_images)
        self.add_folder_button.pack(fill='x', pady=2)
        
        self.clear_images_button = ttk.Button(file_ops_frame, text="Clear All", 
                                            command=self.clear_image_list)
        self.clear_images_button.pack(fill='x', pady=2)
        
        self.remove_selected_button = ttk.Button(file_ops_frame, text="Remove Selected", 
                                               command=self.remove_selected_images)
        self.remove_selected_button.pack(fill='x', pady=2)
        
        # Order operations
        order_ops_frame = ttk.LabelFrame(button_frame, text="Change Order", padding=5)
        order_ops_frame.pack(fill='x', pady=5)
        
        self.move_image_up_button = ttk.Button(order_ops_frame, text="⬆ Move Up", 
                                             command=self.move_image_up)
        self.move_image_up_button.pack(fill='x', pady=2)
        
        self.move_image_down_button = ttk.Button(order_ops_frame, text="⬇ Move Down", 
                                               command=self.move_image_down)
        self.move_image_down_button.pack(fill='x', pady=2)
        
        self.sort_name_button = ttk.Button(order_ops_frame, text="Sort by Name", 
                                         command=self.sort_by_name)
        self.sort_name_button.pack(fill='x', pady=2)
        
        self.sort_date_button = ttk.Button(order_ops_frame, text="Sort by Date", 
                                         command=self.sort_by_date)
        self.sort_date_button.pack(fill='x', pady=2)
        
        self.reverse_order_button = ttk.Button(order_ops_frame, text="Reverse Order", 
                                             command=self.reverse_image_order)
        self.reverse_order_button.pack(fill='x', pady=2)
        
        # System info
        info_frame = ttk.LabelFrame(button_frame, text="System Info", padding=5)
        info_frame.pack(fill='x', pady=5)
        
        self.cpu_label = ttk.Label(info_frame, text="CPU: 0%", font=('Helvetica', 9))
        self.cpu_label.pack(anchor='w', pady=1)
        self.memory_label = ttk.Label(info_frame, text="Memory: 0 MB", font=('Helvetica', 9))
        self.memory_label.pack(anchor='w', pady=1)
        self.status_label = ttk.Label(info_frame, text="Ready", font=('Helvetica', 9, 'bold'))
        self.status_label.pack(anchor='w', pady=1)
        
        # Update system info periodically
        self.update_system_info()
        
        # Right panel - image list
        list_frame = ttk.Frame(file_frame)
        list_frame.grid(row=0, column=1, sticky="nsew")
        
        # Configure list frame grid
        list_frame.grid_rowconfigure(1, weight=1)
        list_frame.grid_columnconfigure(0, weight=1)
        
        # List header
        header_frame = ttk.Frame(list_frame)
        header_frame.grid(row=0, column=0, sticky="ew", pady=(0, 5))
        
        ttk.Label(header_frame, text="Images (drag to reorder):", font=('Helvetica', 10, 'bold')).pack(side='left')
        
        # Preview button
        self.preview_button = ttk.Button(header_frame, text="Preview Selected", 
                                       command=self.preview_selected_image)
        self.preview_button.pack(side='right', padx=5)
        
        # Listbox with scrollbar
        listbox_frame = ttk.Frame(list_frame)
        listbox_frame.grid(row=1, column=0, sticky="nsew")
        
        listbox_frame.grid_rowconfigure(0, weight=1)
        listbox_frame.grid_columnconfigure(0, weight=1)
        
        # Create listbox with extended selection
        self.image_listbox = tk.Listbox(listbox_frame, selectmode='extended')
        scrollbar = ttk.Scrollbar(listbox_frame, orient="vertical")
        
        self.image_listbox.config(yscrollcommand=scrollbar.set)
        scrollbar.config(command=self.image_listbox.yview)
        
        self.image_listbox.grid(row=0, column=0, sticky="nsew")
        scrollbar.grid(row=0, column=1, sticky="ns")
        
        # Bind events
        self.bind_drag_drop_events()
        self.image_listbox.bind('<Double-Button-1>', lambda e: self.preview_selected_image())
        self.image_listbox.bind('<Delete>', lambda e: self.remove_selected_images())
        
        # Image count and stats
        self.stats_label = ttk.Label(list_frame, text="No images loaded", font=('Helvetica', 9))
        self.stats_label.grid(row=2, column=0, pady=(5, 0), sticky="w")
        
        # Options frame
        options_frame = ttk.LabelFrame(main_frame, text="Conversion Options", padding=10)
        options_frame.grid(row=2, column=0, sticky="ew", pady=10)
        
        # Compression settings
        compress_frame = ttk.Frame(options_frame)
        compress_frame.pack(fill='x', pady=(0, 10))
        
        self.compress_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(compress_frame, text="Compress images", 
                       variable=self.compress_var,
                       command=self.toggle_compression).pack(side='left')
        
        ttk.Label(compress_frame, text="Quality:").pack(side='left', padx=(20, 5))
        self.quality_var = tk.IntVar(value=85)
        self.quality_scale = ttk.Scale(compress_frame, from_=10, to=100, 
                                      variable=self.quality_var, orient='horizontal', length=200)
        self.quality_scale.pack(side='left', padx=5)
        self.quality_label = ttk.Label(compress_frame, text="85%")
        self.quality_label.pack(side='left')
        
        self.quality_scale.config(command=lambda v: self.quality_label.config(text=f"{int(float(v))}%"))
        
        # Page settings
        page_frame = ttk.Frame(options_frame)
        page_frame.pack(fill='x', pady=(0, 10))
        
        ttk.Label(page_frame, text="Page Size:").pack(side='left', padx=(0, 10))
        self.page_size_var = tk.StringVar(value="A4")
        ttk.Radiobutton(page_frame, text="A4", variable=self.page_size_var, 
                       value="A4").pack(side='left', padx=5)
        ttk.Radiobutton(page_frame, text="Letter", variable=self.page_size_var, 
                       value="Letter").pack(side='left', padx=5)
        ttk.Radiobutton(page_frame, text="Fit to Image", variable=self.page_size_var, 
                       value="Fit").pack(side='left', padx=5)
        
        # Advanced options
        advanced_frame = ttk.Frame(options_frame)
        advanced_frame.pack(fill='x')
        
        self.preserve_exif_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(advanced_frame, text="Auto-rotate (EXIF)", 
                       variable=self.preserve_exif_var).pack(side='left', padx=(0, 20))
        
        self.optimize_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(advanced_frame, text="Optimize for web", 
                       variable=self.optimize_var).pack(side='left')
        
        # Progress bar
        self.img_progress = EnhancedProgressBar(main_frame, self.root)
        self.img_progress.frame.grid(row=3, column=0, sticky="ew", pady=10)
        
        # Action buttons
        button_container = ttk.Frame(main_frame)
        button_container.grid(row=4, column=0, pady=(0, 10))
        
        self.convert_to_pdf_button = ttk.Button(
            button_container, 
            text="Convert to PDF", 
            command=self.images_to_pdf_thread, 
            style='Success.TButton'
        )
        self.convert_to_pdf_button.pack(side='left', padx=5)
        
        self.cancel_button = ttk.Button(
            button_container,
            text="Cancel",
            command=self.request_cancel,
            style='Cancel.TButton',
            state='disabled'
        )
        self.cancel_button.pack(side='left', padx=5)
        
        # View log button
        self.view_log_button = ttk.Button(
            button_container,
            text="View Log",
            command=self.view_log_file
        )
        self.view_log_button.pack(side='left', padx=5)
        
    def show_format_support(self):
        """Show supported formats"""
        formats = []
        if HEIF_SUPPORT:
            formats.append("HEIF/HEIC ✓")
        else:
            formats.append("HEIF/HEIC ✗")
            
        self.format_label.config(text=f"Supported: JPG, PNG, BMP, GIF, TIFF, WebP, {', '.join(formats)}")
        
    def toggle_compression(self):
        """Toggle compression settings"""
        if self.compress_var.get():
            self.quality_scale.config(state='normal')
        else:
            self.quality_scale.config(state='disabled')
    
    def update_system_info(self):
        """Update system resource information"""
        try:
            cpu_usage = ResourceMonitor.get_cpu_usage()
            memory_usage = ResourceMonitor.get_memory_usage()
            
            # Color code based on usage
            cpu_color = 'black' if cpu_usage < 70 else 'orange' if cpu_usage < 90 else 'red'
            mem_color = 'black' if memory_usage < 800 else 'orange' if memory_usage < 1000 else 'red'
            
            self.cpu_label.config(text=f"CPU: {cpu_usage:.1f}%", foreground=cpu_color)
            self.memory_label.config(text=f"Memory: {memory_usage:.0f} MB", foreground=mem_color)
            
            # Update status
            if self.operation_in_progress:
                self.status_label.config(text="Processing...", foreground='blue')
            else:
                self.status_label.config(text="Ready", foreground='green')
            
            # Schedule next update
            self.root.after(1000, self.update_system_info)
        except Exception as e:
            logger.debug(f"System info update error: {e}")
        
    def register_all_ui_elements(self):
        """Register all UI elements for state management"""
        ui_elements = [
            self.select_images_button, self.add_folder_button,
            self.clear_images_button, self.remove_selected_button,
            self.move_image_up_button, self.move_image_down_button,
            self.sort_name_button, self.sort_date_button,
            self.reverse_order_button, self.convert_to_pdf_button,
            self.preview_button
        ]
        
        for element in ui_elements:
            self.register_ui_element(element)

    def select_image_files(self):
        """Select and add image files"""
        if self.operation_in_progress:
            return
            
        try:
            # Build file types string
            extensions = list(Settings.SUPPORTED_FORMATS)
            file_types_str = " ".join([f"*{ext}" for ext in extensions])
            
            files = filedialog.askopenfilenames(
                title="Select image files",
                filetypes=[
                    ("All Images", file_types_str),
                    ("JPEG files", "*.jpg *.jpeg"),
                    ("PNG files", "*.png"),
                    ("HEIF files", "*.heic *.heif"),
                    ("All files", "*.*")
                ]
            )
            
            self.add_images(files)
            
        except Exception as e:
            error_msg = f"Error selecting files: {str(e)}"
            logger.error(error_msg)
            messagebox.showerror("Error", error_msg)
    
    def add_folder_images(self):
        """Add all images from a folder"""
        if self.operation_in_progress:
            return
            
        try:
            folder_path = filedialog.askdirectory(title="Select folder containing images")
            if not folder_path:
                return
            
            # Find all image files
            image_files = []
            for root, dirs, files in os.walk(folder_path):
                for file in files:
                    if os.path.splitext(file)[1].lower() in Settings.SUPPORTED_FORMATS:
                        image_files.append(os.path.join(root, file))
                # Don't recurse into subdirectories
                break
            
            if image_files:
                self.add_images(image_files)
                messagebox.showinfo("Success", f"Added {len(image_files)} images from folder")
            else:
                messagebox.showinfo("Info", "No supported image files found in folder")
                
        except Exception as e:
            error_msg = f"Error adding folder: {str(e)}"
            logger.error(error_msg)
            messagebox.showerror("Error", error_msg)
    
    def add_images(self, files):
        """Add images to the list with validation"""
        added_count = 0
        skipped_count = 0
        errors = []
        
        for file in files:
            try:
                # Skip if already in list
                # if file in self.image_files:
                #     skipped_count += 1
                #     continue
                
                # Validate file
                if not os.path.exists(file):
                    errors.append(f"{os.path.basename(file)}: File not found")
                    continue
                
                # Check file size
                file_size_mb = os.path.getsize(file) / 1024 / 1024
                if file_size_mb > Settings.MAX_FILE_SIZE_MB:
                    errors.append(f"{os.path.basename(file)}: Too large ({file_size_mb:.1f}MB)")
                    continue
                
                # Try to get image info
                info = self.image_processor.get_image_info(file)
                if not info:
                    errors.append(f"{os.path.basename(file)}: Invalid image")
                    continue
                
                # Add to list
                self.image_files.append(file)
                display_text = f"{os.path.basename(file)} ({info['size'][0]}x{info['size'][1]})"
                self.image_listbox.insert(tk.END, display_text)
                added_count += 1
                
            except Exception as e:
                errors.append(f"{os.path.basename(file)}: {str(e)}")
        
        # Update stats
        self.update_image_stats()
        
        # Show results
        if errors:
            error_msg = "Some files could not be added:\n\n"
            for error in errors[:10]:  # Show first 10 errors
                error_msg += f"• {error}\n"
            if len(errors) > 10:
                error_msg += f"\n... and {len(errors) - 10} more"
            messagebox.showwarning("Warning", error_msg)
        
        logger.info(f"Added {added_count} images, skipped {skipped_count}, errors {len(errors)}")

    def clear_image_list(self):
        """Clear all images"""
        if self.operation_in_progress:
            return
            
        if self.image_files and messagebox.askyesno("Confirm", "Clear all images from the list?"):
            self.image_files.clear()
            self.image_listbox.delete(0, tk.END)
            self.img_progress.reset()
            self.update_image_stats()
            logger.info("Cleared all images")

    def remove_selected_images(self):
        """Remove selected images"""
        if self.operation_in_progress:
            return
            
        selected_indices = list(self.image_listbox.curselection())
        if not selected_indices:
            return
        
        # Remove in reverse order
        for idx in reversed(selected_indices):
            removed_file = self.image_files[idx]
            del self.image_files[idx]
            self.image_listbox.delete(idx)
            logger.debug(f"Removed {os.path.basename(removed_file)}")
        
        self.update_image_stats()

    def move_image_up(self):
        """Move selected image up"""
        if self.operation_in_progress:
            return
            
        selected = self.image_listbox.curselection()
        if not selected or selected[0] == 0:
            return
        
        idx = selected[0]
        # Swap in both lists
        self.image_files[idx], self.image_files[idx-1] = self.image_files[idx-1], self.image_files[idx]
        
        # Update listbox
        item = self.image_listbox.get(idx)
        self.image_listbox.delete(idx)
        self.image_listbox.insert(idx-1, item)
        self.image_listbox.selection_set(idx-1)
        self.image_listbox.see(idx-1)

    def move_image_down(self):
        """Move selected image down"""
        if self.operation_in_progress:
            return
            
        selected = self.image_listbox.curselection()
        if not selected or selected[0] >= len(self.image_files) - 1:
            return
        
        idx = selected[0]
        # Swap in both lists
        self.image_files[idx], self.image_files[idx+1] = self.image_files[idx+1], self.image_files[idx]
        
        # Update listbox
        item = self.image_listbox.get(idx)
        self.image_listbox.delete(idx)
        self.image_listbox.insert(idx+1, item)
        self.image_listbox.selection_set(idx+1)
        self.image_listbox.see(idx+1)

    def sort_by_name(self):
        """Sort images by filename"""
        if self.operation_in_progress or not self.image_files:
            return
        
        # Create sorted pairs
        sorted_pairs = sorted(zip(self.image_files, range(len(self.image_files))), 
                            key=lambda x: os.path.basename(x[0]).lower())
        
        # Update lists
        self.image_files = [pair[0] for pair in sorted_pairs]
        
        # Rebuild listbox
        self.refresh_image_listbox()
        logger.info("Sorted images by name")

    def sort_by_date(self):
        """Sort images by modification date"""
        if self.operation_in_progress or not self.image_files:
            return
        
        # Create sorted pairs
        sorted_pairs = sorted(zip(self.image_files, range(len(self.image_files))), 
                            key=lambda x: os.path.getmtime(x[0]))
        
        # Update lists
        self.image_files = [pair[0] for pair in sorted_pairs]
        
        # Rebuild listbox
        self.refresh_image_listbox()
        logger.info("Sorted images by date")

    def reverse_image_order(self):
        """Reverse image order"""
        if self.operation_in_progress or not self.image_files:
            return
        
        self.image_files.reverse()
        self.refresh_image_listbox()
        logger.info("Reversed image order")

    def refresh_image_listbox(self):
        """Refresh the listbox with current file order"""
        self.image_listbox.delete(0, tk.END)
        
        for file in self.image_files:
            try:
                info = self.image_processor.get_image_info(file)
                if info:
                    display_text = f"{os.path.basename(file)} ({info['size'][0]}x{info['size'][1]})"
                else:
                    display_text = os.path.basename(file)
                self.image_listbox.insert(tk.END, display_text)
            except:
                self.image_listbox.insert(tk.END, os.path.basename(file))

    def update_image_stats(self):
        """Update image statistics display"""
        count = len(self.image_files)
        
        if count == 0:
            self.stats_label.config(text="No images loaded")
            return
        
        # Calculate total size
        total_size = 0
        for file in self.image_files:
            try:
                total_size += os.path.getsize(file)
            except:
                pass
        
        total_size_mb = total_size / (1024 * 1024)
        
        # Estimate output size
        if self.compress_var.get():
            quality = self.quality_var.get()
            estimated_size_mb = total_size_mb * (quality / 100) * 0.3  # Rough estimate
        else:
            estimated_size_mb = total_size_mb * 0.8
        
        stats_text = f"Images: {count} | Total size: {total_size_mb:.1f} MB | Est. PDF: ~{estimated_size_mb:.1f} MB"
        self.stats_label.config(text=stats_text)

    def preview_selected_image(self):
        """Preview selected image with info"""
        selected = self.image_listbox.curselection()
        if not selected:
            messagebox.showinfo("Info", "Please select an image to preview")
            return
        
        idx = selected[0]
        image_path = self.image_files[idx]
        
        # Create preview window
        preview_window = tk.Toplevel(self.root)
        preview_window.title(f"Preview: {os.path.basename(image_path)}")
        preview_window.geometry("800x700")
        
        try:
            # Get image info
            info = self.image_processor.get_image_info(image_path)
            if not info:
                raise Exception("Could not read image")
            
            # Info frame
            info_frame = ttk.Frame(preview_window, padding=10)
            info_frame.pack(fill='x')
            
            # Display detailed info
            info_text = f"File: {os.path.basename(image_path)}\n"
            info_text += f"Format: {info.get('format', 'Unknown')}\n"
            info_text += f"Size: {info['size'][0]} x {info['size'][1]} pixels\n"
            info_text += f"Mode: {info.get('mode', 'Unknown')}\n"
            info_text += f"File size: {os.path.getsize(image_path) / 1024:.1f} KB"
            
            # Check for EXIF data
            if info.get('exif'):
                exif_info = []
                important_tags = ['Make', 'Model', 'DateTime', 'Orientation']
                for tag in important_tags:
                    if tag in info['exif']:
                        exif_info.append(f"{tag}: {info['exif'][tag]}")
                if exif_info:
                    info_text += f"\n\nEXIF Data:\n" + "\n".join(exif_info)
            
            info_label = ttk.Label(info_frame, text=info_text, justify='left')
            info_label.pack(anchor='w')
            
            # Image display frame
            img_frame = ttk.Frame(preview_window)
            img_frame.pack(fill='both', expand=True, padx=10, pady=5)
            
            # Load and display image
            with Image.open(image_path) as img:
                # Apply EXIF rotation
                img = self.image_processor.handle_exif_orientation(img)
                
                # Resize for display
                display_size = (750, 450)
                img.thumbnail(display_size, Image.Resampling.LANCZOS)
                
                # Convert to PhotoImage
                photo = ImageTk.PhotoImage(img)
                
                # Display
                img_label = ttk.Label(img_frame, image=photo)
                img_label.image = photo  # Keep reference
                img_label.pack(expand=True)
            
            # Close button
            ttk.Button(preview_window, text="Close", 
                      command=preview_window.destroy).pack(pady=10)
            
        except Exception as e:
            messagebox.showerror("Error", f"Could not preview image:\n{str(e)}")
            preview_window.destroy()
            logger.error(f"Preview error for {image_path}: {e}")

    # Drag and drop functionality
    def on_image_click(self, event):
        """Handle mouse click on image list"""
        if not self.drag_drop_enabled or self.operation_in_progress:
            return
            
        self.drag_start_index = self.image_listbox.nearest(event.y)

    def on_image_drag(self, event):
        """Handle dragging of image"""
        if not self.drag_drop_enabled or self.operation_in_progress or self.drag_start_index is None:
            return
        
        # Visual feedback
        self.image_listbox.config(cursor="hand2")
        
        # Show drag line
        current_index = self.image_listbox.nearest(event.y)
        if 0 <= current_index < len(self.image_files):
            self.image_listbox.selection_clear(0, tk.END)
            self.image_listbox.selection_set(self.drag_start_index)

    def on_image_drop(self, event):
        """Handle dropping of image"""
        if not self.drag_drop_enabled or self.operation_in_progress or self.drag_start_index is None:
            return
        
        # Reset cursor
        self.image_listbox.config(cursor="")
        
        # Get drop position
        drop_index = self.image_listbox.nearest(event.y)
        
        if drop_index != self.drag_start_index and 0 <= drop_index < len(self.image_files):
            # Move item
            item = self.image_files.pop(self.drag_start_index)
            self.image_files.insert(drop_index, item)
            
            # Update listbox
            self.refresh_image_listbox()
            
            # Select moved item
            self.image_listbox.selection_set(drop_index)
            self.image_listbox.see(drop_index)
        
        self.drag_start_index = None

    def view_log_file(self):
        """Open the log file"""
        try:
            if os.path.exists(log_filename):
                if sys.platform.startswith('win'):
                    os.startfile(log_filename)
                elif sys.platform.startswith('darwin'):
                    os.system(f'open "{log_filename}"')
                else:
                    os.system(f'xdg-open "{log_filename}"')
            else:
                messagebox.showinfo("Info", "No log file found")
        except Exception as e:
            messagebox.showerror("Error", f"Could not open log file:\n{str(e)}")

    def images_to_pdf_thread(self):
        """Thread wrapper for conversion"""
        if self.check_operation_in_progress():
            return
            
        if not self.image_files:
            messagebox.showwarning("Warning", "Please select at least one image.")
            return
        
        # Validate files exist
        missing_files = [f for f in self.image_files if not os.path.exists(f)]
        
        if missing_files:
            error_msg = f"The following files were not found:\n"
            for file in missing_files[:5]:
                error_msg += f"• {os.path.basename(file)}\n"
            if len(missing_files) > 5:
                error_msg += f"... and {len(missing_files) - 5} more"
            
            messagebox.showerror("Error", error_msg)
            
            # Remove missing files
            for file in missing_files:
                if file in self.image_files:
                    idx = self.image_files.index(file)
                    self.image_files.remove(file)
                    self.image_listbox.delete(idx)
            
            self.update_image_stats()
            
            if not self.image_files:
                return
        
        # Check estimated output size
        estimated_size = sum(os.path.getsize(f) for f in self.image_files if os.path.exists(f)) / (1024 * 1024)
        if estimated_size > Settings.MAX_PDF_SIZE_MB:
            if not messagebox.askyesno("Warning", 
                f"Estimated PDF size is large ({estimated_size:.0f} MB).\n"
                "This may take a while. Continue?"):
                return
        
        thread = threading.Thread(target=self.images_to_pdf_worker, daemon=True)
        thread.start()

    def images_to_pdf_worker(self):
        """Worker function for PDF conversion"""
        self.operation_started()
        output_path = None
        
        try:
            # Get output path
            output_path = filedialog.asksaveasfilename(
                defaultextension=".pdf",
                filetypes=[("PDF files", "*.pdf"), ("All files", "*.*")],
                initialfile=f"converted_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
            )
            
            if not output_path:
                return
            
            logger.info(f"Starting PDF creation: {output_path}")
            logger.info(f"Processing {len(self.image_files)} images")
            
            # Get settings
            compress = self.compress_var.get()
            quality = self.quality_var.get()
            page_size = self.page_size_var.get()
            preserve_exif = self.preserve_exif_var.get()
            
            # Progress callback
            def progress_callback(current, total, message="", detail=""):
                self.root.after(0, lambda: self.img_progress.update(current, message, detail))
            
            # Cancel check
            def cancel_check():
                return self._cancel_requested
            
            # Start progress
            total_steps = len(self.image_files) + 1
            self.root.after(0, lambda: self.img_progress.start(total_steps, "Processing images..."))
            
            # Process images
            processed_images, errors = self.image_processor.preprocess_images_batch(
                self.image_files,
                target_size=Settings.MAX_IMAGE_SIZE if compress else None,
                quality=quality if compress else 100,
                progress_callback=progress_callback,
                cancel_check=cancel_check
            )
            
            # Check cancellation
            if self._cancel_requested:
                self.root.after(0, lambda: self.img_progress.reset())
                self.root.after(0, lambda: messagebox.showinfo("Cancelled", "Operation was cancelled."))
                logger.info("Operation cancelled by user")
                return
            
            # Show errors
            if errors:
                error_summary = f"Failed to process {len(errors)} image(s):\n\n"
                for path, error in errors[:5]:
                    error_summary += f"• {os.path.basename(path)}: {error}\n"
                if len(errors) > 5:
                    error_summary += f"\n... and {len(errors) - 5} more errors"
                
                self.root.after(0, lambda: messagebox.showwarning("Processing Errors", error_summary))
                
                if not processed_images:
                    raise Exception("No images could be processed successfully")
            
            # Create PDF
            self.root.after(0, lambda: self.img_progress.update(
                len(processed_images), 
                "Creating PDF...",
                "Generating document..."
            ))
            
            # Determine page size
            if page_size == "A4":
                pdf_page_size = A4
            elif page_size == "Letter":
                pdf_page_size = letter
            else:
                pdf_page_size = None
            
            # Create PDF
            c = canvas.Canvas(output_path, pagesize=pdf_page_size if pdf_page_size else A4)
            
            # Add metadata
            c.setAuthor("Image to PDF Converter Pro")
            c.setTitle(f"PDF created on {datetime.now().strftime('%Y-%m-%d %H:%M')}")
            c.setSubject(f"Contains {len(processed_images)} images")
            
            # Process each image
            for idx, img_data in enumerate(processed_images):
                if cancel_check():
                    logger.info("PDF creation cancelled")
                    c.save()
                    if os.path.exists(output_path):
                        os.remove(output_path)
                    self.root.after(0, lambda: self.img_progress.reset())
                    self.root.after(0, lambda: messagebox.showinfo("Cancelled", "Operation was cancelled."))
                    return
                
                try:
                    # Update progress
                    self.root.after(0, lambda idx=idx, total=len(processed_images): 
                        self.img_progress.update(
                            len(processed_images) + (idx / total),
                            f"Adding page {idx + 1} of {total}",
                            f"Processing {os.path.basename(img_data['path'])}"
                        )
                    )
                    
                    # Load image
                    img = Image.open(io.BytesIO(img_data['data']))
                    img_width, img_height = img.size
                    
                    if page_size == "Fit":
                        # Set page size to match image
                        c.setPageSize((img_width, img_height))
                        c.drawImage(ImageReader(img), 0, 0, width=img_width, height=img_height)
                    else:
                        # Fit image to page
                        page_width, page_height = pdf_page_size
                        
                        # Calculate scaling
                        scale = min(page_width/img_width, page_height/img_height, 1.0)
                        
                        # Apply margin (5%)
                        margin = 0.05
                        scale *= (1 - margin)
                        
                        scaled_width = img_width * scale
                        scaled_height = img_height * scale
                        
                        # Center image
                        x = (page_width - scaled_width) / 2
                        y = (page_height - scaled_height) / 2
                        
                        c.drawImage(ImageReader(img), x, y, width=scaled_width, height=scaled_height)
                    
                    c.showPage()
                    img.close()
                    
                    # Periodic cleanup
                    if idx % 10 == 0:
                        gc.collect()
                    
                except Exception as e:
                    logger.error(f"Error adding image to PDF: {e}")
                    logger.debug(f"Image data: {img_data}")
            
            # Save PDF
            if not self._cancel_requested:
                self.root.after(0, lambda: self.img_progress.update(
                    total_steps - 0.5,
                    "Finalizing PDF...",
                    "Saving document..."
                ))
                
                c.save()
                
                # Verify PDF
                if os.path.exists(output_path):
                    file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
                    
                    # Log success
                    logger.info(f"PDF created successfully: {output_path}")
                    logger.info(f"Size: {file_size_mb:.2f} MB, Pages: {len(processed_images)}")
                    
                    # Complete progress
                    self.root.after(0, lambda: self.img_progress.complete("PDF created successfully!"))
                    
                    # Show success message
                    success_msg = (
                        f"✓ PDF created successfully!\n\n"
                        f"📄 File: {os.path.basename(output_path)}\n"
                        f"💾 Size: {file_size_mb:.2f} MB\n"
                        f"📑 Pages: {len(processed_images)}\n"
                        f"📁 Location: {os.path.dirname(output_path)}"
                    )
                    
                    if errors:
                        success_msg += f"\n\n⚠️ Note: {len(errors)} images could not be processed"
                    
                    result = messagebox.askyesno("Success", success_msg + "\n\nOpen the PDF file?")
                    
                    if result:
                        # Open PDF
                        try:
                            if sys.platform.startswith('win'):
                                os.startfile(output_path)
                            elif sys.platform.startswith('darwin'):
                                os.system(f'open "{output_path}"')
                            else:
                                os.system(f'xdg-open "{output_path}"')
                        except:
                            pass
                else:
                    raise Exception("PDF file was not created")
            
        except MemoryError:
            error_msg = (
                "⚠️ Out of memory!\n\n"
                "Try:\n"
                "• Processing fewer images at once\n"
                "• Enabling compression\n"
                "• Reducing quality settings\n"
                "• Closing other applications"
            )
            logger.error("Memory error during PDF creation")
            self.root.after(0, lambda: self.img_progress.error("Memory error"))
            self.root.after(0, lambda: messagebox.showerror("Memory Error", error_msg))
            
        except Exception as e:
            error_msg = f"❌ An error occurred:\n\n{str(e)}"
            logger.error(f"PDF creation error: {e}")
            logger.debug(traceback.format_exc())
            
            self.root.after(0, lambda: self.img_progress.error("Error occurred"))
            self.root.after(0, lambda: messagebox.showerror("Error", error_msg))
            
            # Clean up partial PDF
            if output_path and os.path.exists(output_path):
                try:
                    os.remove(output_path)
                    logger.info(f"Cleaned up partial PDF: {output_path}")
                except:
                    pass
                    
        finally:
            # Force cleanup
            gc.collect()
            self.operation_completed()
            logger.info("PDF creation process completed")


def main():
    """Main entry point"""
    try:
        # Check Python version
        import sys
        if sys.version_info < (3, 7):
            messagebox.showerror(
                "Python Version Error",
                "This application requires Python 3.7 or higher.\n"
                f"You are using Python {sys.version}"
            )
            return
        
        # Check required libraries
        required_modules = {
            'PIL': 'Pillow',
            'reportlab': 'reportlab',
            'psutil': 'psutil'
        }
        
        missing_modules = []
        for module, package in required_modules.items():
            try:
                __import__(module)
            except ImportError:
                missing_modules.append(package)
        
        if missing_modules:
            error_msg = (
                "Missing required libraries:\n\n"
                f"{', '.join(missing_modules)}\n\n"
                "Install with:\n"
                f"pip install {' '.join(missing_modules)}"
            )
            messagebox.showerror("Missing Dependencies", error_msg)
            return
        
        # Create and run application
        root = tk.Tk()
        
        # Set icon if available
        try:
            icon_path = os.path.join(os.path.dirname(__file__), 'icon.ico')
            if os.path.exists(icon_path):
                root.iconbitmap(icon_path)
        except:
            pass
        
        # Set minimum size
        root.minsize(800, 600)
        
        # Center window
        root.update_idletasks()
        x = (root.winfo_screenwidth() // 2) - (950 // 2)
        y = (root.winfo_screenheight() // 2) - (750 // 2)
        root.geometry(f"950x750+{x}+{y}")
        
        # Create application
        app = ImageToPDFTool(root)
        
        # Log startup info
        logger.info("="*50)
        logger.info("Application started successfully")
        logger.info(f"Platform: {sys.platform}")
        logger.info(f"CPU count: {multiprocessing.cpu_count()}")
        logger.info(f"Max workers: {Settings.MAX_WORKERS}")
        logger.info(f"HEIF support: {HEIF_SUPPORT}")
        logger.info("="*50)
        
        # Run
        root.mainloop()
        
    except Exception as e:
        logger.critical(f"Failed to start application: {e}")
        logger.critical(traceback.format_exc())
        
        try:
            messagebox.showerror(
                "Startup Error",
                f"Failed to start application:\n\n{str(e)}\n\n"
                "Check the log file for details."
            )
        except:
            print(f"CRITICAL ERROR: {e}")


if __name__ == "__main__":
    # Configure multiprocessing for Windows
    import sys
    if sys.platform.startswith('win'):
        multiprocessing.freeze_support()
    
    # Run application
    main()      