import tkinter as tk
from tkinter import ttk, filedialog, messagebox, simpledialog
from PIL import Image, ImageTk, ExifTags, ImageOps, ImageEnhance
import os
import threading
import time
from reportlab.lib.pagesizes import letter, A4, legal
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
import subprocess
import tempfile
import json
import sys
from collections import OrderedDict

# Try to import optional libraries
try:
    from PyPDF2 import PdfReader, PdfWriter
    PYPDF2_AVAILABLE = True
except ImportError:
    PYPDF2_AVAILABLE = False

try:
    import fitz
    PYMUPDF_AVAILABLE = True
except ImportError:
    PYMUPDF_AVAILABLE = False

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
logger.info(f"Starting Image to PDF Converter Pro v2.0")
logger.info(f"Python version: {sys.version}")
logger.info(f"PIL version: {Image.__version__}")
logger.info(f"HEIF support: {HEIF_SUPPORT}")
logger.info(f"PyMuPDF available: {PYMUPDF_AVAILABLE}")
logger.info(f"PyPDF2 available: {PYPDF2_AVAILABLE}")
logger.info(f"Color management: {COLOR_MANAGEMENT}")

class Settings:
    """Application settings with production defaults"""
    # Image processing
    DEFAULT_QUALITY = 85
    MAX_IMAGE_SIZE = (1920, 1080)
    THUMBNAIL_SIZE = (200, 200)
    MAX_IMAGE_DIMENSION = 10000
    
    # PDF optimization
    PDF_IMAGE_QUALITY = 85
    PDF_DPI = 150
    JPEG_SUBSAMPLING = 2
    
    # Compression presets
    COMPRESSION_PRESETS = {
        'Maximum': {'quality': 95, 'dpi': 300, 'name': 'Maximum Quality'},
        'High': {'quality': 85, 'dpi': 200, 'name': 'High Quality'},
        'Medium': {'quality': 75, 'dpi': 150, 'name': 'Balanced'},
        'Low': {'quality': 60, 'dpi': 100, 'name': 'Small Size'},
        'Minimum': {'quality': 40, 'dpi': 72, 'name': 'Minimum Size'}
    }
    
    # Resource management
    MAX_CPU_PERCENT = 70
    MAX_MEMORY_MB = 1024
    CRITICAL_CPU_PERCENT = 90
    CRITICAL_MEMORY_MB = 1500
    BATCH_SIZE = 5
    MAX_WORKERS = max(1, min(multiprocessing.cpu_count() // 2, 4))
    PROCESS_DELAY = 0.1
    
    # File handling
    MAX_FILE_SIZE_MB = 100
    SUPPORTED_FORMATS = {
        '.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff', '.tif', 
        '.webp', '.ico', '.heic', '.heif', '.avif'
    }
    
    # PDF settings
    DEFAULT_DPI = 150
    MAX_PDF_SIZE_MB = 500
    
    # UI settings
    PREVIEW_SIZE = (400, 400)
    THEME_OPTIONS = ['clam', 'alt', 'default', 'classic']

class ResourceMonitor:
    """Enhanced resource monitor with alerts"""
    
    def __init__(self):
        self.process = psutil.Process()
        self.alert_shown = False
        
    def get_cpu_usage(self):
        """Get current CPU usage percentage"""
        return psutil.cpu_percent(interval=0.1)
    
    def get_memory_usage(self):
        """Get current memory usage in MB"""
        return self.process.memory_info().rss / 1024 / 1024
    
    def get_memory_percent(self):
        """Get memory usage as percentage"""
        return self.process.memory_percent()
    
    def get_system_memory_available(self):
        """Get available system memory in MB"""
        return psutil.virtual_memory().available / 1024 / 1024
    
    def should_throttle(self):
        """Check if we should throttle processing"""
        cpu_usage = self.get_cpu_usage()
        memory_usage = self.get_memory_usage()
        
        return (cpu_usage > Settings.MAX_CPU_PERCENT or 
                memory_usage > Settings.MAX_MEMORY_MB)
    
    def is_critical(self):
        """Check if resource usage is critical"""
        cpu_usage = self.get_cpu_usage()
        memory_usage = self.get_memory_usage()
        
        return (cpu_usage > Settings.CRITICAL_CPU_PERCENT or 
                memory_usage > Settings.CRITICAL_MEMORY_MB)
    
    def get_resource_status(self):
        """Get detailed resource status"""
        cpu = self.get_cpu_usage()
        mem = self.get_memory_usage()
        mem_percent = self.get_memory_percent()
        available_mem = self.get_system_memory_available()
        
        status = "Normal"
        if self.is_critical():
            status = "Critical"
        elif self.should_throttle():
            status = "High"
            
        return {
            'cpu': cpu,
            'memory_mb': mem,
            'memory_percent': mem_percent,
            'available_mb': available_mem,
            'status': status
        }

class UIStateMixin:
    """Enhanced UI state management"""
    
    def __init__(self):
        self.ui_elements = []
        self.operation_in_progress = False
        self._lock = threading.Lock()
        self.drag_drop_enabled = True
        self._cancel_requested = False
        self.resource_monitor = ResourceMonitor()
        
    def register_ui_element(self, element):
        """Register UI element for state management"""
        if element and element not in self.ui_elements:
            self.ui_elements.append(element)
            
    def set_ui_state(self, enabled):
        """Enable/disable all registered UI elements"""
        state = 'normal' if enabled else 'disabled'
        
        self.drag_drop_enabled = enabled
        
        for element in self.ui_elements:
            try:
                if hasattr(element, 'config'):
                    element.config(state=state)
            except Exception as e:
                logger.debug(f"Could not set state for element: {e}")
                
        # Special handling for widgets
        if hasattr(self, 'image_listbox'):
            if enabled:
                self.image_listbox.state(["!disabled"])
                self.bind_drag_drop_events()
            else:
                self.image_listbox.state(["disabled"])
                self.unbind_drag_drop_events()
                
        if hasattr(self, 'cancel_button'):
            self.cancel_button.config(state='normal' if not enabled else 'disabled')
        
        if hasattr(self, 'pause_button'):
            self.pause_button.config(state='normal' if not enabled else 'disabled')
            
    def bind_drag_drop_events(self):
        """Bind drag and drop events"""
        self.image_listbox.bind('<Button-1>', self.on_image_click)
        self.image_listbox.bind('<B1-Motion>', self.on_image_drag)
        self.image_listbox.bind('<ButtonRelease-1>', self.on_image_drop)
        
    def unbind_drag_drop_events(self):
        """Unbind drag and drop events"""
        self.image_listbox.unbind('<Button-1>')
        self.image_listbox.unbind('<B1-Motion>')
        self.image_listbox.unbind('<ButtonRelease-1>')
                
    def operation_started(self):
        """Call when operation starts"""
        with self._lock:
            self.operation_in_progress = True
            self._cancel_requested = False
            self._pause_requested = False
            
        self.root.after(0, lambda: self.set_ui_state(False))
        self.root.after(0, lambda: self.root.config(cursor="watch"))
            
    def operation_completed(self):
        """Call when operation completes"""
        with self._lock:
            self.operation_in_progress = False
            self._cancel_requested = False
            self._pause_requested = False
                
        self.root.after(0, lambda: self.set_ui_state(True))
        self.root.after(0, lambda: self.root.config(cursor=""))
                
    def check_operation_in_progress(self):
        """Check if any operation is in progress"""
        if self.operation_in_progress:
            messagebox.showwarning("Warning", "Another operation is in progress. Please wait.")
            return True
        return False
    
    def request_cancel(self):
        """Request cancellation"""
        with self._lock:
            self._cancel_requested = True
        logger.info("Cancellation requested by user")
        
    def request_pause(self):
        """Request pause/resume"""
        with self._lock:
            self._pause_requested = not getattr(self, '_pause_requested', False)
        
    def check_resource_alert(self):
        """Check and alert for high resource usage"""
        status = self.resource_monitor.get_resource_status()
        
        if status['status'] == 'Critical' and not self.resource_monitor.alert_shown:
            self.resource_monitor.alert_shown = True
            
            response = messagebox.askyesno(
                "High Resource Usage",
                f"System resources are running high!\n\n"
                f"CPU: {status['cpu']:.1f}%\n"
                f"Memory: {status['memory_mb']:.0f}MB ({status['memory_percent']:.1f}%)\n\n"
                f"Would you like to pause the operation?"
            )
            
            if response:
                self.request_pause()
                
        elif status['status'] == 'Normal':
            self.resource_monitor.alert_shown = False

class EnhancedProgressBar:
    """Enhanced progress bar with pause support"""
    
    def __init__(self, parent, root):
        self.root = root
        self.frame = ttk.Frame(parent)
        
        # Progress bar
        self.progress = ttk.Progressbar(self.frame, mode='determinate')
        
        # Labels
        self.label = tk.Label(self.frame, text="Ready", font=('Helvetica', 10))
        self.detail_label = tk.Label(self.frame, text="", font=('Helvetica', 9))
        self.eta_label = tk.Label(self.frame, text="", font=('Helvetica', 9), fg='blue')
        
        # Pack widgets
        self.label.pack(fill='x', padx=5, pady=(0, 2))
        self.detail_label.pack(fill='x', padx=5)
        self.progress.pack(fill='x', padx=5, pady=5)
        self.eta_label.pack(fill='x', padx=5)
        
        self.start_time = None
        self.pause_time = None
        self.paused_duration = 0
        
    def start(self, total, message="Processing..."):
        self.progress['maximum'] = total
        self.progress['value'] = 0
        self.start_time = time.time()
        self.paused_duration = 0
        self.label.config(text=message, fg='black')
        self.detail_label.config(text="")
        self.eta_label.config(text="")
        self.root.update_idletasks()
        
    def pause(self):
        """Pause progress tracking"""
        self.pause_time = time.time()
        self.label.config(fg='orange')
        self.eta_label.config(text="PAUSED", fg='orange')
        
    def resume(self):
        """Resume progress tracking"""
        if self.pause_time:
            self.paused_duration += time.time() - self.pause_time
            self.pause_time = None
        self.label.config(fg='black')
        self.eta_label.config(fg='blue')
        
    def update(self, current, message="", detail=""):
        try:
            self.progress['value'] = current
            
            if self.start_time and current > 0 and current < self.progress['maximum']:
                elapsed = time.time() - self.start_time - self.paused_duration
                if elapsed > 0:
                    rate = current / elapsed
                    remaining = (self.progress['maximum'] - current) / rate
                    eta = time.strftime('%M:%S', time.gmtime(remaining))
                    speed = rate * 60  # items per minute
                    
                    eta_text = f"ETA: {eta} | Speed: {speed:.1f} items/min"
                    if not self.pause_time:
                        self.eta_label.config(text=eta_text, fg='blue')
            
            if message:
                self.label.config(text=message)
            if detail:
                self.detail_label.config(text=detail)
                
            self.root.update_idletasks()
        except Exception as e:
            logger.error(f"Progress update error: {e}")
            
    def complete(self, message="Complete"):
        self.progress['value'] = self.progress['maximum']
        self.label.config(text=message, fg='green')
        self.detail_label.config(text="")
        
        if self.start_time:
            total_time = time.time() - self.start_time - self.paused_duration
            time_str = time.strftime('%M:%S', time.gmtime(total_time))
            self.eta_label.config(text=f"Completed in {time_str}", fg='green')
        
        self.root.update_idletasks()
        
    def error(self, message="Error occurred"):
        self.label.config(text=message, fg='red')
        self.detail_label.config(text="")
        self.eta_label.config(text="", fg='red')
        self.root.update_idletasks()
        
    def reset(self):
        self.progress['value'] = 0
        self.label.config(text="Ready", fg='black')
        self.detail_label.config(text="")
        self.eta_label.config(text="")
        self.start_time = None
        self.pause_time = None
        self.paused_duration = 0
        
    def pack(self, **kwargs):
        self.frame.pack(**kwargs)
        
    def grid(self, **kwargs):
        self.frame.grid(**kwargs)

class ImagePreviewDialog:
    """Enhanced image preview with metadata"""
    
    def __init__(self, parent, image_path):
        self.dialog = tk.Toplevel(parent)
        self.dialog.title(f"Preview: {os.path.basename(image_path)}")
        self.image_path = image_path
        
        # Make dialog responsive
        self.dialog.grid_rowconfigure(1, weight=1)
        self.dialog.grid_columnconfigure(0, weight=1)
        
        # Set minimum size
        self.dialog.minsize(600, 500)
        
        self.setup_ui()
        self.load_image()
        
        # Center dialog
        self.dialog.wait_visibility()  # Ensure the window is visible before grabbing
        
        self.dialog.grab_set()
        self.dialog.transient(parent)
        

        # Bind resize event
        self.dialog.bind('<Configure>', self.on_resize)

        #parent.wait_window(self.dialog)
        
    def setup_ui(self):
        # Info frame (top)
        info_frame = ttk.LabelFrame(self.dialog, text="Image Information", padding=10)
        info_frame.grid(row=0, column=0, sticky="ew", padx=10, pady=5)
        
        self.info_text = tk.Text(info_frame, height=6, wrap=tk.WORD)
        self.info_text.pack(fill='both', expand=True)
        
        # Image frame (center)
        self.image_frame = ttk.Frame(self.dialog)
        self.image_frame.grid(row=1, column=0, sticky="nsew", padx=10, pady=5)
        
        # Canvas for image
        self.canvas = tk.Canvas(self.image_frame, bg='gray')
        self.canvas.pack(fill='both', expand=True)
        
        # Control frame (bottom)
        control_frame = ttk.Frame(self.dialog)
        control_frame.grid(row=2, column=0, sticky="ew", padx=10, pady=5)
        
        # Zoom controls
        # zoom_frame = ttk.Frame(control_frame)
        # zoom_frame.pack(side='left')
        
        # # ttk.Label(zoom_frame, text="Zoom:").pack(side='left', padx=5)
        # # ttk.Button(zoom_frame, text="-", command=lambda: self.zoom(0.8)).pack(side='left')
        # # self.zoom_label = ttk.Label(zoom_frame, text="100%")
        # # self.zoom_label.pack(side='left', padx=5)
        # # ttk.Button(zoom_frame, text="+", command=lambda: self.zoom(1.2)).pack(side='left')
        # #ttk.Button(zoom_frame, text="Fit", command=self.fit_to_window).pack(side='left', padx=5)
        
        # Close button
        ttk.Button(control_frame, text="Close", command=self.dialog.destroy).pack(side='right', padx=5)
        
        self.current_zoom = 1.0
        
    def load_image(self):
        try:
            # Load image
            self.original_image = Image.open(self.image_path)
            
            # Get image info
            info = self.get_image_info()
            
            # Display info
            self.info_text.delete(1.0, tk.END)
            self.info_text.insert(tk.END, info)
            self.info_text.config(state='disabled')
            
            # Display image
            self.display_image()
            
        except Exception as e:
            messagebox.showerror("Error", f"Could not load image:\n{str(e)}")
            self.dialog.destroy()
    
    def get_image_info(self):
        """Get detailed image information"""
        img = self.original_image
        
        # Basic info
        info = f"File: {os.path.basename(self.image_path)}\n"
        info += f"Format: {img.format or 'Unknown'}\n"
        info += f"Mode: {img.mode}\n"
        info += f"Size: {img.width} x {img.height} pixels\n"
        
        # File size
        file_size = os.path.getsize(self.image_path)
        if file_size < 1024:
            info += f"File size: {file_size} bytes\n"
        elif file_size < 1024 * 1024:
            info += f"File size: {file_size/1024:.1f} KB\n"
        else:
            info += f"File size: {file_size/(1024*1024):.2f} MB\n"
        
        # EXIF data
        if hasattr(img, '_getexif') and img._getexif():
            exif_data = []
            exif = img._getexif()
            
            important_tags = {
                'Make': 271,
                'Model': 272,
                'DateTime': 306,
                'Orientation': 274,
                'XResolution': 282,
                'YResolution': 283
            }
            
            for name, tag in important_tags.items():
                if tag in exif:
                    exif_data.append(f"{name}: {exif[tag]}")
            
            if exif_data:
                info += "\nEXIF Data:\n" + "\n".join(exif_data)
        
        return info
    
    def display_image(self):
        """Display image on canvas"""
        # Calculate display size
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        
        if canvas_width <= 1 or canvas_height <= 1:
            self.dialog.after(100, self.display_image)
            return
        
        if canvas_width > 1 and canvas_height > 1:
            width_ratio = canvas_width / self.original_image.width
            height_ratio = canvas_height / self.original_image.height
            self.current_zoom = min(width_ratio, height_ratio) * 0.9  # 90% to add margin

        # Apply zoom
        display_width = int(self.original_image.width * self.current_zoom)
        display_height = int(self.original_image.height * self.current_zoom)
        
        # Resize image
        resized = self.original_image.resize(
            (display_width, display_height),
            Image.Resampling.LANCZOS
        )
        
        # Convert to PhotoImage
        self.photo = ImageTk.PhotoImage(resized)
        
        # Clear canvas and display
        self.canvas.delete("all")
        self.canvas.create_image(
            canvas_width // 2,
            canvas_height // 2,
            anchor='center',
            image=self.photo
        )
        
        # Update zoom label
        # self.zoom_label.config(text=f"{int(self.current_zoom * 100)}%")
    
    def zoom(self, factor):
        """Zoom in/out"""
        new_zoom = self.current_zoom * factor
        if 0.1 <= new_zoom <= 5.0:  # Limit zoom range
            self.current_zoom = new_zoom
            self.display_image()
    
    def fit_to_window(self):
        """Fit image to window"""
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        
        if canvas_width > 1 and canvas_height > 1:
            width_ratio = canvas_width / self.original_image.width
            height_ratio = canvas_height / self.original_image.height
            self.current_zoom = min(width_ratio, height_ratio) * 0.9  # 90% to add margin
            self.display_image()
    
    def on_resize(self, event):
        """Handle window resize"""
        if hasattr(self, 'original_image'):
            self.display_image()

class OptimizedImageProcessor:
    """Enhanced image processor with compression presets"""
    
    def __init__(self):
        self.max_workers = Settings.MAX_WORKERS
        self._init_color_profiles()
        
    def _init_color_profiles(self):
        """Initialize color profiles"""
        if COLOR_MANAGEMENT:
            try:
                self.srgb_profile = ImageCms.createProfile("sRGB")
            except:
                self.srgb_profile = None
        else:
            self.srgb_profile = None
    
    @staticmethod
    def get_compression_settings(preset_name):
        """Get compression settings for preset"""
        if preset_name in Settings.COMPRESSION_PRESETS:
            return Settings.COMPRESSION_PRESETS[preset_name]
        return Settings.COMPRESSION_PRESETS['Medium']
    
    @staticmethod
    def calculate_optimal_size(img_size, page_size, dpi):
        """Calculate optimal image size for PDF"""
        if not page_size:
            return img_size
            
        img_width, img_height = img_size
        page_width_pts, page_height_pts = page_size
        
        # Convert points to pixels at target DPI
        max_width_px = int(page_width_pts * dpi / 72)
        max_height_px = int(page_height_pts * dpi / 72)
        
        # Apply margin
        margin = 0.95
        max_width_px = int(max_width_px * margin)
        max_height_px = int(max_height_px * margin)
        
        # Calculate scale
        scale = min(max_width_px/img_width, max_height_px/img_height, 1.0)
        
        new_width = int(img_width * scale)
        new_height = int(img_height * scale)
        
        return (new_width, new_height)
    
    @staticmethod
    def handle_exif_orientation(img):
        """Handle EXIF orientation"""
        try:
            if hasattr(img, '_getexif') and img._getexif():
                exif = dict(img._getexif().items())
                
                orientation_tag = 274
                if orientation_tag in exif:
                    orientation = exif[orientation_tag]
                    
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
            logger.debug(f"EXIF orientation error: {e}")
            return img
    
    @staticmethod
    def optimize_image_for_pdf(img, quality, target_size):
        """Optimize image for PDF embedding"""
        # Convert to RGB
        if img.mode in ('RGBA', 'LA', 'P'):
            background = Image.new('RGB', img.size, (255, 255, 255))
            if img.mode == 'P':
                img = img.convert('RGBA')
            if img.mode in ('RGBA', 'LA'):
                background.paste(img, mask=img.split()[-1] if img.mode == 'RGBA' else img.split()[1])
            else:
                background.paste(img)
            img = background
        elif img.mode != 'RGB':
            img = img.convert('RGB')
        
        # Resize if needed
        if target_size and (img.width > target_size[0] or img.height > target_size[1]):
            img.thumbnail(target_size, Image.Resampling.LANCZOS)
            
            # Slight sharpening after resize
            try:
                enhancer = ImageEnhance.Sharpness(img)
                img = enhancer.enhance(1.1)
            except:
                pass
        
        # Save with optimization
        buffer = io.BytesIO()
        
        save_kwargs = {
            'format': 'JPEG',
            'quality': quality,
            'optimize': True,
            'progressive': False,
            'subsampling': Settings.JPEG_SUBSAMPLING
        }
        
        img.save(buffer, **save_kwargs)
        buffer.seek(0)
        
        return buffer
    
    def preprocess_images_batch(self, file_paths, compression_preset, page_size=None,
                              progress_callback=None, cancel_check=None, pause_check=None):
        """Batch process images with compression presets"""
        settings = self.get_compression_settings(compression_preset)
        quality = settings['quality']
        dpi = settings['dpi']
        
        total = len(file_paths)
        processed = []
        errors = []
        
        logger.info(f"Processing {total} images with {compression_preset} preset (Q:{quality}, DPI:{dpi})")
        
        try:
            for batch_start in range(0, total, Settings.BATCH_SIZE):
                # Check for cancellation
                if cancel_check and cancel_check():
                    logger.info("Processing cancelled")
                    break
                
                # Check for pause
                while pause_check and pause_check():
                    time.sleep(0.5)
                    if cancel_check and cancel_check():
                        break
                
                # Resource throttling
                while ResourceMonitor().should_throttle():
                    logger.debug("Throttling due to high resource usage")
                    time.sleep(0.5)
                    if cancel_check and cancel_check():
                        break
                
                batch_end = min(batch_start + Settings.BATCH_SIZE, total)
                batch_files = file_paths[batch_start:batch_end]
                
                with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
                    futures = {}
                    
                    for i, path in enumerate(batch_files, batch_start):
                        future = executor.submit(
                            self._process_single_image_safe,
                            path, quality, dpi, page_size
                        )
                        futures[future] = (i, path)
                    
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
                                    f"Processing {os.path.basename(path)}"
                                )
                        except Exception as e:
                            errors.append((path, str(e)))
                
                # Cleanup
                time.sleep(Settings.PROCESS_DELAY)
                gc.collect()
        
        except Exception as e:
            logger.error(f"Batch processing error: {e}")
            raise
        
        logger.info(f"Processed {len(processed)}/{total} images successfully")
        return processed, errors
    
    @staticmethod
    def _process_single_image_safe(path, quality, dpi, page_size):
        """Safely process single image"""
        try:
            return OptimizedImageProcessor._process_single_image(path, quality, dpi, page_size), None
        except Exception as e:
            logger.error(f"Error processing {path}: {e}")
            return None, str(e)
    
    @staticmethod
    def _process_single_image(path, quality, dpi, page_size):
        """Process single image with optimization"""
        # Validate file
        if not os.path.exists(path):
            raise FileNotFoundError(f"File not found: {path}")
        
        # Check file size
        file_size_mb = os.path.getsize(path) / (1024 * 1024)
        if file_size_mb > Settings.MAX_FILE_SIZE_MB:
            raise ValueError(f"File too large: {file_size_mb:.1f}MB")
        
        with Image.open(path) as img:
            # Verify image
            img.verify()
            img = Image.open(path)
            
            # Handle EXIF orientation
            img = OptimizedImageProcessor.handle_exif_orientation(img)
            
            # Calculate optimal size
            if page_size and page_size != "Fit":
                optimal_size = OptimizedImageProcessor.calculate_optimal_size(
                    img.size, page_size, dpi
                )
            else:
                optimal_size = None
            
            # Optimize for PDF
            buffer = OptimizedImageProcessor.optimize_image_for_pdf(
                img, quality, optimal_size
            )
            
            return {
                'data': buffer.getvalue(),
                'size': img.size if not optimal_size else optimal_size,
                'path': path
            }

class PDFCreator:
    """Enhanced PDF creator with multiple backends"""
    
    @staticmethod
    def create_pdf_reportlab(output_path, processed_images, page_size_option="A4",
                           progress_callback=None, cancel_check=None):
        """Create PDF using ReportLab"""
        # Determine page size
        page_sizes = {
            "A4": A4,
            "Letter": letter,
            "Legal": legal
        }
        
        page_size = page_sizes.get(page_size_option)
        
        # Create canvas
        c = canvas.Canvas(output_path, pagesize=page_size or A4, compress=1)
        c.setPageCompression(1)
        
        for idx, img_data in enumerate(processed_images):
            if cancel_check and cancel_check():
                c.save()
                return False
            
            if progress_callback:
                progress_callback(idx + 1, len(processed_images), f"Creating page {idx + 1}")
            
            # Load image
            img = Image.open(io.BytesIO(img_data['data']))
            
            if page_size_option == "Fit":
                # Fit page to image
                img_width, img_height = img.size
                c.setPageSize((img_width, img_height))
                c.drawImage(ImageReader(img), 0, 0, width=img_width, height=img_height)
            else:
                # Fit image to page
                page_width, page_height = page_size
                img_width, img_height = img.size
                
                # Calculate position
                margin = 0.01
                available_width = page_width * (1 - 2 * margin)
                available_height = page_height * (1 - 2 * margin)
                
                scale = min(available_width/img_width, available_height/img_height, 1.0)
                
                scaled_width = img_width * scale
                scaled_height = img_height * scale
                
                x = (page_width - scaled_width) / 2
                y = (page_height - scaled_height) / 2
                
                c.drawImage(ImageReader(img), x, y, width=scaled_width, height=scaled_height,
                          preserveAspectRatio=True, anchor='c')
            
            c.showPage()
            img.close()
        
        c.save()
        return True
    
    @staticmethod
    def create_pdf_pymupdf(output_path, processed_images, page_size_option="A4",
                         progress_callback=None, cancel_check=None):
        """Create PDF using PyMuPDF for better compression"""
        if not PYMUPDF_AVAILABLE:
            return False
        
        try:
            doc = fitz.open()
            
            # Page sizes in points
            page_sizes = {
                "A4": (595, 842),
                "Letter": (612, 792),
                "Legal": (612, 1008)
            }
            
            for idx, img_data in enumerate(processed_images):
                if cancel_check and cancel_check():
                    doc.close()
                    return False
                
                if progress_callback:
                    progress_callback(idx + 1, len(processed_images), f"Creating page {idx + 1}")
                
                # Create page
                if page_size_option == "Fit":
                    img = Image.open(io.BytesIO(img_data['data']))
                    page = doc.new_page(width=img.width, height=img.height)
                    img.close()
                else:
                    width, height = page_sizes.get(page_size_option, (595, 842))
                    page = doc.new_page(width=width, height=height)
                
                # Insert image
                img_rect = page.rect
                page.insert_image(img_rect, stream=img_data['data'])
            
            # Save with maximum compression
            doc.save(output_path,
                    garbage=4,
                    deflate=True,
                    clean=True,
                    deflate_images=True,
                    deflate_fonts=True)
            doc.close()
            return True
            
        except Exception as e:
            logger.error(f"PyMuPDF error: {e}")
            return False
    
    @staticmethod
    def optimize_pdf_size(input_path, output_path):
        """Further optimize PDF size"""
        optimized = False
        
        # Try PyPDF2
        if PYPDF2_AVAILABLE:
            try:
                reader = PdfReader(input_path)
                writer = PdfWriter()
                
                for page in reader.pages:
                    page.compress_content_streams()
                    writer.add_page(page)
                
                writer.add_metadata(reader.metadata)
                writer.compress_identical_objects(remove_use_as_template=True)
                
                with open(output_path, 'wb') as f:
                    writer.write(f)
                
                optimized = True
            except Exception as e:
                logger.error(f"PyPDF2 optimization error: {e}")
        
        # Try Ghostscript
        if not optimized:
            try:
                gs_cmd = [
                    'gs' if os.name != 'nt' else 'gswin64c',
                    '-sDEVICE=pdfwrite',
                    '-dCompatibilityLevel=1.4',
                    '-dPDFSETTINGS=/ebook',
                    '-dNOPAUSE',
                    '-dQUIET',
                    '-dBATCH',
                    f'-sOutputFile={output_path}',
                    input_path
                ]
                
                subprocess.run(gs_cmd, check=True, capture_output=True)
                optimized = True
            except:
                logger.debug("Ghostscript not available")
        
        return optimized

class ImageToPDFTool(UIStateMixin):
    """Main application with all production features"""
    
    def __init__(self, root):
        super().__init__()
        self.root = root
        self.root.title("Image to PDF Converter Pro v2.0")
        
        # Make window responsive
        self.root.grid_rowconfigure(0, weight=1)
        self.root.grid_columnconfigure(0, weight=1)
        
        # Set minimum size
        self.root.minsize(900, 700)
        
        # Initialize components
        self.image_processor = OptimizedImageProcessor()
        self.pdf_creator = PDFCreator()
        
        # File list and data
        self.image_files = []
        self.current_theme = 'clam'
        
        # Load saved preferences
        self.load_preferences()
        
        # Setup UI
        self.setup_ui()
        self.register_all_ui_elements()
        
        # Apply theme
        self.apply_theme()
        
        # Start resource monitoring
        self.start_resource_monitoring()
        
        # Show capabilities
        self.show_capabilities()
        
        # Bind window events
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        
        logger.info("Application started successfully")
    
    def setup_ui(self):
        """Setup responsive UI"""
        # Main container
        main_container = ttk.Frame(self.root)
        main_container.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)
        
        # Configure grid weights for responsiveness
        main_container.grid_rowconfigure(2, weight=1)  # File list area
        main_container.grid_columnconfigure(0, weight=1)
        
        # Top toolbar
        self.setup_toolbar(main_container)
        
        # Status bar
        self.setup_status_bar(main_container)
        
        # File management area
        self.setup_file_area(main_container)
        
        # Options area
        self.setup_options_area(main_container)
        
        # Progress area
        self.setup_progress_area(main_container)
        
        # Action buttons
        self.setup_action_buttons(main_container)
    
    def setup_toolbar(self, parent):
        """Setup top toolbar"""
        toolbar_frame = ttk.Frame(parent)
        toolbar_frame.grid(row=0, column=0, sticky="ew", pady=(0, 5))
        
        # Title
        title_label = ttk.Label(toolbar_frame, text="Image to PDF Converter Pro",
                               font=('Helvetica', 16, 'bold'))
        title_label.pack(side='left', padx=10)
        
        # Capability indicators
        self.capability_frame = ttk.Frame(toolbar_frame)
        self.capability_frame.pack(side='right', padx=10)
        
        # Theme selector
        theme_frame = ttk.Frame(toolbar_frame)
        theme_frame.pack(side='right', padx=20)
        
        ttk.Label(theme_frame, text="Theme:").pack(side='left', padx=5)
        self.theme_var = tk.StringVar(value=self.current_theme)
        theme_menu = ttk.Combobox(theme_frame, textvariable=self.theme_var,
                                  values=Settings.THEME_OPTIONS, width=10, state='readonly')
        theme_menu.pack(side='left')
        theme_menu.bind('<<ComboboxSelected>>', lambda e: self.apply_theme())
    
    def setup_status_bar(self, parent):
        """Setup status bar with resource monitoring"""
        status_frame = ttk.Frame(parent)
        status_frame.grid(row=1, column=0, sticky="ew", pady=5)
        
        # Resource monitors
        self.cpu_label = ttk.Label(status_frame, text="CPU: 0%", font=('Helvetica', 9))
        self.cpu_label.pack(side='left', padx=10)
        
        self.memory_label = ttk.Label(status_frame, text="Memory: 0 MB", font=('Helvetica', 9))
        self.memory_label.pack(side='left', padx=10)
        
        self.status_label = ttk.Label(status_frame, text="Ready", font=('Helvetica', 9, 'bold'))
        self.status_label.pack(side='left', padx=20)
        
        # File stats
        self.file_stats_label = ttk.Label(status_frame, text="", font=('Helvetica', 9))
        self.file_stats_label.pack(side='right', padx=10)
    
    def setup_file_area(self, parent):
        """Setup file management area"""
        file_container = ttk.Frame(parent)
        file_container.grid(row=2, column=0, sticky="nsew", pady=5)
        
        # Configure grid
        file_container.grid_rowconfigure(0, weight=1)
        file_container.grid_columnconfigure(1, weight=1)
        
        # Left panel - controls
        control_panel = ttk.Frame(file_container)
        control_panel.grid(row=0, column=0, sticky="ns", padx=(0, 5))
        
        # File operations
        file_ops = ttk.LabelFrame(control_panel, text="File Operations", padding=5)
        file_ops.pack(fill='x', pady=(0, 5))
        
        self.add_files_btn = ttk.Button(file_ops, text="Add Images",
                                       command=self.add_images)
        self.add_files_btn.pack(fill='x', pady=2)
        
        self.add_folder_btn = ttk.Button(file_ops, text="Add Folder",
                                        command=self.add_folder)
        self.add_folder_btn.pack(fill='x', pady=2)
        
        self.clear_btn = ttk.Button(file_ops, text="Clear All",
                                   command=self.clear_files)
        self.clear_btn.pack(fill='x', pady=2)
        
        self.remove_btn = ttk.Button(file_ops, text="Remove Selected",
                                    command=self.remove_selected)
        self.remove_btn.pack(fill='x', pady=2)
        
        # Sorting operations
        sort_ops = ttk.LabelFrame(control_panel, text="Sort & Order", padding=5)
        sort_ops.pack(fill='x', pady=5)
        
        self.move_up_btn = ttk.Button(sort_ops, text="⬆ Move Up",
                                     command=self.move_up)
        self.move_up_btn.pack(fill='x', pady=2)
        
        self.move_down_btn = ttk.Button(sort_ops, text="⬇ Move Down",
                                       command=self.move_down)
        self.move_down_btn.pack(fill='x', pady=2)
        
        ttk.Separator(sort_ops, orient='horizontal').pack(fill='x', pady=5)
        
        self.sort_name_btn = ttk.Button(sort_ops, text="Sort by Name",
                                       command=lambda: self.sort_files('name'))
        self.sort_name_btn.pack(fill='x', pady=2)
        
        self.sort_date_btn = ttk.Button(sort_ops, text="Sort by Date",
                                       command=lambda: self.sort_files('date'))
        self.sort_date_btn.pack(fill='x', pady=2)
        
        self.sort_size_btn = ttk.Button(sort_ops, text="Sort by Size",
                                       command=lambda: self.sort_files('size'))
        self.sort_size_btn.pack(fill='x', pady=2)
        
        self.reverse_btn = ttk.Button(sort_ops, text="Reverse Order",
                                     command=self.reverse_order)
        self.reverse_btn.pack(fill='x', pady=2)
        
        # Image operations
        img_ops = ttk.LabelFrame(control_panel, text="Image Operations", padding=5)
        img_ops.pack(fill='x', pady=5)
        
        self.preview_btn = ttk.Button(img_ops, text="Preview Selected",
                                     command=self.preview_selected)
        self.preview_btn.pack(fill='x', pady=2)
        
        # self.rotate_left_btn = ttk.Button(img_ops, text="↺ Rotate Left",
        #                                  command=lambda: self.rotate_selected(-90))
        # self.rotate_left_btn.pack(fill='x', pady=2)
        
        # self.rotate_right_btn = ttk.Button(img_ops, text="↻ Rotate Right",
        #                                   command=lambda: self.rotate_selected(90))
        # self.rotate_right_btn.pack(fill='x', pady=2)
        
        # Right panel - file list
        list_panel = ttk.Frame(file_container)
        list_panel.grid(row=0, column=1, sticky="nsew")
        
        # Configure grid
        list_panel.grid_rowconfigure(1, weight=1)
        list_panel.grid_columnconfigure(0, weight=1)
        
        # List header
        header_frame = ttk.Frame(list_panel)
        header_frame.grid(row=0, column=0, sticky="ew", pady=(0, 5))
        
        ttk.Label(header_frame, text="Images (drag to reorder):",
                 font=('Helvetica', 10, 'bold')).pack(side='left')
        
        self.select_all_btn = ttk.Button(header_frame, text="Select All",
                                        command=self.select_all)
        self.select_all_btn.pack(side='right', padx=5)
        
        # File list with scrollbar
        list_frame = ttk.Frame(list_panel)
        list_frame.grid(row=1, column=0, sticky="nsew")
        
        # Configure grid
        list_frame.grid_rowconfigure(0, weight=1)
        list_frame.grid_columnconfigure(0, weight=1)
        
        # Treeview for better file display
        self.file_tree = ttk.Treeview(list_frame, columns=('size', 'type'),
                                     show='tree headings', selectmode='extended')
        
        # Configure columns
        self.file_tree.heading('#0', text='Filename')
        self.file_tree.heading('size', text='Size')
        self.file_tree.heading('type', text='Type')
        
        self.file_tree.column('#0', width=300)
        self.file_tree.column('size', width=100)
        self.file_tree.column('type', width=100)
        
        # Scrollbars
        v_scroll = ttk.Scrollbar(list_frame, orient='vertical', command=self.file_tree.yview)
        h_scroll = ttk.Scrollbar(list_frame, orient='horizontal', command=self.file_tree.xview)
        
        self.file_tree.configure(yscrollcommand=v_scroll.set, xscrollcommand=h_scroll.set)
        
        # Grid widgets
        self.file_tree.grid(row=0, column=0, sticky="nsew")
        v_scroll.grid(row=0, column=1, sticky="ns")
        h_scroll.grid(row=1, column=0, sticky="ew")
        
        # Bind events
        self.file_tree.bind('<Double-Button-1>', lambda e: self.preview_selected())
        self.file_tree.bind('<Delete>', lambda e: self.remove_selected())
        self.file_tree.bind('<Control-a>', lambda e: self.select_all())
        
        # Setup drag and drop
        self.setup_drag_drop()
        
        # For backward compatibility
        self.image_listbox = self.file_tree
    
    def setup_options_area(self, parent):
        """Setup conversion options"""
        options_container = ttk.LabelFrame(parent, text="Conversion Options", padding=10)
        options_container.grid(row=3, column=0, sticky="ew", pady=5)
        
        # Configure grid
        options_container.grid_columnconfigure(0, weight=1)
        options_container.grid_columnconfigure(1, weight=1)
        
        # Left side - compression
        left_frame = ttk.Frame(options_container)
        left_frame.grid(row=0, column=0, sticky="ew", padx=(0, 10))
        
        # Compression preset
        preset_frame = ttk.Frame(left_frame)
        preset_frame.pack(fill='x', pady=5)
        
        ttk.Label(preset_frame, text="PDF Quality:").pack(side='left')
        
        self.compression_var = tk.StringVar(value="Medium")
        compression_menu = ttk.Combobox(preset_frame, textvariable=self.compression_var,
                                       width=20, state='readonly')
        compression_menu['values'] = list(Settings.COMPRESSION_PRESETS.keys())
        compression_menu.pack(side='left', padx=10)
        compression_menu.bind('<<ComboboxSelected>>', self.on_compression_change)
        
        # Compression details
        self.compression_info = ttk.Label(left_frame, text="", font=('Helvetica', 9))
        self.compression_info.pack(pady=5)
        
        # Custom quality (disabled by default)
        custom_frame = ttk.Frame(left_frame)
        custom_frame.pack(fill='x', pady=5)
        
        self.custom_quality_var = tk.BooleanVar(value=False)
        self.custom_check = ttk.Checkbutton(custom_frame, text="Custom quality:",
                                           variable=self.custom_quality_var,
                                           command=self.toggle_custom_quality)
        self.custom_check.pack(side='left')
        
        self.quality_var = tk.IntVar(value=85)
        self.quality_scale = ttk.Scale(custom_frame, from_=10, to=100,
                                      variable=self.quality_var,
                                      orient='horizontal', length=150)
        self.quality_scale.pack(side='left', padx=10)
        self.quality_scale.config(state='disabled')
        
        self.quality_label = ttk.Label(custom_frame, text="85%")
        self.quality_label.pack(side='left')
        
        self.quality_scale.config(command=lambda v: self.quality_label.config(text=f"{int(float(v))}%"))
        
        # Right side - page settings
        right_frame = ttk.Frame(options_container)
        right_frame.grid(row=0, column=1, sticky="ew")
        
        # Page size
        page_frame = ttk.Frame(right_frame)
        page_frame.pack(fill='x', pady=5)
        
        ttk.Label(page_frame, text="Page Size:").pack(side='left')
        
        self.page_size_var = tk.StringVar(value="A4")
        sizes = ["A4", "Letter", "Legal", "Fit to Image"]
        
        for size in sizes:
            ttk.Radiobutton(page_frame, text=size, variable=self.page_size_var,
                           value=size if size != "Fit to Image" else "Fit").pack(side='left', padx=5)
        
        # Advanced options
        advanced_frame = ttk.Frame(right_frame)
        advanced_frame.pack(fill='x', pady=10)
        
        self.use_pymupdf_var = tk.BooleanVar(value=PYMUPDF_AVAILABLE)
        if PYMUPDF_AVAILABLE:
            ttk.Checkbutton(advanced_frame, text="Enhanced compression",
                           variable=self.use_pymupdf_var).pack(side='left', padx=5)
        
        self.optimize_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(advanced_frame, text="Optimize PDF size",
                       variable=self.optimize_var).pack(side='left', padx=5)
        
        self.auto_rotate_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(advanced_frame, text="Auto-rotate (EXIF)",
                       variable=self.auto_rotate_var).pack(side='left', padx=5)
        
        # Update compression info
        self.on_compression_change()
    
    def setup_progress_area(self, parent):
        """Setup progress area"""
        progress_container = ttk.Frame(parent)
        progress_container.grid(row=4, column=0, sticky="ew", pady=5)
        
        self.progress_bar = EnhancedProgressBar(progress_container, self.root)
        self.progress_bar.pack(fill='x')
    
    def setup_action_buttons(self, parent):
        """Setup action buttons"""
        button_container = ttk.Frame(parent)
        button_container.grid(row=5, column=0, pady=10)
        
        self.convert_btn = ttk.Button(button_container, text="Convert to PDF",
                                     command=self.start_conversion,
                                     style='Accent.TButton')
        self.convert_btn.pack(side='left', padx=5)
        
        self.pause_button = ttk.Button(button_container, text="Pause",
                                      command=self.toggle_pause,
                                      state='disabled')
        self.pause_button.pack(side='left', padx=5)
        
        self.cancel_button = ttk.Button(button_container, text="Cancel",
                                       command=self.request_cancel,
                                       state='disabled')
        self.cancel_button.pack(side='left', padx=5)
        
        ttk.Separator(button_container, orient='vertical').pack(side='left', fill='y', padx=10)
        
        # ttk.Button(button_container, text="Settings",
        #           command=self.show_settings).pack(side='left', padx=5)
        
        ttk.Button(button_container, text="View Log",
                  command=self.view_log).pack(side='left', padx=5)
        
        ttk.Button(button_container, text="Help",
                  command=self.show_help).pack(side='left', padx=5)
    
    def apply_theme(self):
        """Apply selected theme"""
        try:
            style = ttk.Style()
            theme = self.theme_var.get()
            style.theme_use(theme)
            
            # Custom style configurations
            style.configure('Accent.TButton', font=('Helvetica', 10, 'bold'))
            
            # Apply colors based on theme
            if theme == 'clam':
                self.root.configure(bg='#f0f0f0')
            
            self.current_theme = theme
            self.save_preferences()
            
        except Exception as e:
            logger.error(f"Error applying theme: {e}")
    
    def show_capabilities(self):
        """Show available features in toolbar"""
        features = []
        
        if PYMUPDF_AVAILABLE:
            label = ttk.Label(self.capability_frame, text="✓ Enhanced Compression",
                            font=('Helvetica', 9), foreground='green')
            label.pack(side='left', padx=5)
        
        if PYPDF2_AVAILABLE:
            label = ttk.Label(self.capability_frame, text="✓ PDF Optimization",
                            font=('Helvetica', 9), foreground='green')
            label.pack(side='left', padx=5)
        
        if HEIF_SUPPORT:
            label = ttk.Label(self.capability_frame, text="✓ HEIF/HEIC Support",
                            font=('Helvetica', 9), foreground='green')
            label.pack(side='left', padx=5)
    
    def start_resource_monitoring(self):
        """Start monitoring system resources"""
        self.update_resource_display()
    
    def update_resource_display(self):
        """Update resource display"""
        try:
            status = self.resource_monitor.get_resource_status()
            
            # Update labels with color coding
            cpu_color = 'black'
            if status['cpu'] > Settings.MAX_CPU_PERCENT:
                cpu_color = 'orange'
            if status['cpu'] > Settings.CRITICAL_CPU_PERCENT:
                cpu_color = 'red'
            
            mem_color = 'black'
            if status['memory_mb'] > Settings.MAX_MEMORY_MB:
                mem_color = 'orange'
            if status['memory_mb'] > Settings.CRITICAL_MEMORY_MB:
                mem_color = 'red'
            
            self.cpu_label.config(
                text=f"CPU: {status['cpu']:.1f}%",
                foreground=cpu_color
            )
            
            self.memory_label.config(
                text=f"Memory: {status['memory_mb']:.0f} MB ({status['memory_percent']:.1f}%)",
                foreground=mem_color
            )
            
            # Update status
            if status['status'] == 'Critical':
                self.status_label.config(text="⚠️ High Resource Usage", foreground='red')
            elif status['status'] == 'High':
                self.status_label.config(text="⚠️ Moderate Load", foreground='orange')
            elif self.operation_in_progress:
                self.status_label.config(text="🔄 Processing...", foreground='blue')
            else:
                self.status_label.config(text="✓ Ready", foreground='green')
            
            # Check for alerts
            if self.operation_in_progress:
                self.check_resource_alert()
            
        except Exception as e:
            logger.debug(f"Resource monitoring error: {e}")
        
        # Schedule next update
        self.root.after(1000, self.update_resource_display)
    
    def register_all_ui_elements(self):
        """Register UI elements for state management"""
        # elements = [
        #     self.add_files_btn, self.add_folder_btn, self.clear_btn, self.remove_btn,
        #     self.move_up_btn, self.move_down_btn, self.sort_name_btn, self.sort_date_btn,
        #     self.sort_size_btn, self.reverse_btn, self.preview_btn, self.rotate_left_btn,
        #     self.rotate_right_btn, self.select_all_btn, self.convert_btn
        # ]
        #removing rotate buttons until functionality is added
        elements = [
            self.add_files_btn, self.add_folder_btn, self.clear_btn, self.remove_btn,
            self.move_up_btn, self.move_down_btn, self.sort_name_btn, self.sort_date_btn,
            self.sort_size_btn, self.reverse_btn, self.preview_btn, self.select_all_btn, self.convert_btn
        ]
        
        for element in elements:
            self.register_ui_element(element)
    
    def setup_drag_drop(self):
        """Setup drag and drop for file reordering"""
        self.drag_start_index = None
        self.file_tree.bind('<Button-1>', self.on_drag_start)
        self.file_tree.bind('<B1-Motion>', self.on_drag_motion)
        self.file_tree.bind('<ButtonRelease-1>', self.on_drag_release)
    
    def on_drag_start(self, event):
        """Start drag operation"""
        if not self.drag_drop_enabled or self.operation_in_progress:
            return
        
        item = self.file_tree.identify_row(event.y)
        if item:
            self.drag_start_index = self.file_tree.index(item)
    
    def on_drag_motion(self, event):
        """Handle drag motion"""
        if self.drag_start_index is None:
            return
        
        self.file_tree.config(cursor="hand2")
    
    def on_drag_release(self, event):
        """Handle drag release"""
        if self.drag_start_index is None:
            return
        
        self.file_tree.config(cursor="")
        
        item = self.file_tree.identify_row(event.y)
        if item:
            drop_index = self.file_tree.index(item)
            
            if drop_index != self.drag_start_index:
                # Reorder files
                file = self.image_files.pop(self.drag_start_index)
                self.image_files.insert(drop_index, file)
                
                # Refresh display
                self.refresh_file_list()
        
        self.drag_start_index = None
    
    # Dummy implementations for backward compatibility
    def on_image_click(self, event):
        self.on_drag_start(event)
    
    def on_image_drag(self, event):
        self.on_drag_motion(event)
    
    def on_image_drop(self, event):
        self.on_drag_release(event)
    
    def add_images(self):
        """Add image files"""
        if self.operation_in_progress:
            return
        
        files = filedialog.askopenfilenames(
            title="Select images",
            filetypes=[
                ("All Images", "*.jpg *.jpeg *.png *.bmp *.gif *.tiff *.webp *.heic *.heif"),
                ("JPEG", "*.jpg *.jpeg"),
                ("PNG", "*.png"),
                ("All files", "*.*")
            ]
        )
        
        self.add_files_to_list(files)
    
    def add_folder(self):
        """Add all images from folder"""
        if self.operation_in_progress:
            return
        
        folder = filedialog.askdirectory(title="Select folder containing images")
        if folder:
            # Find all images
            files = []
            for ext in Settings.SUPPORTED_FORMATS:
                files.extend(Path(folder).glob(f"*{ext}"))
                files.extend(Path(folder).glob(f"*{ext.upper()}"))
            
            # Convert to strings and add
            file_paths = [str(f) for f in files]
            self.add_files_to_list(file_paths)
    
    def add_files_to_list(self, files):
        """Add files to the list with validation"""
        added = 0
        errors = []
        
        for file in files:
            # if file in self.image_files:
            #     continue
            
            try:
                # Validate file
                if not os.path.exists(file):
                    errors.append((file, "File not found"))
                    continue
                
                # Check size
                size_mb = os.path.getsize(file) / (1024 * 1024)
                if size_mb > Settings.MAX_FILE_SIZE_MB:
                    errors.append((file, f"File too large ({size_mb:.1f} MB)"))
                    continue
                
                # Add to list
                self.image_files.append(file)
                added += 1
                
            except Exception as e:
                errors.append((file, str(e)))
        
        # Refresh display
        self.refresh_file_list()
        
        # Show results
        if errors:
            error_msg = "Some files could not be added:\n\n"
            for file, error in errors[:5]:
                error_msg += f"• {os.path.basename(file)}: {error}\n"
            if len(errors) > 5:
                error_msg += f"\n... and {len(errors) - 5} more"
            messagebox.showwarning("Warning", error_msg)
        
        if added > 0:
            logger.info(f"Added {added} images")
    
    def refresh_file_list(self):
        """Refresh the file list display"""
        # Clear tree
        for item in self.file_tree.get_children():
            self.file_tree.delete(item)
        
        # Add files
        total_size = 0
        for file in self.image_files:
            try:
                size = os.path.getsize(file)
                total_size += size
                
                # Format size
                if size < 1024:
                    size_str = f"{size} B"
                elif size < 1024 * 1024:
                    size_str = f"{size/1024:.1f} KB"
                else:
                    size_str = f"{size/(1024*1024):.1f} MB"
                
                # Get type
                ext = os.path.splitext(file)[1].upper()
                if ext.startswith('.'):
                    ext = ext[1:]
                
                # Insert item
                self.file_tree.insert('', 'end', text=os.path.basename(file),
                                     values=(size_str, ext))
                
            except Exception as e:
                logger.error(f"Error adding file to list: {e}")
        
        # Update stats
        self.update_file_stats(total_size)
    
    def update_file_stats(self, total_size=None):
        """Update file statistics"""
        count = len(self.image_files)
        
        if count == 0:
            self.file_stats_label.config(text="")
            return
        
        if total_size is None:
            total_size = sum(os.path.getsize(f) for f in self.image_files if os.path.exists(f))
        
        total_mb = total_size / (1024 * 1024)
        
        # Estimate output size
        preset = self.compression_var.get()
        settings = Settings.COMPRESSION_PRESETS.get(preset, Settings.COMPRESSION_PRESETS['Medium'])
        quality = settings['quality']
        
        # Estimation based on quality
        compression_ratios = {
            95: 0.4,
            85: 0.25,
            75: 0.15,
            60: 0.1,
            40: 0.06
        }
        
        ratio = compression_ratios.get(quality, 0.2)
        estimated_mb = total_mb * ratio
        
        stats_text = f"Files: {count} | Input: {total_mb:.1f} MB | Est. PDF: ~{estimated_mb:.1f} MB"
        self.file_stats_label.config(text=stats_text)
    
    def clear_files(self):
        """Clear all files"""
        if self.operation_in_progress:
            return
        
        if self.image_files and messagebox.askyesno("Confirm", "Clear all files?"):
            self.image_files.clear()
            self.refresh_file_list()
            self.progress_bar.reset()
    
    def remove_selected(self):
        """Remove selected files"""
        if self.operation_in_progress:
            return
        
        selected = self.file_tree.selection()
        if not selected:
            return
        
        # Get indices
        indices = [self.file_tree.index(item) for item in selected]
        
        # Remove in reverse order
        for idx in sorted(indices, reverse=True):
            del self.image_files[idx]
        
        self.refresh_file_list()
    
    def move_up(self):
        """Move selected file up"""
        if self.operation_in_progress:
            return
        
        selected = self.file_tree.selection()
        if not selected:
            return
        
        idx = self.file_tree.index(selected[0])
        if idx > 0:
            self.image_files[idx], self.image_files[idx-1] = self.image_files[idx-1], self.image_files[idx]
            self.refresh_file_list()
            
            # Restore selection
            new_item = self.file_tree.get_children()[idx-1]
            self.file_tree.selection_set(new_item)
            self.file_tree.see(new_item)
    
    def move_down(self):
        """Move selected file down"""
        if self.operation_in_progress:
            return
        
        selected = self.file_tree.selection()
        if not selected:
            return
        
        idx = self.file_tree.index(selected[0])
        if idx < len(self.image_files) - 1:
            self.image_files[idx], self.image_files[idx+1] = self.image_files[idx+1], self.image_files[idx]
            self.refresh_file_list()
            
            # Restore selection
            new_item = self.file_tree.get_children()[idx+1]
            self.file_tree.selection_set(new_item)
            self.file_tree.see(new_item)
    
    def sort_files(self, sort_type):
        """Sort files by specified type"""
        if self.operation_in_progress or not self.image_files:
            return
        
        if sort_type == 'name':
            self.image_files.sort(key=lambda x: os.path.basename(x).lower())
        elif sort_type == 'date':
            self.image_files.sort(key=lambda x: os.path.getmtime(x))
        elif sort_type == 'size':
            self.image_files.sort(key=lambda x: os.path.getsize(x))
        
        self.refresh_file_list()
        logger.info(f"Sorted files by {sort_type}")
    
    def reverse_order(self):
        """Reverse file order"""
        if self.operation_in_progress or not self.image_files:
            return
        
        self.image_files.reverse()
        self.refresh_file_list()
    
    def select_all(self):
        """Select all files"""
        for item in self.file_tree.get_children():
            self.file_tree.selection_add(item)
    
    def preview_selected(self):
        """Preview selected image"""
        selected = self.file_tree.selection()
        if not selected:
            messagebox.showinfo("Info", "Please select an image to preview")
            return
        
        idx = self.file_tree.index(selected[0])
        image_path = self.image_files[idx]
        
        # Show preview dialog
        ImagePreviewDialog(self.root, image_path)
    
    def rotate_selected(self, angle):
        """Rotate selected images (temporarily for preview)"""
        selected = self.file_tree.selection()
        if not selected:
            messagebox.showinfo("Info", "Please select images to rotate")
            return
        
        # Note: This is a placeholder - actual rotation would be applied during processing
        count = len(selected)
        messagebox.showinfo("Info", f"Rotation of {angle}° will be applied to {count} image(s) during conversion")
    
    def on_compression_change(self, event=None):
        """Handle compression preset change"""
        preset = self.compression_var.get()
        settings = Settings.COMPRESSION_PRESETS.get(preset, Settings.COMPRESSION_PRESETS['Medium'])
        
        info_text = f"{settings['name']} - Quality: {settings['quality']}%, DPI: {settings['dpi']}"
        self.compression_info.config(text=info_text)
        
        # Update file stats with new estimation
        self.update_file_stats()
        
        # Disable custom quality when preset is selected
        if not self.custom_quality_var.get():
            self.quality_var.set(settings['quality'])
            self.quality_label.config(text=f"{settings['quality']}%")
    
    def toggle_custom_quality(self):
        """Toggle custom quality controls"""
        if self.custom_quality_var.get():
            self.quality_scale.config(state='normal')
            self.compression_var.set("Custom")
            self.compression_info.config(text="Custom compression settings")
        else:
            self.quality_scale.config(state='disabled')
            self.compression_var.set("Medium")
            self.on_compression_change()
    
    def toggle_pause(self):
        """Toggle pause state"""
        self.request_pause()
        
        if hasattr(self, '_pause_requested') and self._pause_requested:
            self.pause_button.config(text="Resume")
            self.progress_bar.pause()
        else:
            self.pause_button.config(text="Pause")
            self.progress_bar.resume()
    
    def start_conversion(self):
        """Start the conversion process"""
        if self.check_operation_in_progress():
            return
        
        if not self.image_files:
            messagebox.showwarning("Warning", "Please add at least one image")
            return
        
        # Validate files exist
        missing = [f for f in self.image_files if not os.path.exists(f)]
        if missing:
            messagebox.showerror(
                "Error",
                f"Some files are missing:\n{chr(10).join(missing[:5])}"
                + (f"\n... and {len(missing)-5} more" if len(missing) > 5 else "")
            )
            # Remove missing files
            self.image_files = [f for f in self.image_files if os.path.exists(f)]
            self.refresh_file_list()
            if not self.image_files:
                return
        
        # Start conversion thread
        thread = threading.Thread(target=self.conversion_worker, daemon=True)
        thread.start()
    
    def conversion_worker(self):
        """Main conversion worker"""
        self.operation_started()
        output_path = None
        temp_path = None
        
        try:
            # Get output path
            output_path = filedialog.asksaveasfilename(
                defaultextension=".pdf",
                filetypes=[("PDF files", "*.pdf"), ("All files", "*.*")],
                initialfile=f"converted_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
            )
            
            if not output_path:
                return
            
            logger.info(f"Starting conversion to: {output_path}")
            
            # Get settings
            if self.custom_quality_var.get():
                preset = "Custom"
                quality = self.quality_var.get()
                dpi = Settings.PDF_DPI
            else:
                preset = self.compression_var.get()
                settings = Settings.COMPRESSION_PRESETS.get(preset, Settings.COMPRESSION_PRESETS['Medium'])
                quality = settings['quality']
                dpi = settings['dpi']
            
            page_size_option = self.page_size_var.get()
            use_pymupdf = self.use_pymupdf_var.get() if PYMUPDF_AVAILABLE else False
            optimize = self.optimize_var.get()
            
            # Progress callbacks
            def progress_callback(current, total, message=""):
                self.root.after(0, lambda: self.progress_bar.update(current, message))
            
            def cancel_check():
                return self._cancel_requested
            
            def pause_check():
                return getattr(self, '_pause_requested', False)
            
            # Start progress
            total_steps = len(self.image_files) + 2
            self.root.after(0, lambda: self.progress_bar.start(total_steps, "Processing images..."))
            
            # Process images
            processed_images, errors = self.image_processor.preprocess_images_batch(
                self.image_files,
                preset if preset != "Custom" else "Medium",
                page_size=A4 if page_size_option == "A4" else letter if page_size_option == "Letter" else None,
                progress_callback=progress_callback,
                cancel_check=cancel_check,
                pause_check=pause_check
            )
            
            if self._cancel_requested:
                self.root.after(0, lambda: self.progress_bar.reset())
                logger.info("Conversion cancelled by user")
                return
            
            # Show errors if any
            if errors:
                error_count = len(errors)
                logger.warning(f"{error_count} images failed to process")
                
                if not processed_images:
                    raise Exception("No images could be processed successfully")
                
                # Show warning
                self.root.after(0, lambda: messagebox.showwarning(
                    "Processing Errors",
                    f"{error_count} image(s) could not be processed.\n"
                    f"Continuing with {len(processed_images)} images."
                ))
            
            # Create PDF
            self.root.after(0, lambda: self.progress_bar.update(
                len(self.image_files) + 1,
                "Creating PDF...",
                "Generating document..."
            ))
            
            # Use appropriate creator
            if use_pymupdf:
                logger.info("Using PyMuPDF for enhanced compression")
                success = PDFCreator.create_pdf_pymupdf(
                    output_path, processed_images, page_size_option,
                    lambda c, t, m: progress_callback(len(self.image_files) + 1, total_steps, m),
                    cancel_check
                )
            else:
                logger.info("Using ReportLab for PDF creation")
                success = PDFCreator.create_pdf_reportlab(
                    output_path, processed_images, page_size_option,
                    lambda c, t, m: progress_callback(len(self.image_files) + 1, total_steps, m),
                    cancel_check
                )
            
            if self._cancel_requested:
                if os.path.exists(output_path):
                    os.remove(output_path)
                return
            
            # Optimize if requested
            if optimize and os.path.exists(output_path):
                self.root.after(0, lambda: self.progress_bar.update(
                    total_steps - 0.5,
                    "Optimizing PDF...",
                    "Reducing file size..."
                ))
                
                temp_path = output_path + ".tmp"
                if PDFCreator.optimize_pdf_size(output_path, temp_path):
                    # Check if optimization reduced size
                    original_size = os.path.getsize(output_path)
                    optimized_size = os.path.getsize(temp_path)
                    
                    if optimized_size < original_size * 0.95:  # At least 5% reduction
                        os.replace(temp_path, output_path)
                        logger.info(f"PDF optimized: {original_size} -> {optimized_size} bytes")
                    else:
                        if os.path.exists(temp_path):
                            os.remove(temp_path)
            
            # Verify result
            if os.path.exists(output_path):
                file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
                input_size_mb = sum(os.path.getsize(f) for f in self.image_files if os.path.exists(f)) / (1024 * 1024)
                compression_ratio = (1 - file_size_mb/input_size_mb) * 100 if input_size_mb > 0 else 0
                
                logger.info(f"PDF created successfully: {output_path}")
                logger.info(f"Compression: {input_size_mb:.1f}MB -> {file_size_mb:.1f}MB ({compression_ratio:.1f}%)")
                
                self.root.after(0, lambda: self.progress_bar.complete("PDF created successfully!"))
                
                # Success dialog
                success_msg = (
                    f"✅ PDF created successfully!\n\n"
                    f"📄 File: {os.path.basename(output_path)}\n"
                    f"💾 Size: {file_size_mb:.2f} MB\n"
                    f"📊 Compression: {compression_ratio:.1f}%\n"
                    f"📑 Pages: {len(processed_images)}"
                )
                
                result = messagebox.askyesno("Success", success_msg + "\n\nOpen the PDF file?")
                
                if result:
                    try:
                        if sys.platform.startswith('win'):
                            os.startfile(output_path)
                        elif sys.platform.startswith('darwin'):
                            subprocess.run(['open', output_path])
                        else:
                            subprocess.run(['xdg-open', output_path])
                    except:
                        pass
            
        except MemoryError:
            logger.error("Out of memory during conversion")
            self.root.after(0, lambda: self.progress_bar.error("Memory error"))
            self.root.after(0, lambda: messagebox.showerror(
                "Memory Error",
                "Out of memory! Try:\n"
                "• Processing fewer images\n"
                "• Using lower quality settings\n"
                "• Closing other applications"
            ))
            
        except Exception as e:
            logger.error(f"Conversion error: {e}")
            logger.debug(traceback.format_exc())
            self.root.after(0, lambda: self.progress_bar.error("Conversion failed"))
            self.root.after(0, lambda: messagebox.showerror("Error", f"Conversion failed:\n{str(e)}"))
            
            # Cleanup
            if output_path and os.path.exists(output_path):
                try:
                    os.remove(output_path)
                except:
                    pass
                    
        finally:
            if temp_path and os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except:
                    pass
            
            gc.collect()
            self.operation_completed()
    
    def show_settings(self):
        """Show settings dialog"""
        messagebox.showinfo("Settings", "Settings dialog not implemented in this version")
    
    def view_log(self):
        """View log file"""
        try:
            if sys.platform.startswith('win'):
                os.startfile(log_filename)
            elif sys.platform.startswith('darwin'):
                subprocess.run(['open', log_filename])
            else:
                subprocess.run(['xdg-open', log_filename])
        except Exception as e:
            messagebox.showerror("Error", f"Could not open log file:\n{str(e)}")
    
    def show_help(self):
        """Show help dialog"""
        help_text = (
            "Image to PDF Converter Pro v2.0\n\n"
            "Features:\n"
            "• Convert multiple image formats to PDF\n"
            "• Support for JPEG, PNG, BMP, GIF, TIFF, WebP\n"
            "• HEIF/HEIC support (if available)\n"
            "• Compression presets for optimal file size\n"
            "• Drag & drop to reorder images\n"
            "• Multiple page size options\n"
            "• Resource monitoring and throttling\n"
            "• Batch processing with pause/resume\n\n"
            "Tips:\n"
            "• Use compression presets for best results\n"
            "• Preview images before conversion\n"
            "• Monitor resource usage during conversion\n"
            "• Sort images before converting\n\n"
            "Keyboard Shortcuts:\n"
            "• Ctrl+A: Select all\n"
            "• Delete: Remove selected\n"
            "• Double-click: Preview image"
        )
        
        messagebox.showinfo("Help", help_text)
    
    def save_preferences(self):
        """Save user preferences"""
        try:
            prefs = {
                'theme': self.current_theme,
                'compression': self.compression_var.get(),
                'page_size': self.page_size_var.get(),
                'optimize': self.optimize_var.get(),
                'auto_rotate': self.auto_rotate_var.get(),
                'use_pymupdf': self.use_pymupdf_var.get()
            }
            
            prefs_file = Path.home() / '.image_to_pdf_prefs.json'
            with open(prefs_file, 'w') as f:
                json.dump(prefs, f)
                
        except Exception as e:
            logger.error(f"Error saving preferences: {e}")
    
    def load_preferences(self):
        """Load user preferences"""
        try:
            prefs_file = Path.home() / '.image_to_pdf_prefs.json'
            if prefs_file.exists():
                with open(prefs_file, 'r') as f:
                    prefs = json.load(f)
                
                self.current_theme = prefs.get('theme', 'clam')
                
                # Apply loaded preferences after UI is created
                self.root.after(100, lambda: self._apply_loaded_preferences(prefs))
                
        except Exception as e:
            logger.error(f"Error loading preferences: {e}")
    
    def _apply_loaded_preferences(self, prefs):
        """Apply loaded preferences to UI"""
        try:
            if hasattr(self, 'compression_var'):
                self.compression_var.set(prefs.get('compression', 'Medium'))
            if hasattr(self, 'page_size_var'):
                self.page_size_var.set(prefs.get('page_size', 'A4'))
            if hasattr(self, 'optimize_var'):
                self.optimize_var.set(prefs.get('optimize', True))
            if hasattr(self, 'auto_rotate_var'):
                self.auto_rotate_var.set(prefs.get('auto_rotate', True))
            if hasattr(self, 'use_pymupdf_var') and PYMUPDF_AVAILABLE:
                self.use_pymupdf_var.set(prefs.get('use_pymupdf', True))
        except Exception as e:
            logger.error(f"Error applying preferences: {e}")
    
    def on_closing(self):
        """Handle window closing"""
        if self.operation_in_progress:
            if messagebox.askokcancel("Confirm", "Operation in progress. Do you want to quit?"):
                self.request_cancel()
                self.save_preferences()
                self.root.destroy()
        else:
            self.save_preferences()
            self.root.destroy()


def main():
    """Main application entry point"""
    try:
        # Check Python version
        if sys.version_info < (3, 7):
            messagebox.showerror(
                "Python Version Error",
                f"This application requires Python 3.7 or higher.\n"
                f"You are using Python {sys.version}"
            )
            return
        
        # Check required modules
        required = {
            'PIL': 'Pillow',
            'reportlab': 'reportlab',
            'psutil': 'psutil'
        }
        
        missing = []
        for module, package in required.items():
            try:
                __import__(module)
            except ImportError:
                missing.append(package)
        
        if missing:
            messagebox.showerror(
                "Missing Dependencies",
                f"Please install required packages:\n\n"
                f"pip install {' '.join(missing)}"
            )
            return
        
        # Create application
        root = tk.Tk()
        
        # Set icon if available
        try:
            icon_path = Path(__file__).parent / 'icon.ico'
            if icon_path.exists():
                root.iconbitmap(str(icon_path))
        except:
            pass
        
        # Create app instance
        app = ImageToPDFTool(root)
        
        # Center window
        root.update_idletasks()
        width = root.winfo_width()
        height = root.winfo_height()
        x = (root.winfo_screenwidth() // 2) - (width // 2)
        y = (root.winfo_screenheight() // 2) - (height // 2)
        root.geometry(f'{width}x{height}+{x}+{y}')
        
        # Log startup
        logger.info("="*60)
        logger.info("Image to PDF Converter Pro v2.0 Started")
        logger.info(f"Platform: {sys.platform}")
        logger.info(f"Python: {sys.version}")
        logger.info(f"CPU cores: {multiprocessing.cpu_count()}")
        logger.info(f"Available memory: {psutil.virtual_memory().available / (1024**3):.1f} GB")
        logger.info("="*60)
        
        # Run application
        root.mainloop()
        
    except Exception as e:
        logger.critical(f"Failed to start application: {e}")
        logger.critical(traceback.format_exc())
        
        try:
            messagebox.showerror(
                "Startup Error",
                f"Failed to start application:\n\n{str(e)}"
            )
        except:
            print(f"CRITICAL ERROR: {e}")


if __name__ == "__main__":
    # Configure multiprocessing for Windows
    if sys.platform.startswith('win'):
        multiprocessing.freeze_support()
    
    # Run application
    main()            