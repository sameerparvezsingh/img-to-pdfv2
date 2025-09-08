import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
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
import queue
import gc

# Setup logging to both file and console
log_filename = f"image_to_pdf_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_filename),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class Settings:
    """Application settings with resource limits"""
    DEFAULT_QUALITY = 85
    MAX_IMAGE_SIZE = (1920, 1080)
    THUMBNAIL_SIZE = (150, 200)
    
    # Resource management settings
    MAX_CPU_PERCENT = 70  # Maximum CPU usage percentage
    MAX_MEMORY_MB = 1024  # Maximum memory usage in MB
    BATCH_SIZE = 5  # Process images in batches
    MAX_WORKERS = max(1, multiprocessing.cpu_count() // 2)  # Use half of CPU cores
    PROCESS_DELAY = 0.1  # Delay between batches in seconds

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
    """Enhanced progress bar with ETA calculation"""
    
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
        
    def reset(self):
        self.progress['value'] = 0
        self.label.config(text="Ready")
        self.detail_label.config(text="")
        self.start_time = None
        
    def pack(self, **kwargs):
        self.frame.pack(**kwargs)

class OptimizedImageProcessor:
    """Optimized image processing with resource management"""
    
    def __init__(self):
        self.max_workers = Settings.MAX_WORKERS
        
    def preprocess_images_batch(self, file_paths, target_size=None, quality=85, 
                               progress_callback=None, cancel_check=None):
        """Batch process images with resource management"""
        if target_size is None:
            target_size = Settings.MAX_IMAGE_SIZE
            
        total = len(file_paths)
        processed = []
        errors = []
        
        logger.info(f"Starting batch processing of {total} images with {self.max_workers} workers")
        
        try:
            # Process images in batches to manage resources
            for batch_start in range(0, total, Settings.BATCH_SIZE):
                # Check for cancellation
                if cancel_check and cancel_check():
                    logger.info("Processing cancelled by user")
                    break
                
                # Wait if system is under load
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
                        future = executor.submit(self._process_single_image, path, target_size, quality)
                        futures[future] = (i, path)
                    
                    # Process completed futures
                    for future in as_completed(futures):
                        i, path = futures[future]
                        
                        try:
                            result = future.result(timeout=30)
                            if result:
                                processed.append(result)
                            else:
                                errors.append((path, "Failed to process image"))
                                
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
                            
                            if progress_callback:
                                progress_callback(
                                    len(processed) + len(errors), 
                                    total, 
                                    f"Error: {os.path.basename(path)}"
                                )
                
                # Add delay between batches to prevent system overload
                time.sleep(Settings.PROCESS_DELAY)
                
                # Force garbage collection after each batch
                gc.collect()
                
        except Exception as e:
            error_msg = f"Batch processing error: {str(e)}"
            logger.error(error_msg)
            raise Exception(error_msg)
        
        logger.info(f"Processed {len(processed)} images successfully, {len(errors)} errors")
        
        return processed, errors
    
    @staticmethod
    def _process_single_image(path, target_size, quality):
        """Process single image with error handling"""
        try:
            # Validate file exists and is readable
            if not os.path.exists(path):
                raise FileNotFoundError(f"File not found: {path}")
            
            if not os.access(path, os.R_OK):
                raise PermissionError(f"Cannot read file: {path}")
            
            # Check file size
            file_size_mb = os.path.getsize(path) / 1024 / 1024
            if file_size_mb > 50:  # Warn for files larger than 50MB
                logger.warning(f"Large file detected ({file_size_mb:.1f}MB): {path}")
            
            with Image.open(path) as img:
                # Validate image
                img.verify()
                img = Image.open(path)  # Reopen after verify
                
                # Convert to RGB if necessary
                if img.mode in ('RGBA', 'LA', 'P'):
                    rgb_img = Image.new('RGB', img.size, (255, 255, 255))
                    if img.mode == 'P':
                        img = img.convert('RGBA')
                    rgb_img.paste(img, mask=img.split()[-1] if img.mode in ('RGBA', 'LA') else None)
                    img = rgb_img
                elif img.mode != 'RGB':
                    img = img.convert('RGB')
                
                # Resize if needed
                img_width, img_height = img.size
                max_width, max_height = target_size
                
                if img_width > max_width or img_height > max_height:
                    ratio = min(max_width/img_width, max_height/img_height)
                    new_size = (int(img_width * ratio), int(img_height * ratio))
                    img = img.resize(new_size, Image.Resampling.LANCZOS)
                
                # Save to bytes with optimization
                buffer = io.BytesIO()
                img.save(buffer, format='JPEG', quality=quality, optimize=True, progressive=True)
                buffer.seek(0)
                
                return {
                    'data': buffer.getvalue(),
                    'size': img.size,
                    'format': 'JPEG',
                    'path': path
                }
                
        except MemoryError:
            logger.error(f"Memory error processing {path} - file too large")
            return None
        except Exception as e:
            logger.error(f"Image processing error for {path}: {e}")
            return None

class ImageToPDFTool(UIStateMixin):
    def __init__(self, root):
        super().__init__()
        self.root = root
        self.root.title("Image to PDF Converter")
        self.root.geometry("900x700")
        
        # Configure grid weights for proper resizing
        self.root.grid_rowconfigure(0, weight=1)
        self.root.grid_columnconfigure(0, weight=1)
        
        # Initialize components
        self.image_processor = OptimizedImageProcessor()
        
        # File list
        self.image_files = []
        
        # Setup UI
        self.setup_ui()
        self.register_all_ui_elements()
        
        # Drag and drop variables
        self.drag_start_index = None
        
        # Apply styles
        self.apply_styles()
        
        # Log startup
        logger.info("Image to PDF Converter started")
        
    def apply_styles(self):
        style = ttk.Style()
        style.theme_use('clam')
        
        bg_color = '#f0f0f0'
        fg_color = '#333333'
        
        self.root.configure(bg=bg_color)
        
        style.configure('Title.TLabel', font=('Helvetica', 16, 'bold'))
        style.configure('Success.TButton', font=('Helvetica', 10, 'bold'))
        style.configure('Cancel.TButton', font=('Helvetica', 10))

    def setup_ui(self):
        # Main container with padding
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky="nsew")
        
        # Configure main frame grid
        main_frame.grid_rowconfigure(1, weight=1)
        main_frame.grid_columnconfigure(0, weight=1)
        
        # Title
        title_label = ttk.Label(main_frame, text="Convert Images to PDF", style='Title.TLabel')
        title_label.grid(row=0, column=0, pady=(0, 10), sticky="w")
        
        # File selection frame
        file_frame = ttk.Frame(main_frame)
        file_frame.grid(row=1, column=0, sticky="nsew", pady=10)
        
        # Configure file frame grid
        file_frame.grid_rowconfigure(0, weight=1)
        file_frame.grid_columnconfigure(1, weight=1)
        
        # Buttons frame (left side)
        button_frame = ttk.Frame(file_frame)
        button_frame.grid(row=0, column=0, sticky="ns", padx=(0, 10))
        
        # File operations
        file_ops_frame = ttk.LabelFrame(button_frame, text="File Operations", padding=5)
        file_ops_frame.pack(fill='x', pady=(0, 5))
        
        self.select_images_button = ttk.Button(file_ops_frame, text="Add Images", 
                                              command=self.select_image_files)
        self.select_images_button.pack(fill='x', pady=2)
        
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
        
        self.reverse_order_button = ttk.Button(order_ops_frame, text="Reverse Order", 
                                             command=self.reverse_image_order)
        self.reverse_order_button.pack(fill='x', pady=2)
        
        # System info
        info_frame = ttk.LabelFrame(button_frame, text="System Info", padding=5)
        info_frame.pack(fill='x', pady=5)
        
        self.cpu_label = ttk.Label(info_frame, text="CPU: 0%")
        self.cpu_label.pack(anchor='w')
        self.memory_label = ttk.Label(info_frame, text="Memory: 0 MB")
        self.memory_label.pack(anchor='w')
        
        # Update system info periodically
        self.update_system_info()
        
        # Image list (right side)
        list_frame = ttk.Frame(file_frame)
        list_frame.grid(row=0, column=1, sticky="nsew")
        
        # Configure list frame grid
        list_frame.grid_rowconfigure(1, weight=1)
        list_frame.grid_columnconfigure(0, weight=1)
        
        # List label
        ttk.Label(list_frame, text="Images:", font=('Helvetica', 10, 'bold')).grid(
            row=0, column=0, sticky="w", pady=(0, 5))
        
        # Listbox with scrollbar frame
        listbox_frame = ttk.Frame(list_frame)
        listbox_frame.grid(row=1, column=0, sticky="nsew")
        
        # Configure listbox frame grid
        listbox_frame.grid_rowconfigure(0, weight=1)
        listbox_frame.grid_columnconfigure(0, weight=1)
        
        # Create listbox and scrollbar
        self.image_listbox = tk.Listbox(listbox_frame, selectmode='extended')
        scrollbar = ttk.Scrollbar(listbox_frame, orient="vertical")
        
        # Configure scrollbar
        self.image_listbox.config(yscrollcommand=scrollbar.set)
        scrollbar.config(command=self.image_listbox.yview)
        
        # Grid listbox and scrollbar
        self.image_listbox.grid(row=0, column=0, sticky="nsew")
        scrollbar.grid(row=0, column=1, sticky="ns")
        
        # Initially bind drag and drop events
        self.bind_drag_drop_events()
        
        # Image count
        self.image_count_label = ttk.Label(list_frame, text="Total images: 0 | Total size: 0 MB")
        self.image_count_label.grid(row=2, column=0, pady=(5, 0), sticky="w")
        
        # Options frame
        options_frame = ttk.LabelFrame(main_frame, text="Conversion Options", padding=10)
        options_frame.grid(row=2, column=0, sticky="ew", pady=10)
        
        # Compression option
        compress_frame = ttk.Frame(options_frame)
        compress_frame.pack(fill='x', pady=(0, 10))
        
        self.compress_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(compress_frame, text="Compress images", 
                       variable=self.compress_var).pack(side='left')
        
        ttk.Label(compress_frame, text="Quality:").pack(side='left', padx=(20, 5))
        self.quality_var = tk.IntVar(value=85)
        quality_scale = ttk.Scale(compress_frame, from_=10, to=90, 
                                variable=self.quality_var, orient='horizontal', length=200)
        quality_scale.pack(side='left', padx=5)
        self.quality_label = ttk.Label(compress_frame, text="85%")
        self.quality_label.pack(side='left')
        
        quality_scale.config(command=lambda v: self.quality_label.config(text=f"{int(float(v))}%"))
        
        # Page size option
        size_frame = ttk.Frame(options_frame)
        size_frame.pack(fill='x')
        
        ttk.Label(size_frame, text="Page Size:").pack(side='left', padx=(0, 10))
        self.page_size_var = tk.StringVar(value="A4")
        ttk.Radiobutton(size_frame, text="A4", variable=self.page_size_var, 
                       value="A4").pack(side='left', padx=5)
        ttk.Radiobutton(size_frame, text="Letter", variable=self.page_size_var, 
                       value="Letter").pack(side='left', padx=5)
        ttk.Radiobutton(size_frame, text="Fit to Image", variable=self.page_size_var, 
                       value="Fit").pack(side='left', padx=5)
        
        # Progress bar
        self.img_progress = EnhancedProgressBar(main_frame, self.root)
        self.img_progress.frame.grid(row=3, column=0, sticky="ew", pady=10)
        
        # Convert and Cancel buttons
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
        
    def update_system_info(self):
        """Update system resource information"""
        try:
            cpu_usage = ResourceMonitor.get_cpu_usage()
            memory_usage = ResourceMonitor.get_memory_usage()
            
            self.cpu_label.config(text=f"CPU: {cpu_usage:.1f}%")
            self.memory_label.config(text=f"Memory: {memory_usage:.0f} MB")
            
            # Schedule next update
            self.root.after(1000, self.update_system_info)
        except Exception as e:
            logger.debug(f"System info update error: {e}")
        
    def register_all_ui_elements(self):
        """Register all UI elements for state management"""
        self.register_ui_element(self.select_images_button)
        self.register_ui_element(self.clear_images_button)
        self.register_ui_element(self.remove_selected_button)
        self.register_ui_element(self.move_image_up_button)
        self.register_ui_element(self.move_image_down_button)
        self.register_ui_element(self.reverse_order_button)
        self.register_ui_element(self.convert_to_pdf_button)

    def select_image_files(self):
        """Select and add image files to the list"""
        # if self.operation_in_progress:
        #     return
            
        try:
            files = filedialog.askopenfilenames(
                title="Select image files",
                filetypes=[
                    ("Image files", "*.jpg *.jpeg *.png *.bmp *.gif *.tiff *.webp"),
                    ("All files", "*.*")
                ]
            )
            self.image_files.extend(files)
            added_count = 0
            for file in files:
                self.image_listbox.insert(tk.END, os.path.basename(file))
                # if file not in self.image_files:
                #     self.image_files.append(file)
                #     self.image_listbox.insert(tk.END, os.path.basename(file))
                #     added_count += 1
            
            if added_count > 0:
                logger.info(f"Added {added_count} images")
                
            self.update_image_count()
            
        except Exception as e:
            error_msg = f"Error selecting files: {str(e)}"
            logger.error(error_msg)
            messagebox.showerror("Error", error_msg)

    def clear_image_list(self):
        """Clear all images from the list"""
        if self.operation_in_progress:
            return
            
        if self.image_files:
            if messagebox.askyesno("Confirm", "Clear all images from the list?"):
                self.image_files.clear()
                self.image_listbox.delete(0, tk.END)
                self.img_progress.reset()
                self.update_image_count()
                logger.info("Cleared all images")

    def remove_selected_images(self):
        """Remove selected images from the list"""
        if self.operation_in_progress:
            return
            
        selected_indices = list(self.image_listbox.curselection())
        if not selected_indices:
            messagebox.showwarning("Warning", "Please select images to remove.")
            return
        
        # Remove in reverse order
        for idx in reversed(selected_indices):
            removed_file = self.image_files[idx]
            del self.image_files[idx]
            self.image_listbox.delete(idx)
            logger.info(f"Removed {os.path.basename(removed_file)}")
        
        self.update_image_count()

    def move_image_up(self):
        """Move selected image up in the list"""
        if self.operation_in_progress:
            return
            
        selected = self.image_listbox.curselection()
        if not selected:
            messagebox.showwarning("Warning", "Please select an image to move.")
            return
        
        idx = selected[0]
        if idx > 0:
            self.image_files[idx], self.image_files[idx-1] = self.image_files[idx-1], self.image_files[idx]
            
            item = self.image_listbox.get(idx)
            self.image_listbox.delete(idx)
            self.image_listbox.insert(idx-1, item)
            self.image_listbox.selection_set(idx-1)

    def move_image_down(self):
        """Move selected image down in the list"""
        if self.operation_in_progress:
            return
            
        selected = self.image_listbox.curselection()
        if not selected:
            messagebox.showwarning("Warning", "Please select an image to move.")
            return
        
        idx = selected[0]
        if idx < len(self.image_files) - 1:
            self.image_files[idx], self.image_files[idx+1] = self.image_files[idx+1], self.image_files[idx]
            
            item = self.image_listbox.get(idx)
            self.image_listbox.delete(idx)
            self.image_listbox.insert(idx+1, item)
            self.image_listbox.selection_set(idx+1)

    def reverse_image_order(self):
        """Reverse the order of all images"""
        if self.operation_in_progress or not self.image_files:
            return
        
        self.image_files.reverse()
        
        self.image_listbox.delete(0, tk.END)
        for file in self.image_files:
            self.image_listbox.insert(tk.END, os.path.basename(file))
        
        logger.info("Reversed image order")

    def update_image_count(self):
        """Update the image count and total size label"""
        count = len(self.image_files)
        
        # Calculate total size
        total_size = 0
        for file in self.image_files:
            try:
                total_size += os.path.getsize(file)
            except:
                pass
        
        total_size_mb = total_size / (1024 * 1024)
        self.image_count_label.config(text=f"Total images: {count} | Total size: {total_size_mb:.1f} MB")

    # Drag and drop functionality
    def on_image_click(self, event):
        """Handle mouse click on image list"""
        if not self.drag_drop_enabled or self.operation_in_progress:
            return
            
        self.drag_start_index = self.image_listbox.nearest(event.y)
        self.image_listbox.selection_clear(0, tk.END)
        self.image_listbox.selection_set(self.drag_start_index)

    def on_image_drag(self, event):
        """Handle dragging of image"""
        if not self.drag_drop_enabled or self.operation_in_progress or self.drag_start_index is None:
            return
        
        # Visual feedback
        self.image_listbox.config(cursor="hand2")

    def on_image_drop(self, event):
        """Handle dropping of image"""
        if not self.drag_drop_enabled or self.operation_in_progress or self.drag_start_index is None:
            return
        
        # Reset cursor
        self.image_listbox.config(cursor="")
        
        # Get the drop position
        drop_index = self.image_listbox.nearest(event.y)
        
        if drop_index != self.drag_start_index and drop_index < len(self.image_files):
            # Move the item in both lists
            item = self.image_files.pop(self.drag_start_index)
            self.image_files.insert(drop_index, item)
            
            # Update listbox
            listbox_item = self.image_listbox.get(self.drag_start_index)
            self.image_listbox.delete(self.drag_start_index)
            self.image_listbox.insert(drop_index, listbox_item)
            
            # Select the moved item
            self.image_listbox.selection_clear(0, tk.END)
            self.image_listbox.selection_set(drop_index)
        
        # Reset drag variables
        self.drag_start_index = None

    def images_to_pdf_thread(self):
        """Thread wrapper for image to PDF conversion"""
        if self.check_operation_in_progress():
            return
            
        if not self.image_files:
            messagebox.showwarning("Warning", "Please select at least one image.")
            return
        
        # Validate all files exist
        missing_files = []
        for file in self.image_files:
            if not os.path.exists(file):
                missing_files.append(file)
        
        if missing_files:
            error_msg = f"The following files were not found:\n" + "\n".join(missing_files[:5])
            if len(missing_files) > 5:
                error_msg += f"\n... and {len(missing_files) - 5} more"
            messagebox.showerror("Error", error_msg)
            
            # Remove missing files from list
            for file in missing_files:
                if file in self.image_files:
                    idx = self.image_files.index(file)
                    self.image_files.remove(file)
                    self.image_listbox.delete(idx)
            
            self.update_image_count()
            return
        
        thread = threading.Thread(target=self.images_to_pdf_worker, daemon=True)
        thread.start()

    def images_to_pdf_worker(self):
        """Worker function for converting images to PDF"""
        self.operation_started()
        output_path = None
        
        try:
            # Get output path first (in main thread context)
            output_path = filedialog.asksaveasfilename(
                defaultextension=".pdf",
                filetypes=[("PDF files", "*.pdf")]
            )
            
            if not output_path:
                return
            
            logger.info(f"Starting PDF creation: {output_path}")
            
            compress = self.compress_var.get()
            quality = self.quality_var.get()
            page_size = self.page_size_var.get()
            
            # Progress callback
            def progress_callback(current, total, message="", detail=""):
                self.root.after(0, lambda: self.img_progress.update(current, message, detail))
            
            # Cancel check callback
            def cancel_check():
                return self._cancel_requested
            
            # Start progress
            total_steps = len(self.image_files) + 1
            self.root.after(0, lambda: self.img_progress.start(total_steps, "Processing images..."))
            
            # Process images with resource management
            if compress:
                processed_images, errors = self.image_processor.preprocess_images_batch(
                    self.image_files,
                    target_size=Settings.MAX_IMAGE_SIZE,
                    quality=quality,
                    progress_callback=progress_callback,
                    cancel_check=cancel_check
                )
            else:
                processed_images = []
                errors = []
                
                for i, path in enumerate(self.image_files):
                    if cancel_check():
                        logger.info("Processing cancelled")
                        break
                    
                    try:
                        # Check file size before loading
                        # file_size_mb = os.path.getsize(path) / (1024 * 1024)
                        if file_size_mb > 50:
                            logger.warning(f"Large file: {path} ({file_size_mb:.1f} MB)")
                        
                        with open(path, 'rb') as f:
                            processed_images.append({
                                'data': f.read(),
                                'path': path
                            })
                        progress_callback(
                            i + 1, 
                            len(self.image_files), 
                            f"Loading {os.path.basename(path)}",
                            f"File {i + 1} of {len(self.image_files)}"
                        )
                        
                    except Exception as e:
                        error_msg = f"Error loading {path}: {str(e)}"
                        logger.error(error_msg)
                        errors.append((path, str(e)))
            
            # Check if cancelled
            if self._cancel_requested:
                self.root.after(0, lambda: messagebox.showinfo("Cancelled", "Operation was cancelled."))
                return
            
            # Show errors if any
            if errors:
                error_msg = "Some images could not be processed:\n\n"
                for path, error in errors[:5]:  # Show first 5 errors
                    error_msg += f"• {os.path.basename(path)}: {error}\n"
                if len(errors) > 5:
                    error_msg += f"\n... and {len(errors) - 5} more errors"
                
                self.root.after(0, lambda: messagebox.showwarning("Warnings", error_msg))
                
                if not processed_images:
                    raise Exception("No images could be processed successfully")
            
            # Create PDF
            self.root.after(0, lambda: self.img_progress.update(
                len(processed_images), 
                "Creating PDF...",
                "This may take a moment for large files"
            ))
            
            # Determine page size
            if page_size == "A4":
                pdf_page_size = A4
            elif page_size == "Letter":
                pdf_page_size = letter
            else:
                pdf_page_size = None
            
            # Create PDF with resource monitoring
            c = canvas.Canvas(output_path, pagesize=pdf_page_size if pdf_page_size else A4)
            
            for idx, img_data in enumerate(processed_images):
                if cancel_check():
                    logger.info("PDF creation cancelled")
                    c.save()  # Save what we have
                    if os.path.exists(output_path):
                        os.remove(output_path)
                    break
                
                try:
                    # Load image
                    img = Image.open(io.BytesIO(img_data['data']))
                    img_width, img_height = img.size
                    
                    if page_size == "Fit":
                        # Set page size to image size
                        c.setPageSize((img_width, img_height))
                        c.drawImage(ImageReader(img), 0, 0, width=img_width, height=img_height)
                    else:
                        # Fit image to page
                        page_width, page_height = pdf_page_size
                        
                        # Calculate scaling
                        scale = min(page_width/img_width, page_height/img_height, 1.0)
                        scaled_width = img_width * scale
                        scaled_height = img_height * scale
                        
                        # Center image
                        x = (page_width - scaled_width) / 2
                        y = (page_height - scaled_height) / 2
                        
                        c.drawImage(ImageReader(img), x, y, width=scaled_width, height=scaled_height)
                    
                    c.showPage()
                    img.close()
                    
                    # Force garbage collection periodically
                    if idx % 10 == 0:
                        gc.collect()
                    
                except Exception as e:
                    logger.error(f"Error adding image to PDF: {e}")
            
            # Save PDF
            if not self._cancel_requested:
                c.save()
                
                # Verify PDF was created
                if os.path.exists(output_path):
                    file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
                    
                    self.root.after(0, lambda: self.img_progress.complete("PDF created successfully!"))
                    
                    success_msg = (
                        f"PDF created successfully!\n\n"
                        f"File: {os.path.basename(output_path)}\n"
                        f"Size: {file_size_mb:.2f} MB\n"
                        f"Pages: {len(processed_images)}\n"
                        f"Location: {output_path}"
                    )
                    
                    self.root.after(0, lambda: messagebox.showinfo("Success", success_msg))
                    logger.info(f"PDF created: {output_path} ({file_size_mb:.2f} MB)")
                else:
                    self.root.after(0, lambda: self.img_progress.complete("Operation cancelled!"))
                    raise Exception("PDF file was not created")
            
        except MemoryError:
            error_msg = "Out of memory! Try:\n• Processing fewer images at once\n• Enabling compression\n• Reducing quality"
            logger.error(f"Memory error creating PDF")
            self.root.after(0, lambda: messagebox.showerror("Memory Error", error_msg))
            
        except Exception as e:
            error_msg = f"An error occurred: {str(e)}"
            logger.error(f"PDF creation error: {e}")
            self.root.after(0, lambda: messagebox.showerror("Error", error_msg))
            
            # Clean up partial PDF
            if output_path and os.path.exists(output_path):
                try:
                    os.remove(output_path)
                except:
                    pass
                    
        finally:
            # Force garbage collection
            gc.collect()
            self.operation_completed()


def main():
    """Main entry point"""
    try:
        # Check for required libraries
        import psutil
        
        root = tk.Tk()
        
        # Make the window properly resizable
        root.minsize(600, 500)
        
        # Create app
        app = ImageToPDFTool(root)
        
        logger.info(f"Starting with {Settings.MAX_WORKERS} workers, max CPU: {Settings.MAX_CPU_PERCENT}%")
        
        root.mainloop()
        
    except ImportError as e:
        messagebox.showerror(
            "Missing Dependency",
            f"Required library not installed: {str(e)}\n\n"
            "Please install with:\n"
            "pip install Pillow reportlab psutil"
        )
    except Exception as e:
        logger.critical(f"Failed to start application: {e}")
        messagebox.showerror("Startup Error", f"Failed to start application:\n{str(e)}")


if __name__ == "__main__":
    # Configure multiprocessing for Windows
    import sys
    if sys.platform.startswith('win'):
        multiprocessing.freeze_support()
    
    main()