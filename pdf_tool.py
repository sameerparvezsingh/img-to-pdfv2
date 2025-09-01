#!/usr/bin/env python3
"""
Image to PDF Converter
A GUI application for converting images to PDF files with batch processing support.
Author: Senior Software Engineer
Version: 1.4
"""

import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import os
import sys
from PIL import Image, ImageTk
import threading
from pathlib import Path
import gc

class ImageToPDFConverter:
    """Main application class for Image to PDF Converter"""
    
    def __init__(self, root):
        self.root = root
        self.root.title("Image to PDF Converter")
        self.root.geometry("900x750")
        self.root.minsize(600, 500)
        
        # Initialize variables
        self.selected_images = []
        self.image_paths = []
        self.thumbnail_refs = []  # Keep references to prevent garbage collection
        self.pdf_option = tk.StringVar(value="single")
        self.output_directory = tk.StringVar(value=str(Path.home()))
        self.output_filename = tk.StringVar(value="converted")
        
        # Quality setting variable (initialize before A4 dimensions)
        self.quality_var = tk.IntVar(value=300)  # Default 300 DPI
        
        # A4 dimensions in points (72 points = 1 inch)
        self.A4_WIDTH = 595
        self.A4_HEIGHT = 842
        self.SCREEN_DPI = 72  # Standard PDF points DPI
        
        # Configure root window for responsive design
        self.root.grid_rowconfigure(0, weight=1)
        self.root.grid_columnconfigure(0, weight=1)
        
        # Setup UI components
        self.setup_ui()
        
        # Bind window close event for cleanup
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        
    def setup_ui(self):
        """Create and arrange all UI elements with main window scrolling"""
        
        # Create main scrollable area
        main_scroll_frame = ttk.Frame(self.root)
        main_scroll_frame.grid(row=0, column=0, sticky="nsew")
        main_scroll_frame.grid_rowconfigure(0, weight=1)
        main_scroll_frame.grid_columnconfigure(0, weight=1)
        
        # Create canvas for main content
        self.main_canvas = tk.Canvas(main_scroll_frame, highlightthickness=0)
        self.main_canvas.grid(row=0, column=0, sticky="nsew")
        
        # Add scrollbars for main window
        main_v_scrollbar = ttk.Scrollbar(main_scroll_frame, orient="vertical", command=self.main_canvas.yview)
        main_h_scrollbar = ttk.Scrollbar(main_scroll_frame, orient="horizontal", command=self.main_canvas.xview)
        main_v_scrollbar.grid(row=0, column=1, sticky="ns")
        main_h_scrollbar.grid(row=1, column=0, sticky="ew")
        
        self.main_canvas.configure(
            yscrollcommand=main_v_scrollbar.set,
            xscrollcommand=main_h_scrollbar.set
        )
        
        # Create frame inside canvas for all content
        self.main_content_frame = ttk.Frame(self.main_canvas)
        self.main_canvas_window = self.main_canvas.create_window((0, 0), window=self.main_content_frame, anchor="nw")
        
        # Main container with padding (now inside scrollable area)
        main_container = ttk.Frame(self.main_content_frame, padding="10")
        main_container.grid(row=0, column=0, sticky="nsew")
        main_container.grid_rowconfigure(1, weight=1)
        main_container.grid_columnconfigure(0, weight=1)
        
        # Configure minimum content size
        self.main_content_frame.grid_rowconfigure(0, weight=1)
        self.main_content_frame.grid_columnconfigure(0, weight=1)
        
        # Top Control Panel
        self.create_control_panel(main_container)
        
        # Middle Preview Area
        self.create_preview_area(main_container)
        
        # Bottom Status Bar
        self.create_status_bar(main_container)
        
        # Configure styles
        self.configure_styles()
        
        # Bind canvas configure events
        self.main_content_frame.bind("<Configure>", self.on_main_frame_configure)
        self.main_canvas.bind("<Configure>", self.on_main_canvas_configure)
        
        # Enable mouse wheel scrolling
        self.bind_mouse_wheel()
        
    def create_control_panel(self, parent):
        """Create the top control panel with all buttons and options"""
        control_frame = ttk.LabelFrame(parent, text="Controls", padding="10")
        control_frame.grid(row=0, column=0, sticky="ew", pady=(0, 10))
        control_frame.grid_columnconfigure(0, weight=1)
        
        # Select Images Button
        self.select_btn = ttk.Button(
            control_frame,
            text="📁 Select Images",
            command=self.select_images,
            style="Action.TButton"
        )
        self.select_btn.grid(row=0, column=0, columnspan=2, sticky="ew", pady=5)
        
        # PDF Options
        options_frame = ttk.LabelFrame(control_frame, text="PDF Options", padding="10")
        options_frame.grid(row=1, column=0, columnspan=2, sticky="ew", pady=10)
        
        ttk.Radiobutton(
            options_frame,
            text="Single PDF (All images combined)",
            variable=self.pdf_option,
            value="single"
        ).pack(anchor="w", pady=2)
        
        ttk.Radiobutton(
            options_frame,
            text="Individual PDFs (One per image)",
            variable=self.pdf_option,
            value="multiple"
        ).pack(anchor="w", pady=2)
        
        # Output Settings
        output_frame = ttk.LabelFrame(control_frame, text="Output Settings", padding="10")
        output_frame.grid(row=2, column=0, columnspan=2, sticky="ew", pady=10)
        output_frame.grid_columnconfigure(1, weight=1)
        
        # Filename
        ttk.Label(output_frame, text="Filename:").grid(row=0, column=0, sticky="w", padx=5, pady=5)
        self.filename_entry = ttk.Entry(output_frame, textvariable=self.output_filename)
        self.filename_entry.grid(row=0, column=1, sticky="ew", padx=5, pady=5)
        ttk.Label(output_frame, text=".pdf").grid(row=0, column=2, sticky="w", pady=5)
        
        # Output Directory
        ttk.Label(output_frame, text="Save to:").grid(row=1, column=0, sticky="w", padx=5, pady=5)
        self.dir_label = ttk.Entry(output_frame, textvariable=self.output_directory, state="readonly")
        self.dir_label.grid(row=1, column=1, sticky="ew", padx=5, pady=5)
        
        ttk.Button(
            output_frame,
            text="Browse...",
            command=self.select_output_directory
        ).grid(row=1, column=2, padx=5, pady=5)
        
        # Quality Settings
        quality_frame = ttk.LabelFrame(control_frame, text="A4 PDF Quality Settings", padding="10")
        quality_frame.grid(row=3, column=0, columnspan=2, sticky="ew", pady=10)
        quality_frame.grid_columnconfigure(1, weight=1)
        
        ttk.Label(quality_frame, text="Quality:").grid(row=0, column=0, sticky="w", padx=5)
        
        # Quality selection with description
        quality_combo = ttk.Combobox(
            quality_frame, 
            textvariable=self.quality_var,
            state="readonly",
            width=15
        )
        quality_combo['values'] = (150, 200, 300, 600)
        quality_combo.grid(row=0, column=1, sticky="w", padx=5)
        
        # DPI label
        ttk.Label(quality_frame, text="DPI").grid(row=0, column=2, sticky="w", padx=5)
        
        # Quality description label
        self.quality_desc_label = ttk.Label(
            quality_frame, 
            text="High quality (recommended)",
            font=("Arial", 9),
            foreground="gray"
        )
        self.quality_desc_label.grid(row=1, column=1, sticky="w", padx=5, pady=(2, 0))
        
        # Update description when quality changes
        def update_quality_description(*args):
            dpi = self.quality_var.get()
            descriptions = {
                150: "Low quality (smaller file size)",
                200: "Medium quality (balanced)",
                300: "High quality (recommended)",
                600: "Very high quality (large files)"
            }
            self.quality_desc_label.config(text=descriptions.get(dpi, ""))
            
            # Also update status to show impact
            if self.image_paths:
                file_size_estimate = len(self.image_paths) * (dpi / 150) * 0.5  # Rough estimate in MB
                self.update_status(f"Estimated A4 PDF size: ~{file_size_estimate:.1f} MB at {dpi} DPI")
        
        self.quality_var.trace('w', update_quality_description)
        
        # Action Buttons Frame
        action_frame = ttk.Frame(control_frame)
        action_frame.grid(row=4, column=0, columnspan=2, sticky="ew", pady=10)
        action_frame.grid_columnconfigure(0, weight=1)
        action_frame.grid_columnconfigure(1, weight=1)
        action_frame.grid_columnconfigure(2, weight=1)
        
        # Clear Button
        self.clear_btn = ttk.Button(
            action_frame,
            text="🗑️ Clear All",
            command=self.clear_all,
            state="disabled"
        )
        self.clear_btn.grid(row=0, column=0, sticky="ew", padx=5)
        
        # Convert Button
        self.convert_btn = ttk.Button(
            action_frame,
            text="📄 Convert to PDF",
            command=self.convert_to_pdf,
            style="Action.TButton",
            state="disabled"
        )
        self.convert_btn.grid(row=0, column=1, sticky="ew", padx=5)
        
        # Convert to A4 PDF Button
        self.convert_a4_btn = ttk.Button(
            action_frame,
            text="📄 Convert to A4 PDF",
            command=self.convert_to_a4_pdf,
            style="Action.TButton",
            state="disabled"
        )
        self.convert_a4_btn.grid(row=0, column=2, sticky="ew", padx=5)
        
    def create_preview_area(self, parent):
        """Create the image preview area with scrolling support"""
        preview_frame = ttk.LabelFrame(parent, text="Image Preview", padding="10")
        preview_frame.grid(row=1, column=0, sticky="nsew", pady=10)
        preview_frame.grid_rowconfigure(0, weight=1)
        preview_frame.grid_columnconfigure(0, weight=1)
        
        # Create canvas with scrollbars
        canvas_frame = ttk.Frame(preview_frame)
        canvas_frame.grid(row=0, column=0, sticky="nsew")
        canvas_frame.grid_rowconfigure(0, weight=1)
        canvas_frame.grid_columnconfigure(0, weight=1)
        
        # Canvas
        self.canvas = tk.Canvas(canvas_frame, bg="white", highlightthickness=1, highlightbackground="gray")
        self.canvas.grid(row=0, column=0, sticky="nsew")
        
        # Scrollbars
        v_scrollbar = ttk.Scrollbar(canvas_frame, orient="vertical", command=self.canvas.yview)
        h_scrollbar = ttk.Scrollbar(canvas_frame, orient="horizontal", command=self.canvas.xview)
        v_scrollbar.grid(row=0, column=1, sticky="ns")
        h_scrollbar.grid(row=1, column=0, sticky="ew")
        
        self.canvas.configure(
            yscrollcommand=v_scrollbar.set,
            xscrollcommand=h_scrollbar.set
        )
        
        # Frame inside canvas for images
        self.images_frame = ttk.Frame(self.canvas)
        self.canvas_window = self.canvas.create_window((0, 0), window=self.images_frame, anchor="nw")
        
        # Bind events
        self.images_frame.bind("<Configure>", self.on_frame_configure)
        self.canvas.bind("<Configure>", self.on_canvas_configure)
        
        # Instructions label (shown when no images selected)
        self.instructions_label = ttk.Label(
            self.images_frame,
            text="No images selected.\nClick 'Select Images' to choose images for conversion.",
            font=("Arial", 12),
            foreground="gray"
        )
        self.instructions_label.grid(row=0, column=0, padx=50, pady=50)
        
    def create_status_bar(self, parent):
        """Create the status bar at the bottom"""
        status_frame = ttk.Frame(parent)
        status_frame.grid(row=2, column=0, sticky="ew", pady=(10, 0))
        status_frame.grid_columnconfigure(0, weight=1)
        
        self.status_label = ttk.Label(status_frame, text="Ready", relief="sunken", anchor="w")
        self.status_label.grid(row=0, column=0, sticky="ew")
        
        self.progress_bar = ttk.Progressbar(status_frame, mode='indeterminate')
        self.progress_bar.grid(row=0, column=1, sticky="e", padx=(10, 0))
        self.progress_bar.grid_remove()
        
    def configure_styles(self):
        """Configure custom styles for widgets"""
        style = ttk.Style()
        style.configure("Action.TButton", font=("Arial", 10, "bold"))
        style.configure("Delete.TButton", font=("Arial", 8), foreground="red")
        
    def select_images(self):
        """Open file dialog to select images"""
        filetypes = (
            ("Image files", "*.jpg *.jpeg *.png *.bmp *.gif *.tiff *.tif *.webp"),
            ("All files", "*.*")
        )
        
        filenames = filedialog.askopenfilenames(
            title="Select Images",
            initialdir=self.output_directory.get(),
            filetypes=filetypes
        )
        
        if filenames:
            self.image_paths = list(filenames)
            self.display_selected_images()
            self.update_button_states()
            self.update_status(f"Selected {len(self.image_paths)} image(s)")
            
    def delete_image(self, index):
        """Delete a specific image from the selection"""
        if 0 <= index < len(self.image_paths):
            deleted_image = self.image_paths.pop(index)
            self.update_status(f"Removed: {os.path.basename(deleted_image)}")
            self.display_selected_images()
            self.update_button_states()
            
    def display_selected_images(self):
        """Display thumbnails of selected images with delete buttons"""
        # Clear previous content
        for widget in self.images_frame.winfo_children():
            widget.destroy()
        
        # Clear references
        self.thumbnail_refs.clear()
        self.selected_images.clear()
        
        if not self.image_paths:
            # Show instructions if no images
            self.instructions_label = ttk.Label(
                self.images_frame,
                text="No images selected.\nClick 'Select Images' to choose images for conversion.",
                font=("Arial", 12),
                foreground="gray"
            )
            self.instructions_label.grid(row=0, column=0, padx=50, pady=50)
            return
        
        # Create image grid
        row = 0
        col = 0
        max_cols = 4  # Maximum columns before wrapping
        
        for idx, image_path in enumerate(self.image_paths):
            try:
                # Create main container frame for image and controls
                main_container = ttk.Frame(self.images_frame)
                main_container.grid(row=row, column=col, padx=10, pady=10)
                
                # Create image container frame
                img_container = ttk.Frame(main_container, relief="raised", borderwidth=2)
                img_container.pack(fill="both", expand=True)
                
                # Load and create thumbnail
                img = Image.open(image_path)
                # Calculate thumbnail size
                img.thumbnail((200, 200), Image.Resampling.LANCZOS)
                photo = ImageTk.PhotoImage(img)
                
                # Store reference to prevent garbage collection
                self.thumbnail_refs.append(photo)
                
                # Display image
                img_label = ttk.Label(img_container, image=photo)
                img_label.pack(padx=5, pady=5)
                
                # Add filename
                filename = os.path.basename(image_path)
                if len(filename) > 25:
                    filename = filename[:22] + "..."
                    
                name_label = ttk.Label(
                    img_container,
                    text=f"{idx + 1}. {filename}",
                    font=("Arial", 9)
                )
                name_label.pack(pady=(0, 5))
                
                # Add delete button with hover effect
                delete_btn = tk.Button(
                    main_container,
                    text="❌ Remove",
                    command=lambda i=idx: self.delete_image(i),
                    bg="#ffcccc",
                    fg="#cc0000",
                    font=("Arial", 9),
                    cursor="hand2",
                    relief="raised",
                    bd=1
                )
                delete_btn.pack(fill="x", pady=(5, 0))
                
                # Hover effects for delete button
                delete_btn.bind("<Enter>", lambda e, btn=delete_btn: btn.config(bg="#ff9999"))
                delete_btn.bind("<Leave>", lambda e, btn=delete_btn: btn.config(bg="#ffcccc"))
                
                # Update grid position
                col += 1
                if col >= max_cols:
                    col = 0
                    row += 1
                    
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load image: {os.path.basename(image_path)}\n{str(e)}")
                
    def select_output_directory(self):
        """Open dialog to select output directory"""
        directory = filedialog.askdirectory(
            title="Select Output Directory",
            initialdir=self.output_directory.get()
        )
        
        if directory:
            self.output_directory.set(directory)
            
    def clear_all(self):
        """Clear all selected images and reset the application"""
        self.image_paths.clear()
        self.selected_images.clear()
        self.thumbnail_refs.clear()
        self.display_selected_images()
        self.update_button_states()
        self.update_status("Cleared all images")
        
    def convert_to_pdf(self):
        """Convert selected images to PDF(s) with original dimensions"""
        if not self.image_paths:
            messagebox.showwarning("No Images", "Please select images first.")
            return
            
        # Start conversion in separate thread
        thread = threading.Thread(target=self._perform_conversion, args=(False,))
        thread.daemon = True
        thread.start()
        
    def convert_to_a4_pdf(self):
        """Convert selected images to PDF(s) with A4 page size"""
        if not self.image_paths:
            messagebox.showwarning("No Images", "Please select images first.")
            return
            
        # Start conversion in separate thread
        thread = threading.Thread(target=self._perform_conversion, args=(True,))
        thread.daemon = True
        thread.start()
        
    def _perform_conversion(self, use_a4=False):
        """Perform the actual conversion (runs in separate thread)"""
        try:
            # Show progress
            self.root.after(0, self.show_progress)
            conversion_type = "A4 PDF" if use_a4 else "PDF"
            self.root.after(0, lambda: self.update_status(f"Converting to {conversion_type}..."))
            
            if self.pdf_option.get() == "single":
                if use_a4:
                    self._create_single_a4_pdf()
                else:
                    self._create_single_pdf()
            else:
                if use_a4:
                    self._create_multiple_a4_pdfs()
                else:
                    self._create_multiple_pdfs()
                
        except Exception as e:
            self.root.after(0, lambda: messagebox.showerror("Conversion Error", str(e)))
        finally:
            self.root.after(0, self.hide_progress)
            self.root.after(0, lambda: self.update_status("Ready"))
            
    def _create_single_pdf(self):
        """Create a single PDF from all images with original dimensions"""
        try:
            # Prepare images
            images = []
            for path in self.image_paths:
                img = Image.open(path)
                # Convert to RGB if necessary
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                images.append(img)
            
            # Create output path
            filename = f"{self.output_filename.get()}.pdf"
            output_path = os.path.join(self.output_directory.get(), filename)
            
            # Save as PDF
            if images:
                images[0].save(
                    output_path,
                    "PDF",
                    save_all=True,
                    append_images=images[1:]
                )
                
            # Clean up
            for img in images:
                img.close()
                
            self.root.after(0, lambda: self.show_success_message(output_path, 1))
            
        except Exception as e:
            raise Exception(f"Failed to create PDF: {str(e)}")
            
    def _create_single_a4_pdf(self):
        """Create a single PDF with all images fitted to A4 pages"""
        try:
            # Prepare images
            a4_images = []
            
            for path in self.image_paths:
                img = Image.open(path)
                # Convert to RGB if necessary
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                
                # Create A4 canvas
                a4_img = self._fit_image_to_a4(img)
                a4_images.append(a4_img)
                img.close()
            
            # Create output path
            filename = f"{self.output_filename.get()}_A4.pdf"
            output_path = os.path.join(self.output_directory.get(), filename)
            
            # Save as PDF with quality settings
            if a4_images:
                a4_images[0].save(
                    output_path,
                    "PDF",
                    save_all=True,
                    append_images=a4_images[1:],
                    resolution=self.quality_var.get(),
                    quality=95,
                    optimize=True
                )
                
            # Clean up
            for img in a4_images:
                img.close()
                
            self.root.after(0, lambda: self.show_success_message(output_path, 1))
            
        except Exception as e:
            raise Exception(f"Failed to create A4 PDF: {str(e)}")
            
    def _create_multiple_pdfs(self):
        """Create individual PDFs for each image with original dimensions"""
        try:
            created_files = []
            
            for idx, path in enumerate(self.image_paths):
                # Open and convert image
                img = Image.open(path)
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                
                # Create filename
                base_name = os.path.splitext(os.path.basename(path))[0]
                filename = f"{self.output_filename.get()}_{idx+1}_{base_name}.pdf"
                output_path = os.path.join(self.output_directory.get(), filename)
                
                # Save as PDF
                img.save(output_path, "PDF")
                img.close()
                
                created_files.append(output_path)
                
            self.root.after(0, lambda: self.show_success_message(created_files[0], len(created_files)))
            
        except Exception as e:
            raise Exception(f"Failed to create PDFs: {str(e)}")
            
    def _create_multiple_a4_pdfs(self):
        """Create individual A4 PDFs for each image"""
        try:
            created_files = []
            
            for idx, path in enumerate(self.image_paths):
                # Open and convert image
                img = Image.open(path)
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                
                # Create A4 version
                a4_img = self._fit_image_to_a4(img)
                img.close()
                
                # Create filename
                base_name = os.path.splitext(os.path.basename(path))[0]
                filename = f"{self.output_filename.get()}_{idx+1}_{base_name}_A4.pdf"
                output_path = os.path.join(self.output_directory.get(), filename)
                
                # Save as PDF with quality settings
                a4_img.save(
                    output_path, 
                    "PDF",
                    resolution=self.quality_var.get(),
                    quality=95,
                    optimize=True
                )
                a4_img.close()
                
                created_files.append(output_path)
                
            self.root.after(0, lambda: self.show_success_message(created_files[0], len(created_files)))
            
        except Exception as e:
            raise Exception(f"Failed to create A4 PDFs: {str(e)}")
            
    def _fit_image_to_a4(self, img):
        """Fit an image to A4 dimensions while maintaining aspect ratio and quality"""
        # Get current quality setting
        current_dpi = self.quality_var.get()
        
        # A4 dimensions at selected DPI for quality
        dpi_scale = current_dpi / self.SCREEN_DPI
        a4_width_px = int(self.A4_WIDTH * dpi_scale)
        a4_height_px = int(self.A4_HEIGHT * dpi_scale)
        
        # Create white A4 canvas at selected resolution
        a4_img = Image.new('RGB', (a4_width_px, a4_height_px), 'white')
        
        # Get image dimensions
        img_width, img_height = img.size
        
        # Calculate scaling factor to fit image within A4
        width_ratio = a4_width_px / img_width
        height_ratio = a4_height_px / img_height
        scale_factor = min(width_ratio, height_ratio)
        
        # Only downscale if necessary, never upscale beyond original
        if scale_factor > 1:
            # Image is smaller than A4, use original size to maintain quality
            new_width = img_width
            new_height = img_height
        else:
            # Image is larger than A4, scale down
            new_width = int(img_width * scale_factor)
            new_height = int(img_height * scale_factor)
        
        # Resize image with high quality resampling
        if (new_width, new_height) != (img_width, img_height):
            resized_img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
        else:
            resized_img = img
        
        # Calculate position to center the image on A4
        x_offset = (a4_width_px - new_width) // 2
        y_offset = (a4_height_px - new_height) // 2
        
        # Paste resized image onto A4 canvas
        a4_img.paste(resized_img, (x_offset, y_offset))
        
        # Set DPI info for proper PDF generation
        a4_img.info['dpi'] = (current_dpi, current_dpi)
        
        return a4_img
        
    def get_optimal_dpi(self, image_path):
        """Calculate optimal DPI based on image size"""
        try:
            img = Image.open(image_path)
            width, height = img.size
            
            # Calculate image resolution
            pixels = width * height
            
            # Recommend DPI based on image size
            if pixels < 1_000_000:  # Less than 1MP
                return 150
            elif pixels < 4_000_000:  # Less than 4MP
                return 200
            elif pixels < 10_000_000:  # Less than 10MP
                return 300
            else:  # High resolution images
                return 600
        except:
            return 300  # Default
            
    def show_success_message(self, path, count):
        """Show success message after conversion"""
        if count == 1:
            message = f"PDF created successfully!\n\nSaved to:\n{path}"
        else:
            directory = os.path.dirname(path)
            message = f"{count} PDFs created successfully!\n\nSaved to:\n{directory}"
            
        messagebox.showinfo("Success", message)
        
    def update_button_states(self):
        """Update button states based on selected images"""
        if self.image_paths:
            self.convert_btn.config(state="normal")
            self.convert_a4_btn.config(state="normal")
            self.clear_btn.config(state="normal")
        else:
            self.convert_btn.config(state="disabled")
            self.convert_a4_btn.config(state="disabled")
            self.clear_btn.config(state="disabled")
            
    def show_progress(self):
        """Show progress bar"""
        self.progress_bar.grid()
        self.progress_bar.start(10)
        
    def hide_progress(self):
        """Hide progress bar"""
        self.progress_bar.stop()
        self.progress_bar.grid_remove()
        
    def update_status(self, message):
        """Update status bar message"""
        self.status_label.config(text=message)
        
    def on_frame_configure(self, event=None):
        """Update scroll region when frame size changes"""
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        
    def on_canvas_configure(self, event=None):
        """Update canvas window size when canvas size changes"""
        canvas_width = event.width if event else self.canvas.winfo_width()
        self.canvas.itemconfig(self.canvas_window, width=canvas_width)
        
    def on_main_frame_configure(self, event=None):
        """Update scroll region when main frame size changes"""
        self.main_canvas.configure(scrollregion=self.main_canvas.bbox("all"))
        
    def on_main_canvas_configure(self, event=None):
        """Update canvas window size when main canvas size changes"""
        canvas_width = event.width if event else self.main_canvas.winfo_width()
        self.main_canvas.itemconfig(self.main_canvas_window, width=canvas_width)
        
    def bind_mouse_wheel(self):
        """Bind mouse wheel events for scrolling"""
        # Windows and MacOS
        self.root.bind_all("<MouseWheel>", self._on_mousewheel)
        # Linux
        self.root.bind_all("<Button-4>", self._on_mousewheel)
        self.root.bind_all("<Button-5>", self._on_mousewheel)
        
    def _on_mousewheel(self, event):
        """Handle mouse wheel scrolling"""
        # Determine which canvas should scroll based on mouse position
        try:
            x, y = self.root.winfo_pointerxy()
            widget = self.root.winfo_containing(x, y)
            
            if widget is None:
                return
            
            # Check if mouse is over the image preview canvas or its children
            if widget == self.canvas or self._is_parent_of(widget, self.canvas):
                canvas = self.canvas
            else:
                canvas = self.main_canvas
            
            # Scroll the appropriate canvas
            if hasattr(event, 'delta') and event.delta:
                # Windows
                canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
            else:
                # Linux
                if event.num == 4:
                    canvas.yview_scroll(-1, "units")
                elif event.num == 5:
                    canvas.yview_scroll(1, "units")
        except Exception as e:
            # Fail silently if there's an issue with scrolling
            pass
            
    def _is_parent_of(self, widget, parent):
        """Check if widget is a child of parent"""
        try:
            while widget:
                if widget == parent:
                    return True
                widget = widget.master if hasattr(widget, 'master') else None
            return False
        except:
            return False
            
    def on_closing(self):
        """Clean up resources and close application"""
        try:
            # Unbind mouse wheel events
            self.root.unbind_all("<MouseWheel>")
            self.root.unbind_all("<Button-4>")
            self.root.unbind_all("<Button-5>")
            
            # Clear image references
            self.thumbnail_refs.clear()
            self.selected_images.clear()
            self.image_paths.clear()
            
            # Force garbage collection
            gc.collect()

            # Destroy window
            self.root.destroy()
            
        except Exception as e:
            print(f"Error during cleanup: {e}")
            self.root.destroy()


def main():
    """Main entry point of the application"""
    root = tk.Tk()
    
    # Set application icon (optional)
    try:
        root.iconbitmap(default='icon.ico')
    except:
        pass
    
    # Create application
    app = ImageToPDFConverter(root)
    
    # Start main loop
    root.mainloop()


if __name__ == "__main__":
    main()
