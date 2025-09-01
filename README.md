
# Image to PDF Converter 📷➡️📄

A simple, user-friendly desktop application that converts your images into PDF files. Perfect for creating PDF documents from photos, scanned documents, or any image files on your computer.

**Tech Used:** Python, Tkinter, Pillow


## 🌟 Features

✅ Convert multiple images to PDF with just a few clicks

✅ Create a single PDF from multiple images OR individual PDFs for each image

✅ Preview selected images before conversion

✅ Choose custom filename and save location

✅ Supports popular image formats (JPG, PNG, BMP, GIF, TIFF, WebP)

✅ Clean, intuitive interface that works on any screen size

✅ No internet connection required - works completely offline

## 📥 Installation
 
Run from Source Code  
If the above doesn't work or isn't available, follow these steps:  

**Step 1:** Install Python  

Visit python.org and Download Python (version 3.7 or newer)  
-Run the installer  
⚠️ Important: Check the box that says "Add Python to PATH"  
-Click "Install Now"  

**Step 2:** Download the Program

Click the green "Code" button above  
Select "Download ZIP"  
Extract the ZIP file to your desired location (e.g., Desktop)  

**Step 3:** Install Required Component

-Open Command Prompt (Windows) or Terminal (Mac/Linux)  
  --Windows: Press Win + R, type cmd, press Enter  
  --Mac: Press Cmd + Space, type terminal, press Enter  
-Type this command and press Enter: ```pip install Pillow```  
-Wait for installation to complete  

**Step 4:** Run the Program

-Navigate to where you extracted the files  
-Double-click on image_to_pdf_converter.py  
-If that doesn't work, right-click → "Open with" → "Python"  
OR  
open terminal in the same folder as the program and use ```python image_to_pdf_converter.py``` 

## 🚀 How to Use
Step-by-Step Guide:  
Start the Program

Double-click the application icon or Python file
Select Your Images

Click the "📁 Select Images" button
Browse to your images location
Select one or more images (hold Ctrl/Cmd to select multiple)
Click "Open"
Choose Conversion Type

Single PDF: Combines all images into one PDF file
Individual PDFs: Creates separate PDF for each image
Set Output Options

Filename: Enter your desired filename (without .pdf)
Save Location: Click "Browse..." to choose where to save
Convert

Click "📄 Convert to PDF"
Wait for the success message
Find your PDF(s) in the chosen location


## 💡 Tips & Tricks
Image Order: Images appear in the PDF in the order they're displayed in the preview

File Naming: When creating individual PDFs, each file gets a number prefix (e.g., document_1_photo.pdf)

Large Images: The program automatically handles large images - no need to resize first

Batch Processing: Select hundreds of images at once - the program can handle it!

## ❓ Troubleshooting
**"The program won't start"**  
-Make sure Python is installed correctly  
-Try running from Command Prompt: python image_to_pdf_converter.py  

**"I can't select images"**  
-Make sure your images are in a supported format (JPG, PNG, etc.)  
-Check that the image files aren't corrupted  

**"Convert button is disabled"**  
-You need to select at least one image first  
-Click "Select Images" and choose your files  

**"PDF creation failed"**  
-Check you have permission to save in the selected folder  
-Make sure there's enough disk space  
-Try a different save location  

## 🔧 For Developers  
Best to use a Python virtual environment.  
Requirements:  
-Python 3.7+  
-Pillow (PIL) library  

-Installation:    
```git clone https://github.com/sameerparvezsingh/image-to-pdf-converter.git```  
```cd image-to-pdf-converter```  
```pip install -r requirements.txt```  
```python image_to_pdf_converter.py```  

-Building Executable:    
```pip install pyinstaller```  
```pyinstaller --onefile --windowed image_to_pdf_converter.py```  


## 📝 License
This project is licensed under the MIT License - see the LICENSE file for details.


## 🤝 Contributing
Contributions are welcome! Feel free to:


## Report bugs
Suggest new features  
Submit pull requests
