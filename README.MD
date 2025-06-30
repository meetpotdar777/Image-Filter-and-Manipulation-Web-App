Advanced Image Processor 🖼️✨

This project is a web-based image processing application built with Flask (Python) and OpenCV for backend image manipulation, and a modern HTML, CSS (Tailwind CSS), and JavaScript frontend for a rich user experience. It allows users to upload images, apply various filters and effects, and download the processed results.

Features 🚀

Image Upload: Easily upload images in common formats (JPG, PNG, BMP).

Real-time Previews: See the original and processed images side-by-side.

Extensive Filter Library: Apply a wide range of image filters including:

Grayscale, Gaussian Blur, Edge Detection (Canny)

Brightness/Contrast, Invert Colors, Sepia Tone, Sharpen

Global Threshold (Binarize), Adaptive Threshold

Rotate (90°, 180°, 270°), Flip (Horizontal, Vertical)

Noise Reduction (Median Blur), Resize

RGB Channel Adjustment, HSL Color Adjustment, Color Temperature

Exposure/Gamma Correction, Dithering (Monochrome)

Selective Color Adjustment, Color Splash Effect

Photo Filters: Warm, Cool, Vintage, B&W Red Tint

Text Watermarking (custom text, size, color, position)

Image Overlay / Blending (upload a second image to blend)

Interactive Cropping (select an area on the original image to crop)

Histogram Equalization, Vignette Effect

Cartoonify, Pencil Sketch

Dynamic Controls: Filters with adjustable parameters (e.g., brightness, blur radius, text) feature interactive sliders and inputs that appear dynamically.

Interactive Cropping: A user-friendly interface to select and crop a specific area of the image directly in the browser.

Download Processed Image: Download the final processed image.

Responsive UI: Designed to be user-friendly on both desktop and mobile devices.

Toast Notifications: Provides informative feedback for actions like uploads, processing, and downloads.

Technologies Used 💻

Python (Flask): Powers the backend server, handling image uploads and processing requests.

OpenCV (cv2): The core image processing library used in the Python backend for applying all filters and transformations.

HTML5, CSS3, JavaScript: Form the interactive and responsive frontend.

Tailwind CSS: Used for rapid and efficient styling of the web interface.

Base64 Encoding: Images are sent between the frontend and backend as Base64 encoded strings.

NumPy: Used by OpenCV for efficient array manipulation of image data.

How It Works 🤔

Image Upload: When a user uploads an image via the web interface, it's read by JavaScript, converted to a Base64 string, and stored temporarily in the frontend.

Server-side Processing: Upon selecting a filter or adjusting parameters, the Base64 image data and filter parameters are sent to the Flask backend via an AJAX request.

OpenCV Magic: The Flask server decodes the Base64 image into an OpenCV (NumPy) array, applies the chosen filter using cv2 functions, and then re-encodes the processed image back to Base64.

Display & Download: The processed Base64 image is sent back to the frontend, displayed to the user, and made available for download.

Temporary Storage 📂

The application includes a temp_images directory to temporarily store uploaded and processed images on the server-side during operation. This directory is created if it doesn't exist. Images are processed in memory and not persistently stored after the request is completed.

Getting Started 🏁

Follow these steps to set up and run the application on your local machine.

Prerequisites ✅

Python 3.x 🐍

pip (Python package installer)

Installation ⬇️

Save the Python Code:

Save the provided Python code as image_processor_version_7.0.py (or any other .py file name you prefer).

Install Required Python Libraries:

Open your terminal or command prompt and navigate to the directory where you saved the Python file. Then, install the necessary libraries using pip:

pip install Flask opencv-python numpy

Running the Application ▶️

Start the Flask Server:

In your terminal or command prompt, from the same directory where you saved the Python file, run:

python image_processor_version_7.0.py

You should see output indicating that the Flask development server is running, typically on http://127.0.0.1:5000/ or http://localhost:5000/.

Open in Browser:

Open your web browser and navigate to the address provided by the Flask server (e.g., http://127.0.0.1:5000/).

You can now upload images, apply filters, and interact with the application!