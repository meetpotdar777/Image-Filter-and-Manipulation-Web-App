import cv2
import numpy as np
from flask import Flask, Response, render_template_string, request, send_from_directory
import io
import base64
import os
import threading

app = Flask(__name__)

# Directory to store uploaded and processed images temporarily
UPLOAD_FOLDER = 'temp_images'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Global variables to hold the current original and processed image paths
# Using locks for thread-safe access in case of concurrent requests (though simple for this demo)
current_original_image_path = None
current_processed_image_path = None
image_path_lock = threading.Lock()

# Supported filters and their display names
FILTERS = {
    "original": "Original",
    "grayscale": "Grayscale",
    "blur": "Gaussian Blur",
    "edge_detection": "Edge Detection (Canny)",
    "brightness_contrast": "Brightness/Contrast",
    "invert": "Invert Colors"
}

def apply_filter(image_np, filter_name, brightness=0, contrast=1):
    """
    Applies the specified image processing filter to a NumPy array image.
    Args:
        image_np (numpy.ndarray): The input image as a NumPy array.
        filter_name (str): The name of the filter to apply.
        brightness (int): Brightness adjustment for 'brightness_contrast' filter.
        contrast (float): Contrast adjustment for 'brightness_contrast' filter.
    Returns:
        numpy.ndarray: The processed image.
    """
    processed_image = image_np.copy()

    if filter_name == "grayscale":
        if len(processed_image.shape) == 3 and processed_image.shape[2] == 3:
            processed_image = cv2.cvtColor(processed_image, cv2.COLOR_BGR2GRAY)
            # Convert grayscale back to BGR for consistent display (3 channels)
            processed_image = cv2.cvtColor(processed_image, cv2.COLOR_GRAY2BGR)
    elif filter_name == "blur":
        # Kernel size for blurring (must be odd)
        processed_image = cv2.GaussianBlur(processed_image, (15, 15), 0)
    elif filter_name == "edge_detection":
        # Convert to grayscale first for Canny
        gray_image = cv2.cvtColor(processed_image, cv2.COLOR_BGR2GRAY)
        # Apply Canny edge detector (thresholds can be adjusted)
        edges = cv2.Canny(gray_image, 100, 200)
        # Convert single-channel edges to 3-channel BGR for display
        processed_image = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    elif filter_name == "brightness_contrast":
        # Convert to float for calculations
        processed_image = processed_image.astype(np.float32) * contrast + brightness
        processed_image = np.clip(processed_image, 0, 255).astype(np.uint8)
    elif filter_name == "invert":
        processed_image = cv2.bitwise_not(processed_image)
    # If "original" or an unknown filter, return original image
    elif filter_name == "original" or filter_name not in FILTERS:
        pass # No processing needed, original image already copied
    
    return processed_image

# HTML template string for the frontend
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Image Processor 🖼️</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap" rel="stylesheet">
    <style>
        body { font-family: 'Inter', sans-serif; }
        .image-display-box {
            min-height: 250px; /* Minimum height for image display area */
            max-height: 500px; /* Maximum height for image display area */
            display: flex;
            align-items: center;
            justify-content: center;
            background-color: #f0f0f0;
            border: 2px dashed #cbd5e1; /* Gray dashed border */
            border-radius: 0.75rem;
            overflow: hidden; /* Hide overflow */
        }
        .image-display-box img {
            max-width: 100%;
            max-height: 100%;
            object-fit: contain; /* Ensure image scales within the box */
            border-radius: 0.5rem;
        }
        .toast-container-custom {
            position: fixed;
            bottom: 20px;
            right: 20px;
            z-index: 1060;
            display: flex;
            flex-direction: column;
            gap: 10px;
        }
        .toast-custom {
            background-color: #e0f7fa;
            border: 1px solid #00bcd4;
            color: #005f6b;
            padding: 15px 20px;
            border-radius: 0.5rem;
            box-shadow: 0 4px 12px rgba(0,0,0,0.1);
            opacity: 0;
            transform: translateY(20px);
            animation: fadeInOutToast 4s forwards;
        }
        .toast-custom.success {
            background-color: #d4edda;
            border: 1px solid #28a745;
            color: #155724;
        }
        .toast-custom.error {
            background-color: #f8d7da;
            border: 1px solid #dc3545;
            color: #721c24;
        }
        @keyframes fadeInOutToast {
            0% { opacity: 0; transform: translateY(20px); }
            10% { opacity: 1; transform: translateY(0); }
            90% { opacity: 1; transform: translateY(0); }
            100% { opacity: 0; transform: translateY(20px); }
        }
    </style>
</head>
<body class="bg-gradient-to-br from-indigo-100 to-purple-100 flex items-center justify-center min-h-screen p-4 sm:p-6 md:p-8">
    <div class="bg-white p-6 md:p-8 rounded-xl shadow-2xl max-w-xl w-full text-center border-t-8 border-indigo-600 transform hover:scale-105 transition-transform duration-300">
        <h1 class="text-3xl sm:text-4xl font-extrabold text-gray-800 mb-6 flex items-center justify-center gap-3">
            <span class="text-indigo-600 text-5xl">🖼️</span> Image Processor
        </h1>

        <form id="uploadForm" class="mb-6" enctype="multipart/form-data">
            <label for="imageUpload" class="block text-gray-700 text-sm font-bold mb-2">
                Upload Image:
            </label>
            <input type="file" id="imageUpload" name="image" accept="image/*" class="block w-full text-sm text-gray-500
                file:mr-4 file:py-2 file:px-4
                file:rounded-full file:border-0
                file:text-sm file:font-semibold
                file:bg-indigo-50 file:text-indigo-700
                hover:file:bg-indigo-100 cursor-pointer
            "/>
            <p class="mt-2 text-xs text-gray-500">Supported formats: JPG, PNG, BMP, etc.</p>
        </form>

        <div class="grid grid-cols-1 md:grid-cols-2 gap-6 mb-6">
            <div>
                <h2 class="text-xl font-semibold text-gray-700 mb-3">Original Image</h2>
                <div id="originalImageContainer" class="image-display-box border-dashed border-gray-400 bg-gray-50">
                    <p class="text-gray-500">No image uploaded yet.</p>
                </div>
            </div>
            <div>
                <h2 class="text-xl font-semibold text-gray-700 mb-3">Processed Image</h2>
                <div id="processedImageContainer" class="image-display-box border-dashed border-indigo-400 bg-indigo-50">
                    <p class="text-gray-500">Apply a filter to see result.</p>
                </div>
            </div>
        </div>

        <div class="flex flex-col sm:flex-row justify-center gap-4 mb-6">
            <label for="filterSelect" class="sr-only">Select Filter</label>
            <select id="filterSelect" class="form-select px-4 py-2 rounded-full shadow-sm text-gray-700 bg-white border border-gray-300 focus:outline-none focus:ring-2 focus:ring-indigo-500">
                <option value="original">Original</option>
                <option value="grayscale">Grayscale</option>
                <option value="blur">Gaussian Blur</option>
                <option value="edge_detection">Edge Detection (Canny)</option>
                <option value="brightness_contrast">Brightness/Contrast</option>
                <option value="invert">Invert Colors</option>
            </select>
            <button id="applyFilterBtn" class="bg-indigo-600 hover:bg-indigo-700 text-white font-bold py-2 px-4 rounded-full shadow-lg transition-all duration-300 transform hover:scale-105 active:scale-95 disabled:opacity-50 disabled:cursor-not-allowed">
                Apply Filter
            </button>
        </div>

        <div id="brightnessContrastControls" class="hidden flex flex-col items-center gap-3 mb-6 p-4 bg-gray-50 rounded-lg shadow-inner">
            <h3 class="text-lg font-medium text-gray-700">Brightness & Contrast Settings</h3>
            <div class="w-full max-w-xs">
                <label for="brightnessSlider" class="block text-sm font-medium text-gray-700">Brightness: <span id="brightnessValue">0</span></label>
                <input type="range" id="brightnessSlider" min="-100" max="100" value="0" class="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer range-lg">
            </div>
            <div class="w-full max-w-xs">
                <label for="contrastSlider" class="block text-sm font-medium text-gray-700">Contrast: <span id="contrastValue">1.0</span></label>
                <input type="range" id="contrastSlider" min="0" max="300" value="100" class="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer range-lg">
            </div>
            <p class="text-xs text-gray-500">Adjust brightness (-100 to 100) and contrast (0.0 to 3.0).</p>
        </div>

        <p class="text-sm text-gray-600 mt-6 leading-relaxed">
            Upload an image, select a filter, and see the magic happen! The processing is done on the server-side using Python and OpenCV.
        </p>
    </div>

    <div id="toast-container" class="toast-container-custom"></div>

    <script>
        const imageUpload = document.getElementById('imageUpload');
        const originalImageContainer = document.getElementById('originalImageContainer');
        const processedImageContainer = document.getElementById('processedImageContainer');
        const filterSelect = document.getElementById('filterSelect');
        const applyFilterBtn = document.getElementById('applyFilterBtn');
        const toastContainer = document.getElementById('toast-container');
        const brightnessContrastControls = document.getElementById('brightnessContrastControls');
        const brightnessSlider = document.getElementById('brightnessSlider');
        const contrastSlider = document.getElementById('contrastSlider');
        const brightnessValueSpan = document.getElementById('brightnessValue');
        const contrastValueSpan = document.getElementById('contrastValue');

        let currentImageBase64 = null; // Holds the base64 of the original image after upload

        function showToast(message, type = 'info') {
            const toast = document.createElement('div');
            toast.className = `toast-custom ${type}`;
            toast.innerHTML = message;
            toastContainer.appendChild(toast);
            setTimeout(() => {
                toast.remove();
            }, 4000);
        }

        async function processImage(filterName, brightness = 0, contrast = 1) {
            if (!currentImageBase64) {
                showToast('Please upload an image first!', 'error');
                return;
            }

            applyFilterBtn.disabled = true;
            applyFilterBtn.textContent = 'Processing...';

            const payload = {
                image_data: currentImageBase64,
                filter: filterName,
                brightness: brightness,
                contrast: contrast
            };

            try {
                const response = await fetch('/process_image', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify(payload)
                });

                if (!response.ok) {
                    const errorText = await response.text();
                    throw new Error(`Server error: ${response.status} - ${errorText}`);
                }

                const data = await response.json();
                if (data.processed_image) {
                    processedImageContainer.innerHTML = `<img src="data:image/jpeg;base64,${data.processed_image}" alt="Processed Image">`;
                    showToast('Filter applied successfully!', 'success');
                } else {
                    showToast('Failed to get processed image data.', 'error');
                }
            } catch (error) {
                console.error('Error processing image:', error);
                showToast(`Error: ${error.message || 'Image processing failed.'}`, 'error');
            } finally {
                applyFilterBtn.disabled = false;
                applyFilterBtn.textContent = 'Apply Filter';
            }
        }

        imageUpload.addEventListener('change', function(event) {
            const file = event.target.files[0];
            if (file) {
                const reader = new FileReader();
                reader.onload = function(e) {
                    currentImageBase64 = e.target.result.split(',')[1]; // Get base64 part
                    originalImageContainer.innerHTML = `<img src="${e.target.result}" alt="Original Image">`;
                    // Reset processed image and filter selection
                    processedImageContainer.innerHTML = `<p class="text-gray-500">Apply a filter to see result.</p>`;
                    filterSelect.value = "original";
                    brightnessSlider.value = 0;
                    contrastSlider.value = 100;
                    brightnessValueSpan.textContent = 0;
                    contrastValueSpan.textContent = 1.0;
                    brightnessContrastControls.classList.add('hidden');
                    showToast('Image uploaded successfully!', 'success');
                };
                reader.onerror = function() {
                    showToast('Error reading file!', 'error');
                };
                reader.readAsDataURL(file);
            } else {
                currentImageBase64 = null;
                originalImageContainer.innerHTML = `<p class="text-gray-500">No image uploaded yet.</p>`;
                processedImageContainer.innerHTML = `<p class="text-gray-500">Apply a filter to see result.</p>`;
                filterSelect.value = "original";
                brightnessContrastControls.classList.add('hidden');
            }
        });

        filterSelect.addEventListener('change', function() {
            if (filterSelect.value === 'brightness_contrast') {
                brightnessContrastControls.classList.remove('hidden');
            } else {
                brightnessContrastControls.classList.add('hidden');
            }
            // Automatically apply filter on selection change if an image is loaded
            if (currentImageBase64) {
                 triggerImageProcessing();
            }
        });

        brightnessSlider.addEventListener('input', () => {
            brightnessValueSpan.textContent = brightnessSlider.value;
            if (currentImageBase64 && filterSelect.value === 'brightness_contrast') {
                triggerImageProcessing();
            }
        });

        contrastSlider.addEventListener('input', () => {
            const contrastVal = (contrastSlider.value / 100).toFixed(1);
            contrastValueSpan.textContent = contrastVal;
            if (currentImageBase64 && filterSelect.value === 'brightness_contrast') {
                triggerImageProcessing();
            }
        });

        applyFilterBtn.addEventListener('click', function() {
            triggerImageProcessing();
        });

        function triggerImageProcessing() {
            const selectedFilter = filterSelect.value;
            const brightness = parseInt(brightnessSlider.value);
            const contrast = parseFloat((contrastSlider.value / 100).toFixed(1));
            processImage(selectedFilter, brightness, contrast);
        }
    </script>
</body>
</html>
"""

@app.route('/')
def index():
    """
    Renders the main HTML page for the image processing app.
    """
    return render_template_string(HTML_TEMPLATE)

@app.route('/process_image', methods=['POST'])
def process_image_route():
    """
    Receives image data, applies the selected filter, and returns the processed image.
    """
    data = request.get_json()
    image_data = data['image_data']
    filter_name = data['filter']
    brightness = int(data.get('brightness', 0))
    contrast = float(data.get('contrast', 1.0))

    try:
        # Decode base64 image data
        nparr = np.frombuffer(base64.b64decode(image_data), np.uint8)
        img_np = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img_np is None:
            return "Could not decode image.", 400

        # Apply filter
        processed_img_np = apply_filter(img_np, filter_name, brightness, contrast)

        # Encode processed image back to JPEG for web display
        _, buffer = cv2.imencode('.jpeg', processed_img_np)
        processed_image_base64 = base64.b64encode(buffer).decode('utf-8')

        return {'processed_image': processed_image_base64}

    except Exception as e:
        print(f"Error during image processing: {e}")
        return f"Image processing failed: {e}", 500

if __name__ == '__main__':
    print("Image Processing Web Application Starting...")
    print("---------------------------------------")
    print("Flask server will start on: http://127.0.0.1:5000")
    print("Open this URL in your web browser to upload and process images.")
    print("Press Ctrl+C in this terminal to stop the server.")
    print("---------------------------------------")
    app.run(host='0.0.0.0', port=5000, debug=False)
