import cv2
import numpy as np
from flask import Flask, Response, render_template_string, request, send_from_directory, jsonify
import io
import base64
import os
import threading

app = Flask(__name__)

# Directory to store uploaded and processed images temporarily
UPLOAD_FOLDER = 'temp_images'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Global variables to hold the current original and processed image paths (no longer strictly used for serving, but kept for context)
current_original_image_path = None
current_processed_image_path = None
image_path_lock = threading.Lock() # Mutex to protect access to global image data if needed

# Supported filters and their display names
FILTERS = {
    "original": "Original",
    "grayscale": "Grayscale",
    "blur": "Gaussian Blur",
    "edge_detection": "Edge Detection (Canny)",
    "brightness_contrast": "Brightness/Contrast",
    "invert": "Invert Colors",
    "sepia": "Sepia Tone", # New filter
    "sharpen": "Sharpen", # New filter
    "threshold": "Threshold (Binarize)", # New filter
    "rotate_90": "Rotate 90° CW", # New filter
    "rotate_180": "Rotate 180°", # New filter
    "rotate_270": "Rotate 270° CW", # New filter
    "flip_horizontal": "Flip Horizontal", # New filter
    "flip_vertical": "Flip Vertical" # New filter
}

def apply_filter(image_np, filter_name, brightness=0, contrast=1, sharpen_amount=0, threshold_val=127, rotation_angle=0, flip_code=None):
    """
    Applies the specified image processing filter to a NumPy array image.
    Args:
        image_np (numpy.ndarray): The input image as a NumPy array.
        filter_name (str): The name of the filter to apply.
        brightness (int): Brightness adjustment for 'brightness_contrast' filter.
        contrast (float): Contrast adjustment for 'brightness_contrast' filter.
        sharpen_amount (float): Amount of sharpening for 'sharpen' filter.
        threshold_val (int): Threshold value for 'threshold' filter.
        rotation_angle (int): Angle for rotation (0, 90, 180, 270).
        flip_code (int): 0 for vertical, 1 for horizontal, -1 for both.
    Returns:
        numpy.ndarray: The processed image.
    """
    processed_image = image_np.copy()

    if filter_name == "grayscale":
        if len(processed_image.shape) == 3 and processed_image.shape[2] == 3:
            processed_image = cv2.cvtColor(processed_image, cv2.COLOR_BGR2GRAY)
            processed_image = cv2.cvtColor(processed_image, cv2.COLOR_GRAY2BGR) # Convert back to 3 channels for consistent display
    elif filter_name == "blur":
        processed_image = cv2.GaussianBlur(processed_image, (15, 15), 0)
    elif filter_name == "edge_detection":
        gray_image = cv2.cvtColor(processed_image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray_image, 100, 200)
        processed_image = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    elif filter_name == "brightness_contrast":
        processed_image = processed_image.astype(np.float32) * contrast + brightness
        processed_image = np.clip(processed_image, 0, 255).astype(np.uint8)
    elif filter_name == "invert":
        processed_image = cv2.bitwise_not(processed_image)
    elif filter_name == "sepia":
        # Sepia transformation matrix (simplified)
        kernel = np.array([[0.272, 0.534, 0.131],
                           [0.349, 0.686, 0.168],
                           [0.393, 0.769, 0.189]])
        # Apply transformation to each pixel (B,G,R -> B',G',R')
        processed_image = cv2.transform(processed_image, kernel.T) # .T for transpose
        processed_image = np.clip(processed_image, 0, 255).astype(np.uint8)
    elif filter_name == "sharpen":
        # Sharpening kernel
        # Using a fixed kernel, strength scaled by sharpen_amount
        kernel = np.array([[-1, -1, -1],
                           [-1,  9, -1],
                           [-1, -1, -1]])
        sharpened = cv2.filter2D(processed_image, -1, kernel * (sharpen_amount / 100.0))
        processed_image = np.clip(sharpened, 0, 255).astype(np.uint8)
    elif filter_name == "threshold":
        gray_image = cv2.cvtColor(processed_image, cv2.COLOR_BGR2GRAY)
        # Apply binary threshold. Pixels > threshold_val become 255, others 0.
        _, thresholded = cv2.threshold(gray_image, threshold_val, 255, cv2.THRESH_BINARY)
        processed_image = cv2.cvtColor(thresholded, cv2.COLOR_GRAY2BGR) # Convert back to 3 channels
    elif filter_name.startswith("rotate_"):
        if rotation_angle == 90:
            processed_image = cv2.rotate(processed_image, cv2.ROTATE_90_CLOCKWISE)
        elif rotation_angle == 180:
            processed_image = cv2.rotate(processed_image, cv2.ROTATE_180)
        elif rotation_angle == 270:
            processed_image = cv2.rotate(processed_image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    elif filter_name.startswith("flip_"):
        if flip_code == 0: # vertical
            processed_image = cv2.flip(processed_image, 0)
        elif flip_code == 1: # horizontal
            processed_image = cv2.flip(processed_image, 1)
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
    <title>Advanced Image Processor 🖼️</title>
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
            position: relative; /* For loading overlay */
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
        .loading-overlay {
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background: rgba(255, 255, 255, 0.8);
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 1.2rem;
            color: #4a5568;
            font-weight: 600;
            border-radius: 0.75rem;
            z-index: 10;
        }
        .spinner {
            border: 4px solid rgba(0, 0, 0, 0.1);
            border-left: 4px solid #3b82f6; /* Blue spinner */
            border-radius: 50%;
            width: 30px;
            height: 30px;
            animation: spin 1s linear infinite;
            margin-right: 10px;
        }
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
    </style>
</head>
<body class="bg-gradient-to-br from-indigo-100 to-purple-100 flex items-center justify-center min-h-screen p-4 sm:p-6 md:p-8">
    <div class="bg-white p-6 md:p-8 rounded-xl shadow-2xl max-w-2xl w-full text-center border-t-8 border-indigo-600 transform hover:scale-105 transition-transform duration-300">
        <h1 class="text-3xl sm:text-4xl font-extrabold text-gray-800 mb-6 flex items-center justify-center gap-3">
            <span class="text-indigo-600 text-5xl">🖼️</span> Advanced Image Processor
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
                    <p class="text-gray-500">Upload an image to get started!</p>
                </div>
            </div>
            <div>
                <h2 class="text-xl font-semibold text-gray-700 mb-3">Processed Image</h2>
                <div id="processedImageContainer" class="image-display-box border-dashed border-indigo-400 bg-indigo-50">
                    <p class="text-gray-500">Processed image will appear here.</p>
                    <div id="processingOverlay" class="loading-overlay hidden">
                        <div class="spinner"></div> Processing...
                    </div>
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
                <option value="sepia">Sepia Tone</option>
                <option value="sharpen">Sharpen</option>
                <option value="threshold">Threshold (Binarize)</option>
                <option value="rotate_90">Rotate 90° CW</option>
                <option value="rotate_180">Rotate 180°</option>
                <option value="rotate_270">Rotate 270° CW</option>
                <option value="flip_horizontal">Flip Horizontal</option>
                <option value="flip_vertical">Flip Vertical</option>
            </select>
            <button id="applyFilterBtn" class="bg-indigo-600 hover:bg-indigo-700 text-white font-bold py-2 px-4 rounded-full shadow-lg transition-all duration-300 transform hover:scale-105 active:scale-95 disabled:opacity-50 disabled:cursor-not-allowed">
                Apply Filter
            </button>
        </div>

        <!-- Dynamic Controls Container -->
        <div id="dynamicControls" class="flex flex-col items-center gap-3 mb-6 p-4 bg-gray-50 rounded-lg shadow-inner hidden">
            <!-- Content will be injected here by JS -->
        </div>

        <div class="flex flex-col sm:flex-row justify-center gap-4 mt-6">
            <button id="downloadBtn" class="bg-green-500 hover:bg-green-700 text-white font-bold py-2 px-4 rounded-full shadow-lg transition-all duration-300 transform hover:scale-105 active:scale-95 disabled:opacity-50 disabled:cursor-not-allowed">
                ⬇️ Download Processed
            </button>
            <button id="resetBtn" class="bg-red-500 hover:bg-red-700 text-white font-bold py-2 px-4 rounded-full shadow-lg transition-all duration-300 transform hover:scale-105 active:scale-95">
                🔄 Reset App
            </button>
        </div>

        <p class="text-sm text-gray-600 mt-6 leading-relaxed">
            Upload an image, select a filter, and fine-tune parameters using the dynamic controls. The processing is done server-side using Python and OpenCV.
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
        const dynamicControls = document.getElementById('dynamicControls');
        const processingOverlay = document.getElementById('processingOverlay');
        const downloadBtn = document.getElementById('downloadBtn');
        const resetBtn = document.getElementById('resetBtn');

        let currentImageBase64 = null; // Holds the base64 of the original image after upload
        let currentProcessedImageBase64 = null; // Holds the base64 of the last processed image

        // Helper to display toast messages
        function showToast(message, type = 'info') {
            const toast = document.createElement('div');
            toast.className = `toast-custom ${type}`;
            toast.innerHTML = message;
            toastContainer.appendChild(toast);
            setTimeout(() => {
                toast.remove();
            }, 4000);
        }

        // Function to render dynamic controls based on selected filter
        function renderDynamicControls(filterName) {
            dynamicControls.innerHTML = ''; // Clear previous controls
            dynamicControls.classList.add('hidden'); // Hide by default

            if (filterName === 'brightness_contrast') {
                dynamicControls.classList.remove('hidden');
                dynamicControls.innerHTML = `
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
                `;
                const brightnessSlider = document.getElementById('brightnessSlider');
                const contrastSlider = document.getElementById('contrastSlider');
                const brightnessValueSpan = document.getElementById('brightnessValue');
                const contrastValueSpan = document.getElementById('contrastValue');

                brightnessSlider.addEventListener('input', () => {
                    brightnessValueSpan.textContent = brightnessSlider.value;
                    triggerImageProcessing();
                });
                contrastSlider.addEventListener('input', () => {
                    const contrastVal = (contrastSlider.value / 100).toFixed(1);
                    contrastValueSpan.textContent = contrastVal;
                    triggerImageProcessing();
                });
            } else if (filterName === 'sharpen') {
                dynamicControls.classList.remove('hidden');
                dynamicControls.innerHTML = `
                    <h3 class="text-lg font-medium text-gray-700">Sharpen Settings</h3>
                    <div class="w-full max-w-xs">
                        <label for="sharpenAmountSlider" class="block text-sm font-medium text-gray-700">Amount: <span id="sharpenAmountValue">0</span></label>
                        <input type="range" id="sharpenAmountSlider" min="0" max="200" value="0" class="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer range-lg">
                    </div>
                    <p class="text-xs text-gray-500">Adjust sharpening strength (0 to 200).</p>
                `;
                const sharpenAmountSlider = document.getElementById('sharpenAmountSlider');
                const sharpenAmountValueSpan = document.getElementById('sharpenAmountValue');
                sharpenAmountSlider.addEventListener('input', () => {
                    sharpenAmountValueSpan.textContent = sharpenAmountSlider.value;
                    triggerImageProcessing();
                });
            } else if (filterName === 'threshold') {
                dynamicControls.classList.remove('hidden');
                dynamicControls.innerHTML = `
                    <h3 class="text-lg font-medium text-gray-700">Threshold Settings</h3>
                    <div class="w-full max-w-xs">
                        <label for="thresholdSlider" class="block text-sm font-medium text-gray-700">Threshold Value: <span id="thresholdValue">127</span></label>
                        <input type="range" id="thresholdSlider" min="0" max="255" value="127" class="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer range-lg">
                    </div>
                    <p class="text-xs text-gray-500">Set the pixel intensity threshold (0-255) for binarization.</p>
                `;
                const thresholdSlider = document.getElementById('thresholdSlider');
                const thresholdValueSpan = document.getElementById('thresholdValue');
                thresholdSlider.addEventListener('input', () => {
                    thresholdValueSpan.textContent = thresholdSlider.value;
                    triggerImageProcessing();
                });
            }
        }

        async function processImage(filterName, brightness = 0, contrast = 1, sharpenAmount = 0, thresholdVal = 127, rotationAngle = 0, flipCode = null) {
            if (!currentImageBase64) {
                showToast('Please upload an image first!', 'error');
                return;
            }

            processingOverlay.classList.remove('hidden'); // Show loading overlay
            applyFilterBtn.disabled = true;
            applyFilterBtn.innerHTML = '<div class="spinner inline-block align-middle mr-2"></div> Processing...';

            const controller = new AbortController();
            const signal = controller.signal;
            const timeoutId = setTimeout(() => controller.abort(), 30000); // 30 second timeout

            const payload = {
                image_data: currentImageBase64, // CORRECTED TYPO HERE
                filter: filterName,
                brightness: brightness,
                contrast: contrast,
                sharpen_amount: sharpenAmount,
                threshold_val: thresholdVal,
                rotation_angle: rotationAngle,
                flip_code: flipCode
            };

            try {
                const response = await fetch('/process_image', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify(payload),
                    signal: signal // Pass the abort signal
                });

                clearTimeout(timeoutId); // Clear timeout if fetch completes

                if (!response.ok) {
                    const errorText = await response.text();
                    throw new Error(`Server error: ${response.status} - ${errorText}`);
                }

                const data = await response.json();
                if (data.processed_image) {
                    currentProcessedImageBase64 = data.processed_image;
                    processedImageContainer.innerHTML = `<img src="data:image/jpeg;base64,${data.processed_image}" alt="Processed Image">`;
                    downloadBtn.disabled = false; // Enable download button
                    showToast('Filter applied successfully! ✨', 'success');
                } else {
                    showToast('Failed to get processed image data.', 'error');
                }
            } catch (error) {
                if (error.name === 'AbortError') {
                    console.error('Image processing timed out:', error);
                    showToast('Processing timed out. The image might be too large or the server is slow.', 'error');
                } else {
                    console.error('Error processing image:', error);
                    showToast(`Error: ${error.message || 'Image processing failed.'}`, 'error');
                }
                processedImageContainer.innerHTML = `<p class="text-gray-500">Processing failed. Try again.</p>`;
                downloadBtn.disabled = true; // Disable download button on error
            } finally {
                processingOverlay.classList.add('hidden'); // Hide loading overlay
                applyFilterBtn.disabled = false;
                applyFilterBtn.innerHTML = 'Apply Filter';
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
                    processedImageContainer.innerHTML = `<p class="text-gray-500">Processed image will appear here.</p>`;
                    filterSelect.value = "original";
                    renderDynamicControls("original"); // Re-render controls for original
                    downloadBtn.disabled = true; // Disable download until processed
                    showToast('Image uploaded successfully!', 'success');
                    triggerImageProcessing(); // Apply 'original' filter by default on upload
                };
                reader.onerror = function() {
                    showToast('Error reading file!', 'error');
                };
                reader.readAsDataURL(file);
            } else {
                currentImageBase64 = null;
                currentProcessedImageBase64 = null;
                originalImageContainer.innerHTML = `<p class="text-gray-500">Upload an image to get started!</p>`;
                processedImageContainer.innerHTML = `<p class="text-gray-500">Processed image will appear here.</p>`;
                filterSelect.value = "original";
                renderDynamicControls("original");
                downloadBtn.disabled = true;
            }
        });

        filterSelect.addEventListener('change', function() {
            renderDynamicControls(filterSelect.value); // Update controls based on selection
            if (currentImageBase64) {
                 triggerImageProcessing(); // Apply filter automatically if image exists
            }
        });

        applyFilterBtn.addEventListener('click', function() {
            triggerImageProcessing();
        });

        downloadBtn.addEventListener('click', () => {
            if (currentProcessedImageBase64) {
                const a = document.createElement('a');
                a.href = `data:image/jpeg;base64,${currentProcessedImageBase64}`;
                const timestamp = new Date().toISOString().replace(/[:.]/g, '-').slice(0, 19);
                a.download = `processed-image-${filterSelect.value}-${timestamp}.jpeg`;
                document.body.appendChild(a);
                a.click();
                document.body.removeChild(a);
                showToast('Processed image downloaded!', 'success');
            } else {
                showToast('No processed image to download!', 'error');
            }
        });

        resetBtn.addEventListener('click', () => {
            imageUpload.value = ''; // Clear file input
            currentImageBase64 = null;
            currentProcessedImageBase64 = null;
            originalImageContainer.innerHTML = `<p class="text-gray-500">Upload an image to get started!</p>`;
            processedImageContainer.innerHTML = `<p class="text-gray-500">Processed image will appear here.</p>`;
            filterSelect.value = "original";
            renderDynamicControls("original"); // Reset dynamic controls
            downloadBtn.disabled = true;
            showToast('Application reset. Ready for new image!', 'info');
        });


        // Central function to gather parameters and call processImage
        function triggerImageProcessing() {
            const selectedFilter = filterSelect.value;
            let brightness = 0;
            let contrast = 1.0;
            let sharpenAmount = 0;
            let thresholdVal = 127;
            let rotationAngle = 0;
            let flipCode = null;

            if (selectedFilter === 'brightness_contrast') {
                brightness = parseInt(document.getElementById('brightnessSlider').value);
                contrast = parseFloat((document.getElementById('contrastSlider').value / 100).toFixed(1));
            } else if (selectedFilter === 'sharpen') {
                sharpenAmount = parseInt(document.getElementById('sharpenAmountSlider').value);
            } else if (selectedFilter === 'threshold') {
                thresholdVal = parseInt(document.getElementById('thresholdSlider').value);
            } else if (selectedFilter === 'rotate_90') {
                rotationAngle = 90;
            } else if (selectedFilter === 'rotate_180') {
                rotationAngle = 180;
            } else if (selectedFilter === 'rotate_270') {
                rotationAngle = 270;
            } else if (selectedFilter === 'flip_horizontal') {
                flipCode = 1; // 1 for horizontal flip
            } else if (selectedFilter === 'flip_vertical') {
                flipCode = 0; // 0 for vertical flip
            }

            processImage(selectedFilter, brightness, contrast, sharpenAmount, thresholdVal, rotationAngle, flipCode);
        }

        // Initialize dynamic controls on page load
        document.addEventListener('DOMContentLoaded', () => {
            renderDynamicControls(filterSelect.value);
            downloadBtn.disabled = true; // Initially disable download button
        });
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
    
    # Extract parameters, providing defaults
    brightness = int(data.get('brightness', 0))
    contrast = float(data.get('contrast', 1.0))
    sharpen_amount = float(data.get('sharpen_amount', 0))
    threshold_val = int(data.get('threshold_val', 127))
    rotation_angle = int(data.get('rotation_angle', 0))
    flip_code = data.get('flip_code', None)
    if flip_code is not None:
        flip_code = int(flip_code) # Ensure flip_code is an integer if present

    try:
        # Decode base64 image data
        nparr = np.frombuffer(base64.b64decode(image_data), np.uint8)
        img_np = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img_np is None:
            # Added more specific error logging for debugging image decoding issues
            print("ERROR: cv2.imdecode returned None. Image data might be corrupt or unsupported format.")
            return "Could not decode image. Please ensure it's a valid image file.", 400

        # Apply filter with all relevant parameters
        processed_img_np = apply_filter(img_np, filter_name,
                                        brightness=brightness,
                                        contrast=contrast,
                                        sharpen_amount=sharpen_amount,
                                        threshold_val=threshold_val,
                                        rotation_angle=rotation_angle,
                                        flip_code=flip_code)

        # Encode processed image back to JPEG for web display
        # Use a higher quality (e.g., 90) for better results, if not a performance bottleneck
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 90]
        _, buffer = cv2.imencode('.jpeg', processed_img_np, encode_param)
        processed_image_base64 = base64.b64encode(buffer).decode('utf-8')

        return jsonify({'processed_image': processed_image_base64})

    except Exception as e:
        # Improved error logging to catch and print specific exceptions
        import traceback
        print(f"AN UNEXPECTED ERROR OCCURRED DURING IMAGE PROCESSING: {e}")
        print(traceback.format_exc()) # Print full traceback for detailed debugging
        return f"Image processing failed on server: {e}", 500

if __name__ == '__main__':
    print("Image Processing Web Application Starting...")
    print("---------------------------------------")
    print("Flask server will start on: http://127.0.0.1:5000")
    print("Open this URL in your web browser to upload and process images.")
    print("Press Ctrl+C in this terminal to stop the server.")
    print("---------------------------------------")
    app.run(host='0.0.0.0', port=5000, debug=False)
