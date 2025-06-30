import cv2
import numpy as np
from flask import Flask, Response, render_template_string, request, jsonify
import io
import base64
import os
import threading

app = Flask(__name__)

# Directory to store uploaded and processed images temporarily
UPLOAD_FOLDER = 'temp_images'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Global variables to hold the current original image data (raw NumPy array)
# This will be updated on image upload.
current_original_img_np = None
image_data_lock = threading.Lock()

# Supported filters and their display names
FILTERS = {
    "original": "Original",
    "grayscale": "Grayscale",
    "blur": "Gaussian Blur", # Now adjustable
    "edge_detection": "Edge Detection (Canny)",
    "brightness_contrast": "Brightness/Contrast",
    "invert": "Invert Colors",
    "sepia": "Sepia Tone",
    "sharpen": "Sharpen",
    "threshold": "Global Threshold (Binarize)",
    "adaptive_threshold": "Adaptive Threshold", # New filter
    "rotate_90": "Rotate 90° CW",
    "rotate_180": "Rotate 180°",
    "rotate_270": "Rotate 270° CW",
    "flip_horizontal": "Flip Horizontal",
    "flip_vertical": "Flip Vertical",
    "noise_reduction": "Noise Reduction (Median)",
    "resize": "Resize Image",
    "color_adjust": "RGB Channel Adjust",
    "hsl_adjust": "HSL Color Adjust", # New filter
    "color_temperature": "Color Temperature", # New filter
    "exposure_gamma": "Exposure/Gamma Correction", # New filter
    "dithering": "Dithering (Monochrome)", # New filter
    "watermark_text": "Text Watermark",
    "image_overlay": "Image Overlay / Blend", # New filter
    "crop": "Crop Image (Interactive)",
    "histogram_equalization": "Histogram Equalization",
    "vignette": "Vignette Effect",
    "cartoonify": "Cartoonify",
    "pencil_sketch": "Pencil Sketch" # New filter
}

def apply_filter(image_np, filter_name, brightness=0, contrast=1, sharpen_amount=0,
                 threshold_val=127, adaptive_block_size=11, adaptive_c=2,
                 rotation_angle=0, flip_code=None,
                 red_adjust=0, green_adjust=0, blue_adjust=0,
                 hue_adjust=0, saturation_adjust=0, lightness_adjust=0,
                 color_temp_adjust=0, exposure_val=1.0,
                 noise_kernel_size=5, gaussian_kernel_size=5, resize_percent=100,
                 watermark_text="", watermark_font_size=30, watermark_color="#000000", watermark_position="bottom_right",
                 overlay_image_data=None, overlay_opacity=0.5,
                 crop_coords=None, vignette_intensity=0.5):
    """
    Applies the specified image processing filter to a NumPy array image.
    Args:
        image_np (numpy.ndarray): The input image as a NumPy array.
        filter_name (str): The name of the filter to apply.
        brightness (int): Brightness adjustment for 'brightness_contrast' filter.
        contrast (float): Contrast adjustment for 'brightness_contrast' filter.
        sharpen_amount (float): Amount of sharpening for 'sharpen' filter.
        threshold_val (int): Threshold value for 'threshold' filter.
        adaptive_block_size (int): Block size for adaptive thresholding.
        adaptive_c (int): C value for adaptive thresholding.
        rotation_angle (int): Angle for rotation (0, 90, 180, 270).
        flip_code (int): 0 for vertical, 1 for horizontal, -1 for both.
        red_adjust (int): Red channel adjustment (-100 to 100).
        green_adjust (int): Green channel adjustment (-100 to 100).
        blue_adjust (int): Blue channel adjustment (-100 to 100).
        hue_adjust (int): Hue adjustment (-180 to 180).
        saturation_adjust (int): Saturation adjustment (-100 to 100).
        lightness_adjust (int): Lightness adjustment (-100 to 100).
        color_temp_adjust (int): Color temperature adjustment (-100 to 100, cooler to warmer).
        exposure_val (float): Exposure/Gamma correction value (0.1 to 5.0).
        noise_kernel_size (int): Kernel size for noise reduction (must be odd).
        gaussian_kernel_size (int): Kernel size for Gaussian blur (must be odd).
        resize_percent (int): Percentage to resize the image (1 to 200).
        watermark_text (str): Text to apply as watermark.
        watermark_font_size (int): Font size of the watermark text.
        watermark_color (str): Hex color of the watermark text.
        watermark_position (str): Position of the watermark ('top_left', 'top_right', etc.).
        overlay_image_data (str): Base64 string of the image to overlay.
        overlay_opacity (float): Opacity of the overlay image (0.0 to 1.0).
        crop_coords (dict): Dictionary with 'x', 'y', 'width', 'height' for cropping.
        vignette_intensity (float): Intensity of the vignette effect (0.0 to 1.0).
    Returns:
        numpy.ndarray: The processed image.
    """
    processed_image = image_np.copy()

    # Apply resizing as a pre-processing step if it's the selected filter
    if filter_name == "resize" and resize_percent != 100:
        width = int(processed_image.shape[1] * resize_percent / 100)
        height = int(processed_image.shape[0] * resize_percent / 100)
        dim = (width, height)
        processed_image = cv2.resize(processed_image, dim, interpolation=cv2.INTER_AREA)

    if filter_name == "grayscale":
        if len(processed_image.shape) == 3 and processed_image.shape[2] == 3:
            processed_image = cv2.cvtColor(processed_image, cv2.COLOR_BGR2GRAY)
            processed_image = cv2.cvtColor(processed_image, cv2.COLOR_GRAY2BGR)
    elif filter_name == "blur":
        # Ensure kernel size is odd and positive
        k_size = max(1, gaussian_kernel_size)
        if k_size % 2 == 0:
            k_size += 1
        processed_image = cv2.GaussianBlur(processed_image, (k_size, k_size), 0)
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
        kernel = np.array([[0.272, 0.534, 0.131],
                           [0.349, 0.686, 0.168],
                           [0.393, 0.769, 0.189]])
        processed_image = cv2.transform(processed_image, kernel.T)
        processed_image = np.clip(processed_image, 0, 255).astype(np.uint8)
    elif filter_name == "sharpen":
        kernel = np.array([[-1, -1, -1],
                           [-1,  9, -1],
                           [-1, -1, -1]])
        sharpened = cv2.filter2D(processed_image, -1, kernel * (sharpen_amount / 100.0))
        processed_image = np.clip(sharpened, 0, 255).astype(np.uint8)
    elif filter_name == "threshold":
        gray_image = cv2.cvtColor(processed_image, cv2.COLOR_BGR2GRAY)
        _, thresholded = cv2.threshold(gray_image, threshold_val, 255, cv2.THRESH_BINARY)
        processed_image = cv2.cvtColor(thresholded, cv2.COLOR_GRAY2BGR)
    elif filter_name == "adaptive_threshold":
        gray_image = cv2.cvtColor(processed_image, cv2.COLOR_BGR2GRAY)
        b_size = max(3, adaptive_block_size)
        if b_size % 2 == 0:
            b_size += 1
        thresholded = cv2.adaptiveThreshold(gray_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, b_size, adaptive_c)
        processed_image = cv2.cvtColor(thresholded, cv2.COLOR_GRAY2BGR)
    elif filter_name.startswith("rotate_"):
        if rotation_angle == 90:
            processed_image = cv2.rotate(processed_image, cv2.ROTATE_90_CLOCKWISE)
        elif rotation_angle == 180:
            processed_image = cv2.rotate(processed_image, cv2.ROTATE_180)
        elif rotation_angle == 270:
            processed_image = cv2.rotate(processed_image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    elif filter_name.startswith("flip_"):
        if flip_code == 0:
            processed_image = cv2.flip(processed_image, 0)
        elif flip_code == 1:
            processed_image = cv2.flip(processed_image, 1)
    elif filter_name == "noise_reduction":
        processed_image = cv2.medianBlur(processed_image, noise_kernel_size)
    elif filter_name == "color_adjust":
        b, g, r = cv2.split(processed_image)
        b = np.clip(b.astype(np.int32) + blue_adjust, 0, 255).astype(np.uint8)
        g = np.clip(g.astype(np.int32) + green_adjust, 0, 255).astype(np.uint8)
        r = np.clip(r.astype(np.int32) + red_adjust, 0, 255).astype(np.uint8)
        processed_image = cv2.merge([b, g, r])
    elif filter_name == "hsl_adjust":
        hls_image = cv2.cvtColor(processed_image, cv2.COLOR_BGR2HLS)
        h, l, s = cv2.split(hls_image)

        h = np.clip(h.astype(np.int32) + hue_adjust // 2, 0, 179).astype(np.uint8)
        l = np.clip(l.astype(np.int32) + (lightness_adjust * 255 / 100), 0, 255).astype(np.uint8)
        s = np.clip(s.astype(np.int32) + (saturation_adjust * 255 / 100), 0, 255).astype(np.uint8)

        processed_image = cv2.merge([h, l, s])
        processed_image = cv2.cvtColor(processed_image, cv2.COLOR_HLS2BGR)
    elif filter_name == "color_temperature": # New: Color Temperature
        # Convert BGR to LAB color space
        lab = cv2.cvtColor(processed_image, cv2.COLOR_BGR2LAB)
        L, A, B = cv2.split(lab)

        # Adjust 'b' channel for yellow-blue balance
        # Positive `color_temp_adjust` for warmer (more yellow), negative for cooler (more blue)
        B = np.clip(B.astype(np.int32) + color_temp_adjust, 0, 255).astype(np.uint8)

        # Optionally, slightly adjust 'a' (green-red) for a more natural shift
        # A = np.clip(A.astype(np.int32) + (color_temp_adjust * 0.2), 0, 255).astype(np.uint8)

        processed_image = cv2.merge([L, A, B])
        processed_image = cv2.cvtColor(processed_image, cv2.COLOR_LAB2BGR)
    elif filter_name == "exposure_gamma": # New: Exposure/Gamma Correction
        # Gamma correction: new_pixel = old_pixel ^ (1 / gamma)
        # Higher exposure_val means less gamma correction (brighter image overall for a given pixel value)
        # A value of 1.0 means no change.
        if exposure_val <= 0: exposure_val = 0.01 # Prevent division by zero or negative gamma
        gamma = 1.0 / exposure_val
        
        # Build a lookup table (LUT)
        table = np.array([((i / 255.0) ** gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
        processed_image = cv2.LUT(processed_image, table)
    elif filter_name == "dithering": # New: Simple Dithering (Monochrome)
        if len(processed_image.shape) == 3 and processed_image.shape[2] == 3:
            gray_image = cv2.cvtColor(processed_image, cv2.COLOR_BGR2GRAY)
        else:
            gray_image = processed_image

        # Add random noise for a dither-like effect before thresholding
        noise = np.random.randint(-30, 30, gray_image.shape, dtype="int16")
        noisy_gray = np.clip(gray_image.astype(np.int16) + noise, 0, 255).astype(np.uint8)

        # Apply adaptive threshold for a more detailed dither (can also use simple cv2.THRESH_BINARY)
        # Using ADAPTIVE_THRESH_MEAN_C with small block size and C value
        dithered_image = cv2.adaptiveThreshold(noisy_gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
        processed_image = cv2.cvtColor(dithered_image, cv2.COLOR_GRAY2BGR)

    elif filter_name == "histogram_equalization":
        if len(processed_image.shape) == 3 and processed_image.shape[2] == 3:
            lab = cv2.cvtColor(processed_image, cv2.COLOR_BGR2LAB)
            l, a, b_channel = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
            cl = clahe.apply(l)
            limg = cv2.merge([cl,a,b_channel])
            processed_image = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
        else:
            processed_image = cv2.equalizeHist(processed_image)
            processed_image = cv2.cvtColor(processed_image, cv2.COLOR_GRAY2BGR)
    elif filter_name == "vignette":
        rows, cols = processed_image.shape[:2]
        
        center_x, center_y = cols // 2, rows // 2
        Y, X = np.ogrid[:rows, :cols]
        dist_from_center = np.sqrt((X - center_x)**2 + (Y - center_y)**2)
        
        max_radius = np.sqrt(center_x**2 + center_y**2)
        normalized_dist = dist_from_center / max_radius

        circular_mask = (1 - normalized_dist).clip(0, 1)
        
        circular_mask = circular_mask ** (1.0 / max(0.1, vignette_intensity * 2))
        circular_mask = circular_mask * (1.0 + vignette_intensity * 0.5) 
        circular_mask = np.clip(circular_mask, 0, 1).astype(np.float32)

        mask_3_channel = np.stack([circular_mask, circular_mask, circular_mask], axis=-1)
        
        processed_image = cv2.multiply(processed_image.astype(np.float32), mask_3_channel).astype(np.uint8)

    elif filter_name == "cartoonify":
        gray = cv2.cvtColor(processed_image, cv2.COLOR_BGR2GRAY)
        gray = cv2.medianBlur(gray, 5)
        edges = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 9)
        color_image = cv2.bilateralFilter(processed_image, 9, 250, 250)
        processed_image = cv2.bitwise_and(color_image, color_image, mask=edges)
    elif filter_name == "pencil_sketch":
        if len(processed_image.shape) == 3 and processed_image.shape[2] == 3:
            gray_image = cv2.cvtColor(processed_image, cv2.COLOR_BGR2GRAY)
        else:
            gray_image = processed_image

        inverted_image = cv2.bitwise_not(gray_image)
        blurred_image = cv2.GaussianBlur(inverted_image, (21, 21), 0)
        inverted_blurred_image = cv2.bitwise_not(blurred_image)
        processed_image_sketch = cv2.divide(gray_image.astype(np.float32), inverted_blurred_image.astype(np.float32) + 1, scale=256.0)
        processed_image_sketch = np.clip(processed_image_sketch, 0, 255).astype(np.uint8)

        if len(processed_image_sketch.shape) == 2:
            processed_image = cv2.cvtColor(processed_image_sketch, cv2.COLOR_GRAY2BGR)
        else:
            processed_image = processed_image_sketch


    # Apply image overlay if provided and filter is selected (can be combined with other filters)
    if filter_name == "image_overlay" and overlay_image_data:
        try:
            nparr_overlay = np.frombuffer(base64.b64decode(overlay_image_data), np.uint8)
            overlay_img_np = cv2.imdecode(nparr_overlay, cv2.IMREAD_UNCHANGED)

            if overlay_img_np is None:
                print("WARNING: Could not decode overlay image.")
            else:
                overlay_resized = cv2.resize(overlay_img_np, (processed_image.shape[1], processed_image.shape[0]), interpolation=cv2.INTER_AREA)

                if overlay_resized.shape[2] == 4:
                    alpha_channel = overlay_resized[:, :, 3] / 255.0
                    alpha_factor = overlay_opacity * alpha_channel
                    
                    img_float = processed_image.astype(np.float32)
                    overlay_bgr = overlay_resized[:, :, :3].astype(np.float32)
                    
                    for c in range(0, 3):
                        processed_image[:, :, c] = np.clip((alpha_factor * overlay_bgr[:, :, c] + (1 - alpha_factor) * img_float[:, :, c]), 0, 255).astype(np.uint8)
                else:
                    processed_image = cv2.addWeighted(processed_image, 1 - overlay_opacity, overlay_resized, overlay_opacity, 0)
        except Exception as e:
            print(f"Error applying image overlay: {e}")

    # Watermark should ideally be the last step if it applies to the final image
    if watermark_text:
        overlay = processed_image.copy()
        h, w, _ = processed_image.shape
        
        watermark_color_bgr = tuple(int(watermark_color[i:i+2], 16) for i in (5, 3, 1))
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = watermark_font_size / 50.0
        font_thickness = max(1, int(watermark_font_size / 20))
        
        text_size_info = cv2.getTextSize(watermark_text, font, font_scale, font_thickness)[0]
        text_w, text_h = text_size_info[0], text_size_info[1]
        
        padding = 10
        if watermark_position == "top_left":
            text_x, text_y = padding, padding + text_h
        elif watermark_position == "top_right":
            text_x, text_y = w - text_w - padding, padding + text_h
        elif watermark_position == "bottom_left":
            text_x, text_y = padding, h - padding
        elif watermark_position == "bottom_right":
            text_x, text_y = w - text_w - padding, h - padding
        elif watermark_position == "center":
            text_x, text_y = (w - text_w) // 2, (h + text_h) // 2
        else:
            text_x, text_y = w - text_w - padding, h - padding

        cv2.putText(overlay, watermark_text, (text_x, text_y), font, font_scale, watermark_color_bgr, font_thickness, cv2.LINE_AA)
        
        alpha = 0.7
        processed_image = cv2.addWeighted(overlay, alpha, processed_image, 1 - alpha, 0)

    # Apply cropping as the very last step, after all other transformations
    if filter_name == "crop" and crop_coords:
        x, y, w_crop, h_crop = crop_coords['x'], crop_coords['y'], crop_coords['width'], crop_coords['height']
        # Ensure coordinates are within image bounds
        x = max(0, x)
        y = max(0, y)
        w_crop = min(w_crop, processed_image.shape[1] - x)
        h_crop = min(h_crop, processed_image.shape[0] - y)
        if w_crop > 0 and h_crop > 0:
            processed_image = processed_image[y:y+h_crop, x:x+w_crop]
        else:
            print(f"Invalid crop dimensions: x={x}, y={y}, w={w_crop}, h={h_crop}. Returning original image.")
            return image_np
    
    elif filter_name == "original" and resize_percent == 100:
        return image_np
    
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
            position: relative; /* For loading overlay and interactive cropping */
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

        /* Cropping overlay styles */
        .crop-overlay {
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            pointer-events: none; /* Allows clicks to pass through by default */
            z-index: 5;
        }
        .crop-selection-rect {
            border: 2px dashed #3b82f6; /* Blue dashed border for selection */
            background-color: rgba(59, 130, 246, 0.2); /* Semi-transparent blue fill */
            position: absolute;
            box-sizing: border-box; /* Ensure border is included in dimensions */
            pointer-events: auto; /* Make the selection rectangle interactive */
            cursor: move;
            resize: both; /* Allow resizing */
            overflow: hidden; /* Hide overflow when resizing */
        }
        .crop-overlay.active {
            pointer-events: auto; /* Enable pointer events when active */
            cursor: crosshair;
            background-color: rgba(0,0,0,0.3); /* Darken background when cropping */
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
                    <p id="originalImagePlaceholder" class="text-gray-500">Upload an image to get started!</p>
                    <img id="originalImagePreview" class="hidden" src="" alt="Original Image Preview">
                    <div id="cropOverlay" class="crop-overlay hidden">
                        <div id="cropSelectionRect" class="crop-selection-rect hidden"></div>
                    </div>
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
                <option value="threshold">Global Threshold (Binarize)</option>
                <option value="adaptive_threshold">Adaptive Threshold</option>
                <option value="rotate_90">Rotate 90° CW</option>
                <option value="rotate_180">Rotate 180°</option>
                <option value="rotate_270">Rotate 270° CW</option>
                <option value="flip_horizontal">Flip Horizontal</option>
                <option value="flip_vertical">Flip Vertical</option>
                <option value="noise_reduction">Noise Reduction (Median)</option>
                <option value="resize">Resize Image</option>
                <option value="color_adjust">RGB Channel Adjust</option>
                <option value="hsl_adjust">HSL Color Adjust</option>
                <option value="color_temperature">Color Temperature</option>
                <option value="exposure_gamma">Exposure/Gamma Correction</option>
                <option value="dithering">Dithering (Monochrome)</option>
                <option value="watermark_text">Text Watermark</option>
                <option value="image_overlay">Image Overlay / Blend</option>
                <option value="histogram_equalization">Histogram Equalization</option>
                <option value="vignette">Vignette Effect</option>
                <option value="cartoonify">Cartoonify</option>
                <option value="pencil_sketch">Pencil Sketch</option>
            </select>
            <button id="applyFilterBtn" class="bg-indigo-600 hover:bg-indigo-700 text-white font-bold py-2 px-4 rounded-full shadow-lg transition-all duration-300 transform hover:scale-105 active:scale-95 disabled:opacity-50 disabled:cursor-not-allowed">
                Apply Filter
            </button>
        </div>

        <!-- Dynamic Controls Container -->
        <div id="dynamicControls" class="flex flex-col items-center gap-3 mb-6 p-4 bg-gray-50 rounded-lg shadow-inner hidden">
            <!-- Content will be injected here by JS -->
        </div>

        <!-- Crop Controls -->
        <div id="cropControls" class="flex flex-col items-center gap-3 mb-6 p-4 bg-gray-50 rounded-lg shadow-inner hidden">
            <h3 class="text-lg font-medium text-gray-700">Interactive Cropping</h3>
            <p class="text-sm text-gray-600">Click and drag on the <b>Original Image</b> above to select an area.</p>
            <button id="startCroppingBtn" class="bg-blue-500 hover:bg-blue-700 text-white font-bold py-2 px-4 rounded-full shadow-md transition-all">
                Start Cropping Selection
            </button>
            <button id="performCropBtn" class="bg-indigo-600 hover:bg-indigo-700 text-white font-bold py-2 px-4 rounded-full shadow-md transition-all hidden" disabled>
                Crop Image
            </button>
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
        const originalImagePreview = document.getElementById('originalImagePreview');
        const originalImagePlaceholder = document.getElementById('originalImagePlaceholder');
        const processedImageContainer = document.getElementById('processedImageContainer');
        const filterSelect = document.getElementById('filterSelect');
        const applyFilterBtn = document.getElementById('applyFilterBtn');
        const toastContainer = document.getElementById('toast-container');
        const dynamicControls = document.getElementById('dynamicControls');
        const processingOverlay = document.getElementById('processingOverlay');
        const downloadBtn = document.getElementById('downloadBtn');
        const resetBtn = document.getElementById('resetBtn');

        // Cropping elements
        const cropControls = document.getElementById('cropControls');
        const startCroppingBtn = document.getElementById('startCroppingBtn');
        const performCropBtn = document.getElementById('performCropBtn');
        const cropOverlay = document.getElementById('cropOverlay');
        const cropSelectionRect = document.getElementById('cropSelectionRect');

        let currentImageBase64 = null; // Holds the base64 of the original image after upload
        let currentOverlayImageBase64 = null; // New: Holds base64 for overlay image
        let currentProcessedImageBase64 = null; // Holds the base64 of the last processed image

        // Cropping state variables
        let isDrawing = false;
        let startX, startY;
        let cropRect = { x: 0, y: 0, width: 0, height: 0 };
        let imageNaturalWidth = 0;
        let imageNaturalHeight = 0;
        let containerClientWidth = 0;
        let containerClientHeight = 0;

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
            cropControls.classList.add('hidden'); // Hide crop controls by default

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
                document.getElementById('brightnessSlider').addEventListener('input', () => {
                    document.getElementById('brightnessValue').textContent = document.getElementById('brightnessSlider').value;
                    triggerImageProcessing();
                });
                document.getElementById('contrastSlider').addEventListener('input', () => {
                    document.getElementById('contrastValue').textContent = (document.getElementById('contrastSlider').value / 100).toFixed(1);
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
                document.getElementById('sharpenAmountSlider').addEventListener('input', () => {
                    document.getElementById('sharpenAmountValue').textContent = document.getElementById('sharpenAmountSlider').value;
                    triggerImageProcessing();
                });
            } else if (filterName === 'threshold') {
                dynamicControls.classList.remove('hidden');
                dynamicControls.innerHTML = `
                    <h3 class="text-lg font-medium text-gray-700">Global Threshold Settings</h3>
                    <div class="w-full max-w-xs">
                        <label for="thresholdSlider" class="block text-sm font-medium text-gray-700">Threshold Value: <span id="thresholdValue">127</span></label>
                        <input type="range" id="thresholdSlider" min="0" max="255" value="127" class="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer range-lg">
                    </div>
                    <p class="text-xs text-gray-500">Set the pixel intensity threshold (0-255) for binarization.</p>
                `;
                document.getElementById('thresholdSlider').addEventListener('input', () => {
                    document.getElementById('thresholdValue').textContent = document.getElementById('thresholdSlider').value;
                    triggerImageProcessing();
                });
            } else if (filterName === 'adaptive_threshold') {
                dynamicControls.classList.remove('hidden');
                dynamicControls.innerHTML = `
                    <h3 class="text-lg font-medium text-gray-700">Adaptive Threshold Settings</h3>
                    <div class="w-full max-w-xs">
                        <label for="adaptiveBlockSizeSlider" class="block text-sm font-medium text-gray-700">Block Size: <span id="adaptiveBlockSizeValue">11</span></label>
                        <input type="range" id="adaptiveBlockSizeSlider" min="3" max="51" value="11" step="2" class="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer range-lg">
                    </div>
                    <div class="w-full max-w-xs">
                        <label for="adaptiveCSlider" class="block text-sm font-medium text-gray-700">C Value: <span id="adaptiveCValue">2</span></label>
                        <input type="range" id="adaptiveCSlider" min="0" max="20" value="2" class="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer range-lg">
                    </div>
                    <p class="text-xs text-gray-500">Block Size must be odd. C is constant subtracted from mean.</p>
                `;
                document.getElementById('adaptiveBlockSizeSlider').addEventListener('input', () => {
                    let val = parseInt(document.getElementById('adaptiveBlockSizeSlider').value);
                    if (val % 2 === 0) val += 1;
                    document.getElementById('adaptiveBlockSizeValue').textContent = val;
                    document.getElementById('adaptiveBlockSizeSlider').value = val;
                    triggerImageProcessing();
                });
                document.getElementById('adaptiveCSlider').addEventListener('input', () => {
                    document.getElementById('adaptiveCValue').textContent = document.getElementById('adaptiveCSlider').value;
                    triggerImageProcessing();
                });
            }
             else if (filterName === 'color_adjust') {
                dynamicControls.classList.remove('hidden');
                dynamicControls.innerHTML = `
                    <h3 class="text-lg font-medium text-gray-700">RGB Channel Adjustment</h3>
                    <div class="w-full max-w-xs">
                        <label for="redAdjustSlider" class="block text-sm font-medium text-gray-700 text-red-600">Red: <span id="redAdjustValue">0</span></label>
                        <input type="range" id="redAdjustSlider" min="-100" max="100" value="0" class="w-full h-2 bg-red-200 rounded-lg appearance-none cursor-pointer range-lg">
                    </div>
                    <div class="w-full max-w-xs">
                        <label for="greenAdjustSlider" class="block text-sm font-medium text-gray-700 text-green-600">Green: <span id="greenAdjustValue">0</span></label>
                        <input type="range" id="greenAdjustSlider" min="-100" max="100" value="0" class="w-full h-2 bg-green-200 rounded-lg appearance-none cursor-pointer range-lg">
                    </div>
                    <div class="w-full max-w-xs">
                        <label for="blueAdjustSlider" class="block text-sm font-medium text-gray-700 text-blue-600">Blue: <span id="blueAdjustValue">0</span></label>
                        <input type="range" id="blueAdjustSlider" min="-100" max="100" value="0" class="w-full h-2 bg-blue-200 rounded-lg appearance-none cursor-pointer range-lg">
                    </div>
                    <p class="text-xs text-gray-500">Adjust individual RGB channel intensities (-100 to 100).</p>
                `;
                document.getElementById('redAdjustSlider').addEventListener('input', triggerImageProcessing);
                document.getElementById('greenAdjustSlider').addEventListener('input', triggerImageProcessing);
                document.getElementById('blueAdjustSlider').addEventListener('input', triggerImageProcessing);
                document.getElementById('redAdjustSlider').addEventListener('input', () => { document.getElementById('redAdjustValue').textContent = document.getElementById('redAdjustSlider').value; });
                document.getElementById('greenAdjustSlider').addEventListener('input', () => { document.getElementById('greenAdjustValue').textContent = document.getElementById('greenAdjustSlider').value; });
                document.getElementById('blueAdjustSlider').addEventListener('input', () => { document.getElementById('blueAdjustValue').textContent = document.getElementById('blueAdjustSlider').value; });
            } else if (filterName === 'hsl_adjust') {
                dynamicControls.classList.remove('hidden');
                dynamicControls.innerHTML = `
                    <h3 class="text-lg font-medium text-gray-700">HSL Color Adjustment</h3>
                    <div class="w-full max-w-xs">
                        <label for="hueAdjustSlider" class="block text-sm font-medium text-gray-700">Hue: <span id="hueAdjustValue">0</span></label>
                        <input type="range" id="hueAdjustSlider" min="-180" max="180" value="0" class="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer range-lg">
                    </div>
                    <div class="w-full max-w-xs">
                        <label for="saturationAdjustSlider" class="block text-sm font-medium text-gray-700">Saturation: <span id="saturationAdjustValue">0</span></label>
                        <input type="range" id="saturationAdjustSlider" min="-100" max="100" value="0" class="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer range-lg">
                    </div>
                    <div class="w-full max-w-xs">
                        <label for="lightnessAdjustSlider" class="block text-sm font-medium text-gray-700">Lightness: <span id="lightnessAdjustValue">0</span></label>
                        <input type="range" id="lightnessAdjustSlider" min="-100" max="100" value="0" class="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer range-lg">
                    </div>
                    <p class="text-xs text-gray-500">Adjust Hue (-180 to 180), Saturation (-100 to 100), and Lightness (-100 to 100).</p>
                `;
                document.getElementById('hueAdjustSlider').addEventListener('input', () => {
                    document.getElementById('hueAdjustValue').textContent = document.getElementById('hueAdjustSlider').value;
                    triggerImageProcessing();
                });
                document.getElementById('saturationAdjustSlider').addEventListener('input', () => {
                    document.getElementById('saturationAdjustValue').textContent = document.getElementById('saturationAdjustSlider').value;
                    triggerImageProcessing();
                });
                document.getElementById('lightnessAdjustSlider').addEventListener('input', () => {
                    document.getElementById('lightnessAdjustValue').textContent = document.getElementById('lightnessAdjustSlider').value;
                    triggerImageProcessing();
                });
            } else if (filterName === 'color_temperature') {
                dynamicControls.classList.remove('hidden');
                dynamicControls.innerHTML = `
                    <h3 class="text-lg font-medium text-gray-700">Color Temperature</h3>
                    <div class="w-full max-w-xs">
                        <label for="colorTempSlider" class="block text-sm font-medium text-gray-700">Temperature: <span id="colorTempValue">0</span></label>
                        <input type="range" id="colorTempSlider" min="-100" max="100" value="0" class="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer range-lg">
                    </div>
                    <p class="text-xs text-gray-500">Adjust color temperature (-100 for cooler/blue, 100 for warmer/yellow).</p>
                `;
                document.getElementById('colorTempSlider').addEventListener('input', () => {
                    document.getElementById('colorTempValue').textContent = document.getElementById('colorTempSlider').value;
                    triggerImageProcessing();
                });
            } else if (filterName === 'exposure_gamma') {
                dynamicControls.classList.remove('hidden');
                dynamicControls.innerHTML = `
                    <h3 class="text-lg font-medium text-gray-700">Exposure/Gamma Correction</h3>
                    <div class="w-full max-w-xs">
                        <label for="exposureSlider" class="block text-sm font-medium text-gray-700">Exposure: <span id="exposureValue">1.0</span></label>
                        <input type="range" id="exposureSlider" min="0.1" max="5.0" value="1.0" step="0.1" class="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer range-lg">
                    </div>
                    <p class="text-xs text-gray-500">Adjust image exposure/gamma (0.1 for darker, 5.0 for brighter).</p>
                `;
                document.getElementById('exposureSlider').addEventListener('input', () => {
                    document.getElementById('exposureValue').textContent = parseFloat(document.getElementById('exposureSlider').value).toFixed(1);
                    triggerImageProcessing();
                });
            } else if (filterName === 'noise_reduction') {
                 dynamicControls.classList.remove('hidden');
                 dynamicControls.innerHTML = `
                    <h3 class="text-lg font-medium text-gray-700">Noise Reduction (Median Blur)</h3>
                    <div class="w-full max-w-xs">
                        <label for="noiseKernelSlider" class="block text-sm font-medium text-gray-700">Kernel Size: <span id="noiseKernelValue">5</span></label>
                        <input type="range" id="noiseKernelSlider" min="3" max="15" value="5" step="2" class="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer range-lg">
                    </div>
                    <p class="text-xs text-gray-500">Adjust strength (odd numbers, larger = more blur).</p>
                 `;
                 document.getElementById('noiseKernelSlider').addEventListener('input', () => {
                    const val = parseInt(document.getElementById('noiseKernelSlider').value);
                    document.getElementById('noiseKernelValue').textContent = val;
                    triggerImageProcessing();
                 });
            } else if (filterName === 'blur') {
                dynamicControls.classList.remove('hidden');
                dynamicControls.innerHTML = `
                    <h3 class="text-lg font-medium text-gray-700">Gaussian Blur Settings</h3>
                    <div class="w-full max-w-xs">
                        <label for="gaussianKernelSlider" class="block text-sm font-medium text-gray-700">Kernel Size: <span id="gaussianKernelValue">5</span></label>
                        <input type="range" id="gaussianKernelSlider" min="1" max="51" value="5" step="2" class="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer range-lg">
                    </div>
                    <p class="text-xs text-gray-500">Adjust blur strength (odd numbers, larger = more blur).</p>
                `;
                document.getElementById('gaussianKernelSlider').addEventListener('input', () => {
                    let val = parseInt(document.getElementById('gaussianKernelSlider').value);
                    if (val % 2 === 0) val += 1;
                    document.getElementById('gaussianKernelValue').textContent = val;
                    document.getElementById('gaussianKernelSlider').value = val;
                    triggerImageProcessing();
                });
            }
            else if (filterName === 'resize') {
                 dynamicControls.classList.remove('hidden');
                 dynamicControls.innerHTML = `
                    <h3 class="text-lg font-medium text-gray-700">Resize Image</h3>
                    <div class="w-full max-w-xs">
                        <label for="resizePercentSlider" class="block text-sm font-medium text-gray-700">Scale: <span id="resizePercentValue">100%</span></label>
                        <input type="range" id="resizePercentSlider" min="10" max="200" value="100" class="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer range-lg">
                    </div>
                    <p class="text-xs text-gray-500">Resize image by percentage (10% to 200%).</p>
                 `;
                 document.getElementById('resizePercentSlider').addEventListener('input', () => {
                    const val = parseInt(document.getElementById('resizePercentSlider').value);
                    document.getElementById('resizePercentValue').textContent = `${val}%`;
                    triggerImageProcessing();
                 });
            } else if (filterName === 'watermark_text') {
                 dynamicControls.classList.remove('hidden');
                 dynamicControls.innerHTML = `
                    <h3 class="text-lg font-medium text-gray-700">Text Watermark Settings</h3>
                    <div class="w-full max-w-xs mb-2">
                        <label for="watermarkTextInput" class="block text-sm font-medium text-gray-700">Watermark Text:</label>
                        <input type="text" id="watermarkTextInput" placeholder="Your Text Here" class="mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-indigo-300 focus:ring focus:ring-indigo-200 focus:ring-opacity-50">
                    </div>
                    <div class="w-full max-w-xs mb-2">
                        <label for="watermarkFontSizeSlider" class="block text-sm font-medium text-gray-700">Font Size: <span id="watermarkFontSizeValue">30</span></label>
                        <input type="range" id="watermarkFontSizeSlider" min="10" max="100" value="30" class="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer range-lg">
                    </div>
                    <div class="w-full max-w-xs mb-2">
                        <label for="watermarkColorPicker" class="block text-sm font-medium text-gray-700">Color:</label>
                        <input type="color" id="watermarkColorPicker" value="#000000" class="mt-1 w-full h-10 cursor-pointer">
                    </div>
                    <div class="w-full max-w-xs mb-2">
                        <label for="watermarkPositionSelect" class="block text-sm font-medium text-gray-700">Position:</label>
                        <select id="watermarkPositionSelect" class="mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-indigo-300 focus:ring focus:ring-indigo-200 focus:ring-opacity-50">
                            <option value="bottom_right">Bottom Right</option>
                            <option value="bottom_left">Bottom Left</option>
                            <option value="top_right">Top Right</option>
                            <option value="top_left">Top Left</option>
                            <option value="center">Center</option>
                        </select>
                    </div>
                    <p class="text-xs text-gray-500">Customize watermark text, size, color, and position.</p>
                 `;
                 document.getElementById('watermarkTextInput').addEventListener('input', triggerImageProcessing);
                 document.getElementById('watermarkFontSizeSlider').addEventListener('input', () => {
                    document.getElementById('watermarkFontSizeValue').textContent = document.getElementById('watermarkFontSizeSlider').value;
                    triggerImageProcessing();
                 });
                 document.getElementById('watermarkColorPicker').addEventListener('input', triggerImageProcessing);
                 document.getElementById('watermarkPositionSelect').addEventListener('change', triggerImageProcessing);
            } else if (filterName === 'image_overlay') {
                dynamicControls.classList.remove('hidden');
                dynamicControls.innerHTML = `
                    <h3 class="text-lg font-medium text-gray-700">Image Overlay Settings</h3>
                    <div class="w-full max-w-xs mb-2">
                        <label for="overlayImageUpload" class="block text-sm font-medium text-gray-700">Upload Overlay Image:</label>
                        <input type="file" id="overlayImageUpload" name="overlay_image" accept="image/*" class="block w-full text-sm text-gray-500
                            file:mr-4 file:py-2 file:px-4
                            file:rounded-full file:border-0
                            file:text-sm file:font-semibold
                            file:bg-purple-50 file:text-purple-700
                            hover:file:bg-purple-100 cursor-pointer
                        "/>
                        <p class="mt-1 text-xs text-gray-500">Supported formats: JPG, PNG (with transparency).</p>
                    </div>
                    <div class="w-full max-w-xs">
                        <label for="overlayOpacitySlider" class="block text-sm font-medium text-gray-700">Opacity: <span id="overlayOpacityValue">0.5</span></label>
                        <input type="range" id="overlayOpacitySlider" min="0.0" max="1.0" value="0.5" step="0.05" class="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer range-lg">
                    </div>
                    <p class="text-xs text-gray-500">Adjust the transparency of the overlay image.</p>
                `;
                document.getElementById('overlayImageUpload').addEventListener('change', function(e) {
                    const file = e.target.files[0];
                    if (file) {
                        const reader = new FileReader();
                        reader.onload = function(evt) {
                            currentOverlayImageBase64 = evt.target.result.split(',')[1];
                            showToast('Overlay image loaded!', 'success');
                            triggerImageProcessing();
                        };
                        reader.onerror = function() {
                            showToast('Error reading overlay file!', 'error');
                        };
                        reader.readAsDataURL(file);
                    } else {
                        currentOverlayImageBase64 = null;
                        showToast('Overlay image cleared.', 'info');
                        triggerImageProcessing();
                    }
                });
                document.getElementById('overlayOpacitySlider').addEventListener('input', () => {
                    document.getElementById('overlayOpacityValue').textContent = parseFloat(document.getElementById('overlayOpacitySlider').value).toFixed(2);
                    triggerImageProcessing();
                });
            }
            else if (filterName === 'crop') {
                cropControls.classList.remove('hidden');
                deactivateCropDrawing();
                performCropBtn.classList.add('hidden');
            } else if (filterName === 'vignette') {
                dynamicControls.classList.remove('hidden');
                dynamicControls.innerHTML = `
                    <h3 class="text-lg font-medium text-gray-700">Vignette Effect</h3>
                    <div class="w-full max-w-xs">
                        <label for="vignetteIntensitySlider" class="block text-sm font-medium text-gray-700">Intensity: <span id="vignetteIntensityValue">0.5</span></label>
                        <input type="range" id="vignetteIntensitySlider" min="0.1" max="1.0" value="0.5" step="0.1" class="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer range-lg">
                    </div>
                    <p class="text-xs text-gray-500">Adjust the darkening effect towards the edges.</p>
                `;
                document.getElementById('vignetteIntensitySlider').addEventListener('input', () => {
                    document.getElementById('vignetteIntensityValue').textContent = parseFloat(document.getElementById('vignetteIntensitySlider').value).toFixed(1);
                    triggerImageProcessing();
                });
            }
        }

        // Cropping functions
        function activateCropDrawing() {
            if (!originalImagePreview.src) {
                showToast('Please upload an image first to crop!', 'error');
                return;
            }
            cropOverlay.classList.add('active');
            cropSelectionRect.classList.add('hidden');
            performCropBtn.classList.add('hidden');
            isDrawing = false;
            cropRect = { x: 0, y: 0, width: 0, height: 0 };
            showToast('Draw a rectangle on the image to select crop area.', 'info');
        }

        function deactivateCropDrawing() {
            cropOverlay.classList.remove('active');
            cropOverlay.removeEventListener('mousedown', startDrawing);
            cropOverlay.removeEventListener('mousemove', drawRect);
            cropOverlay.removeEventListener('mouseup', stopDrawing);
            isDrawing = false;
        }

        function startDrawing(e) {
            isDrawing = true;
            const rect = originalImagePreview.getBoundingClientRect();
            startX = e.clientX - rect.left;
            startY = e.clientY - rect.top;

            cropSelectionRect.style.left = `${startX}px`;
            cropSelectionRect.style.top = `${startY}px`;
            cropSelectionRect.style.width = '0px';
            cropSelectionRect.style.height = '0px';
            cropSelectionRect.classList.remove('hidden');
            performCropBtn.classList.add('hidden');
        }

        function drawRect(e) {
            if (!isDrawing) return;

            const rect = originalImagePreview.getBoundingClientRect();
            const currentX = e.clientX - rect.left;
            const currentY = e.clientY - rect.top;

            const width = Math.max(0, currentX - startX);
            const height = Math.max(0, currentY - startY);

            cropSelectionRect.style.left = `${Math.min(startX, currentX)}px`;
            cropSelectionRect.style.top = `${Math.min(startY, currentY)}px`;
            cropSelectionRect.style.width = `${Math.abs(width)}px`;
            cropSelectionRect.style.height = `${Math.abs(height)}px`;

            const displayedWidth = originalImagePreview.clientWidth;
            const displayedHeight = originalImagePreview.clientHeight;

            const imgScaleX = imageNaturalWidth / displayedWidth;
            const imgScaleY = imageNaturalHeight / displayedHeight;

            const cropX_display = Math.min(startX, currentX);
            const cropY_display = Math.min(startY, currentY);
            const cropW_display = Math.abs(currentX - startX);
            const cropH_display = Math.abs(currentY - startY);

            cropRect.x = Math.floor(cropX_display * imgScaleX);
            cropRect.y = Math.floor(cropY_display * imgScaleY);
            cropRect.width = Math.floor(cropW_display * imgScaleX);
            cropRect.height = Math.floor(cropH_display * imgScaleY);
            
            cropRect.x = Math.max(0, Math.min(cropRect.x, imageNaturalWidth));
            cropRect.y = Math.max(0, Math.min(cropRect.y, imageNaturalHeight));
            cropRect.width = Math.max(0, Math.min(cropRect.width, imageNaturalWidth - cropRect.x));
            cropRect.height = Math.max(0, Math.min(cropRect.height, imageNaturalHeight - cropRect.y));

            if (cropRect.width > 0 && cropRect.height > 0) {
                performCropBtn.disabled = false;
                performCropBtn.classList.remove('hidden');
            } else {
                performCropBtn.disabled = true;
                performCropBtn.classList.add('hidden');
            }
        }

        function stopDrawing() {
            isDrawing = false;
        }

        startCroppingBtn.addEventListener('click', () => {
            activateCropDrawing();
            cropOverlay.addEventListener('mousedown', startDrawing);
            cropOverlay.addEventListener('mousemove', drawRect);
            cropOverlay.addEventListener('mouseup', stopDrawing);
        });

        performCropBtn.addEventListener('click', () => {
            if (cropRect.width > 0 && cropRect.height > 0) {
                triggerImageProcessing();
                deactivateCropDrawing();
                cropSelectionRect.classList.add('hidden');
            } else {
                showToast('Please select a valid cropping area.', 'warning');
            }
        });


        async function processImage(filterName, params = {}) {
            if (!currentImageBase64) {
                showToast('Please upload an image first!', 'error');
                return;
            }

            processingOverlay.classList.remove('hidden');
            applyFilterBtn.disabled = true;
            applyFilterBtn.innerHTML = '<div class="spinner inline-block align-middle mr-2"></div> Processing...';
            downloadBtn.disabled = true;

            const controller = new AbortController();
            const signal = controller.signal;
            const timeoutId = setTimeout(() => controller.abort(), 60000);

            const payload = {
                image_data: currentImageBase64,
                filter: filterName,
                ...params
            };

            if (filterName === 'crop') {
                if (cropRect.width === 0 || cropRect.height === 0) {
                     showToast('Please select a valid area for cropping!', 'error');
                     processingOverlay.classList.add('hidden');
                     applyFilterBtn.disabled = false;
                     applyFilterBtn.innerHTML = 'Apply Filter';
                     return;
                }
                payload.crop_coords = cropRect;
            }

            if (filterName === 'image_overlay' && currentOverlayImageBase64) {
                payload.overlay_image_data = currentOverlayImageBase64;
            }


            try {
                const response = await fetch('/process_image', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify(payload),
                    signal: signal
                });

                clearTimeout(timeoutId);

                if (!response.ok) {
                    const errorText = await response.text();
                    throw new Error(`Server error: ${response.status} - ${errorText}`);
                }

                const data = await response.json();
                if (data.processed_image) {
                    currentProcessedImageBase64 = data.processed_image;
                    processedImageContainer.innerHTML = `<img src="data:image/jpeg;base64,${data.processed_image}" alt="Processed Image">`;
                    downloadBtn.disabled = false;
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
                downloadBtn.disabled = true;
            } finally {
                processingOverlay.classList.add('hidden');
                applyFilterBtn.disabled = false;
                applyFilterBtn.innerHTML = 'Apply Filter';
            }
        }

        imageUpload.addEventListener('change', function(event) {
            const file = event.target.files[0];
            if (file) {
                const reader = new FileReader();
                reader.onload = function(e) {
                    currentImageBase64 = e.target.result.split(',')[1];
                    
                    originalImagePreview.src = e.target.result;
                    originalImagePreview.classList.remove('hidden');
                    originalImagePlaceholder.classList.add('hidden');

                    originalImagePreview.onload = () => {
                        imageNaturalWidth = originalImagePreview.naturalWidth;
                        imageNaturalHeight = originalImagePreview.naturalHeight;
                        containerClientWidth = originalImageContainer.clientWidth;
                        containerClientHeight = originalImageContainer.clientHeight;
                        console.log(`Original Image: ${imageNaturalWidth}x${imageNaturalHeight}`);
                        console.log(`Container: ${containerClientWidth}x${containerClientHeight}`);
                    };

                    processedImageContainer.innerHTML = `<p class="text-gray-500">Processed image will appear here.</p>`;
                    filterSelect.value = "original";
                    renderDynamicControls("original");
                    downloadBtn.disabled = true;
                    deactivateCropDrawing();
                    cropSelectionRect.classList.add('hidden');
                    performCropBtn.classList.add('hidden');
                    showToast('Image uploaded successfully! 🎉', 'success');
                    triggerImageProcessing();
                };
                reader.onerror = function() {
                    showToast('Error reading file!', 'error');
                };
                reader.readAsDataURL(file);
            } else {
                currentImageBase64 = null;
                currentOverlayImageBase64 = null;
                currentProcessedImageBase64 = null;
                originalImagePreview.src = "";
                originalImagePreview.classList.add('hidden');
                originalImagePlaceholder.classList.remove('hidden');
                processedImageContainer.innerHTML = `<p class="text-gray-500">Processed image will appear here.</p>`;
                filterSelect.value = "original";
                renderDynamicControls("original");
                downloadBtn.disabled = true;
                deactivateCropDrawing();
                cropSelectionRect.classList.add('hidden');
                performCropBtn.classList.add('hidden');
            }
        });

        filterSelect.addEventListener('change', function() {
            renderDynamicControls(filterSelect.value);
            deactivateCropDrawing();
            cropSelectionRect.classList.add('hidden');
            performCropBtn.classList.add('hidden');
            if (currentImageBase64) {
                 triggerImageProcessing();
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
            imageUpload.value = '';
            currentImageBase64 = null;
            currentOverlayImageBase64 = null;
            currentProcessedImageBase64 = null;
            originalImagePreview.src = "";
            originalImagePreview.classList.add('hidden');
            originalImagePlaceholder.classList.remove('hidden');
            processedImageContainer.innerHTML = `<p class="text-gray-500">Processed image will appear here.</p>`;
            filterSelect.value = "original";
            renderDynamicControls("original");
            downloadBtn.disabled = true;
            deactivateCropDrawing();
            cropSelectionRect.classList.add('hidden');
            performCropBtn.classList.add('hidden');
            showToast('Application reset. Ready for new image!', 'info');
        });


        // Central function to gather parameters and call processImage
        function triggerImageProcessing() {
            const selectedFilter = filterSelect.value;
            const params = {};

            if (selectedFilter === 'brightness_contrast') {
                params.brightness = parseInt(document.getElementById('brightnessSlider').value);
                params.contrast = parseFloat((document.getElementById('contrastSlider').value / 100).toFixed(1));
            } else if (selectedFilter === 'sharpen') {
                params.sharpen_amount = parseInt(document.getElementById('sharpenAmountSlider').value);
            } else if (selectedFilter === 'threshold') {
                params.threshold_val = parseInt(document.getElementById('thresholdSlider').value);
            } else if (selectedFilter === 'adaptive_threshold') {
                params.adaptive_block_size = parseInt(document.getElementById('adaptiveBlockSizeSlider').value);
                params.adaptive_c = parseInt(document.getElementById('adaptiveCSlider').value);
            } else if (selectedFilter === 'rotate_90') {
                params.rotation_angle = 90;
            } else if (selectedFilter === 'rotate_180') {
                params.rotation_angle = 180;
            } else if (selectedFilter === 'rotate_270') {
                params.rotation_angle = 270;
            } else if (selectedFilter === 'flip_horizontal') {
                params.flip_code = 1;
            } else if (selectedFilter === 'flip_vertical') {
                params.flip_code = 0;
            } else if (selectedFilter === 'color_adjust') {
                params.red_adjust = parseInt(document.getElementById('redAdjustSlider').value);
                params.green_adjust = parseInt(document.getElementById('greenAdjustSlider').value);
                params.blue_adjust = parseInt(document.getElementById('blueAdjustSlider').value);
            } else if (selectedFilter === 'hsl_adjust') {
                params.hue_adjust = parseInt(document.getElementById('hueAdjustSlider').value);
                params.saturation_adjust = parseInt(document.getElementById('saturationAdjustSlider').value);
                params.lightness_adjust = parseInt(document.getElementById('lightnessAdjustSlider').value);
            } else if (selectedFilter === 'color_temperature') {
                params.color_temp_adjust = parseInt(document.getElementById('colorTempSlider').value);
            } else if (selectedFilter === 'exposure_gamma') {
                params.exposure_val = parseFloat(document.getElementById('exposureSlider').value);
            } else if (selectedFilter === 'noise_reduction') {
                params.noise_kernel_size = parseInt(document.getElementById('noiseKernelSlider').value);
            } else if (selectedFilter === 'blur') {
                params.gaussian_kernel_size = parseInt(document.getElementById('gaussianKernelSlider').value);
            } else if (selectedFilter === 'resize') {
                params.resize_percent = parseInt(document.getElementById('resizePercentSlider').value);
            } else if (selectedFilter === 'watermark_text') {
                params.watermark_text = document.getElementById('watermarkTextInput').value;
                params.watermark_font_size = parseInt(document.getElementById('watermarkFontSizeSlider').value);
                params.watermark_color = document.getElementById('watermarkColorPicker').value;
                params.watermark_position = document.getElementById('watermarkPositionSelect').value;
            } else if (selectedFilter === 'image_overlay') {
                params.overlay_opacity = parseFloat(document.getElementById('overlayOpacitySlider').value);
            } else if (selectedFilter === 'vignette') {
                params.vignette_intensity = parseFloat(document.getElementById('vignetteIntensitySlider').value);
            }

            processImage(selectedFilter, params);
        }

        document.addEventListener('DOMContentLoaded', () => {
            renderDynamicControls(filterSelect.value);
            downloadBtn.disabled = true;
            originalImagePreview.classList.add('hidden');
            originalImagePlaceholder.classList.remove('hidden');
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
    
    brightness = int(data.get('brightness', 0))
    contrast = float(data.get('contrast', 1.0))
    sharpen_amount = float(data.get('sharpen_amount', 0))
    threshold_val = int(data.get('threshold_val', 127))
    adaptive_block_size = int(data.get('adaptive_block_size', 11))
    adaptive_c = int(data.get('adaptive_c', 2))
    rotation_angle = int(data.get('rotation_angle', 0))
    flip_code = data.get('flip_code', None)
    if flip_code is not None:
        flip_code = int(flip_code)

    red_adjust = int(data.get('red_adjust', 0))
    green_adjust = int(data.get('green_adjust', 0))
    blue_adjust = int(data.get('blue_adjust', 0))
    hue_adjust = int(data.get('hue_adjust', 0))
    saturation_adjust = int(data.get('saturation_adjust', 0))
    lightness_adjust = int(data.get('lightness_adjust', 0))
    color_temp_adjust = int(data.get('color_temp_adjust', 0)) # New parameter
    exposure_val = float(data.get('exposure_val', 1.0)) # New parameter
    noise_kernel_size = int(data.get('noise_kernel_size', 5))
    gaussian_kernel_size = int(data.get('gaussian_kernel_size', 5))
    resize_percent = int(data.get('resize_percent', 100))
    watermark_text = data.get('watermark_text', "")
    watermark_font_size = int(data.get('watermark_font_size', 30))
    watermark_color = data.get('watermark_color', "#000000")
    watermark_position = data.get('watermark_position', "bottom_right")
    overlay_image_data = data.get('overlay_image_data', None)
    overlay_opacity = float(data.get('overlay_opacity', 0.5))
    crop_coords = data.get('crop_coords', None)
    vignette_intensity = float(data.get('vignette_intensity', 0.5))


    try:
        nparr = np.frombuffer(base64.b64decode(image_data), np.uint8)
        img_np = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img_np is None:
            print("ERROR: cv2.imdecode returned None. Image data might be corrupt or unsupported format.")
            return "Could not decode image. Please ensure it's a valid image file.", 400

        with image_data_lock:
            global current_original_img_np
            current_original_img_np = img_np.copy()
        
        if filter_name == "original":
             processed_img_np = current_original_img_np.copy()
        else:
             processed_img_np = apply_filter(current_original_img_np, filter_name,
                                            brightness=brightness,
                                            contrast=contrast,
                                            sharpen_amount=sharpen_amount,
                                            threshold_val=threshold_val,
                                            adaptive_block_size=adaptive_block_size,
                                            adaptive_c=adaptive_c,
                                            rotation_angle=rotation_angle,
                                            flip_code=flip_code,
                                            red_adjust=red_adjust,
                                            green_adjust=green_adjust,
                                            blue_adjust=blue_adjust,
                                            hue_adjust=hue_adjust,
                                            saturation_adjust=saturation_adjust,
                                            lightness_adjust=lightness_adjust,
                                            color_temp_adjust=color_temp_adjust, # Pass new parameter
                                            exposure_val=exposure_val, # Pass new parameter
                                            noise_kernel_size=noise_kernel_size,
                                            gaussian_kernel_size=gaussian_kernel_size,
                                            resize_percent=resize_percent,
                                            watermark_text=watermark_text,
                                            watermark_font_size=watermark_font_size,
                                            watermark_color=watermark_color,
                                            watermark_position=watermark_position,
                                            overlay_image_data=overlay_image_data,
                                            overlay_opacity=overlay_opacity,
                                            crop_coords=crop_coords,
                                            vignette_intensity=vignette_intensity)


        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 90]
        _, buffer = cv2.imencode('.jpeg', processed_img_np, encode_param)
        processed_image_base64 = base64.b64encode(buffer).decode('utf-8')

        return jsonify({'processed_image': processed_image_base64})

    except Exception as e:
        import traceback
        print(f"AN UNEXPECTED ERROR OCCURRED DURING IMAGE PROCESSING: {e}")
        print(traceback.format_exc())
        return f"Image processing failed on server: {e}", 500

if __name__ == '__main__':
    print("Image Processing Web Application Starting...")
    print("---------------------------------------")
    print("Flask server will start on: http://127.0.0.1:5000")
    print("Open this URL in your web browser to upload and process images.")
    print("Press Ctrl+C in this terminal to stop the server.")
    print("---------------------------------------")
    app.run(host='0.0.0.0', port=5000, debug=False)
