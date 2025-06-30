import os
from PIL import Image, ImageFilter, ImageEnhance, ImageDraw, ImageFont

# Placeholder imports for OpenCV, typically used for more advanced tasks
# This block attempts to import OpenCV and set a flag.
# If OpenCV is not installed, the relevant functions will gracefully print a message.
try:
    import cv2
    import numpy as np
    OPENCV_AVAILABLE = True
except ImportError:
    print("Warning: OpenCV not installed. Advanced edge detection and object detection functions will not work.")
    OPENCV_AVAILABLE = False

class ImageProcessor:
    def __init__(self, input_image_path):
        """
        Initializes the ImageProcessor with the path to the input image.
        """
        self.input_image_path = input_image_path
        # Use a flag to indicate if the input file exists, not the image object itself
        self.image_file_exists = os.path.exists(input_image_path)
        
        if not self.image_file_exists:
            print(f"Error: Input image not found at {input_image_path}")
        else:
            print(f"ImageProcessor initialized for: {input_image_path}")

    def _save_image(self, img_object, output_path, operation_name):
        """Helper to save the processed image."""
        try:
            img_object.save(output_path)
            print(f"{operation_name} applied and saved as {output_path}")
            return True
        except Exception as e:
            print(f"Error saving {operation_name} to {output_path}: {e}")
            return False

    def _load_font_for_drawing(self, font_size=30):
        """
        Helper function to robustly load a font for ImageDraw operations.
        Tries common system font paths before falling back to default.
        """
        font_to_use = None
        try:
            # Common font paths to try (add more if needed for specific systems/distros)
            font_paths_to_try = [
                "arial.ttf",                                # Default search on Windows
                "C:/Windows/Fonts/arial.ttf",               # Direct Windows path
                "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", # Common Linux path
                "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf", # Another common Linux path
                "/System/Library/Fonts/Supplemental/Arial.ttf", # Common macOS path
                "/Library/Fonts/Arial.ttf",                 # Another macOS path
            ]

            for f_path in font_paths_to_try:
                if os.path.exists(f_path):
                    font_to_use = ImageFont.truetype(f_path, font_size)
                    break

            if font_to_use is None: # If no truetype font found after trying common paths
                print("Warning: No common TrueType font found, using default Pillow font.")
                font_to_use = ImageFont.load_default()

        except IOError:
            print("Error loading TrueType font, using default Pillow font.")
            font_to_use = ImageFont.load_default()
        except Exception as e:
            print(f"Unexpected error loading font: {e}, using default Pillow font.")
            font_to_use = ImageFont.load_default()

        return font_to_use


    def convert_to_grayscale(self, output_path="output_grayscale.jpg"):
        """Converts the image to grayscale."""
        try:
            img = Image.open(self.input_image_path) # Reload to ensure original state
            grayscale_img = img.convert("L")  # "L" mode is for grayscale
            return self._save_image(grayscale_img, output_path, "Grayscale conversion")
        except FileNotFoundError:
            print(f"Error: Input image not found at {self.input_image_path}")
            return False
        except Exception as e:
            print(f"Error converting to grayscale: {e}")
            return False

    def apply_blur(self, output_path="output_blurred.jpg", radius=2):
        """Applies a Gaussian blur filter to the image."""
        try:
            img = Image.open(self.input_image_path)
            blurred_img = img.filter(ImageFilter.GaussianBlur(radius))
            return self._save_image(blurred_img, output_path, f"Blur filter (radius={radius})")
        except FileNotFoundError:
            print(f"Error: Input image not found at {self.input_image_path}")
            return False
        except Exception as e:
            print(f"Error applying blur: {e}")
            return False

    def apply_sharpen(self, output_path="output_sharpened.jpg"):
        """Applies a sharpen filter to the image."""
        try:
            img = Image.open(self.input_image_path)
            sharpened_img = img.filter(ImageFilter.SHARPEN)
            return self._save_image(sharpened_img, output_path, "Sharpen filter")
        except FileNotFoundError:
            print(f"Error: Input image not found at {self.input_image_path}")
            return False
        except Exception as e:
            print(f"Error applying sharpen: {e}")
            return False

    def detect_edges_pil(self, output_path="output_edges_pil.jpg"):
        """Detects edges using Pillow's FIND_EDGES filter."""
        try:
            img = Image.open(self.input_image_path)
            edges_img = img.filter(ImageFilter.FIND_EDGES)
            return self._save_image(edges_img, output_path, "Pillow Edge detection")
        except FileNotFoundError:
            print(f"Error: Input image not found at {self.input_image_path}")
            return False
        except Exception as e:
            print(f"Error detecting edges (Pillow): {e}")
            return False

    def detect_edges_opencv(self, output_path="output_edges_opencv.jpg", low_threshold=100, high_threshold=200):
        """
        Detects edges using OpenCV's Canny edge detector.
        Requires OpenCV to be installed.
        """
        if not OPENCV_AVAILABLE:
            print("OpenCV is not available. Cannot perform Canny edge detection.")
            return False
        try:
            img = cv2.imread(self.input_image_path)
            if img is None:
                raise FileNotFoundError(f"OpenCV could not load image at {self.input_image_path}")

            gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray_img, low_threshold, high_threshold)
            cv2.imwrite(output_path, edges) # OpenCV saves directly with cv2.imwrite
            print(f"Edges detected (OpenCV Canny) and saved as {output_path}")
            return True
        except FileNotFoundError as e:
            print(f"Error: {e}")
            return False
        except Exception as e:
            print(f"An error occurred during OpenCV edge detection: {e}")
            return False

    def resize_image(self, size=(300, 200), output_path="output_resized.jpg"):
        """Resizes the image to the specified dimensions."""
        try:
            img = Image.open(self.input_image_path)
            resized_img = img.resize(size)
            return self._save_image(resized_img, output_path, f"Image resized to {size}")
        except FileNotFoundError:
            print(f"Error: Input image not found at {self.input_image_path}")
            return False
        except Exception as e:
            print(f"Error resizing image: {e}")
            return False

    def crop_image(self, box=(100, 100, 400, 400), output_path="output_cropped.jpg"):
        """
        Crops the image.
        The box is a 4-tuple defining the left, upper, right, and lower pixel coordinate.
        """
        try:
            img = Image.open(self.input_image_path)
            cropped_img = img.crop(box)
            return self._save_image(cropped_img, output_path, f"Image cropped with box {box}")
        except FileNotFoundError:
            print(f"Error: Input image not found at {self.input_image_path}")
            return False
        except Exception as e:
            print(f"Error cropping image: {e}")
            return False

    def adjust_colors(self, brightness_factor=1.0, contrast_factor=1.0, color_factor=1.0, output_path="output_color_adjusted.jpg"):
        """
        Adjusts brightness, contrast, and color balance of the image.
        Factors > 1.0 increase, factors < 1.0 decrease.
        """
        try:
            img = Image.open(self.input_image_path)

            enhancer_brightness = ImageEnhance.Brightness(img)
            img = enhancer_brightness.enhance(brightness_factor)

            enhancer_contrast = ImageEnhance.Contrast(img)
            img = enhancer_contrast.enhance(contrast_factor)

            enhancer_color = ImageEnhance.Color(img)
            img = enhancer_color.enhance(color_factor)

            return self._save_image(img, output_path, "Color adjustments")
        except FileNotFoundError:
            print(f"Error: Input image not found at {self.input_image_path}")
            return False
        except Exception as e:
            print(f"Error adjusting colors: {e}")
            return False

    def add_text_watermark(self, text="WATERMARK", position=(50, 50), font_size=30,
                           font_color=(0, 0, 0, 128), output_path="output_watermarked.jpg"):
        """Adds a text watermark to the image."""
        try:
            img = Image.open(self.input_image_path).convert("RGBA") # Ensure alpha channel for transparency
            draw = ImageDraw.Draw(img)

            watermark_font = self._load_font_for_drawing(font_size)
            if watermark_font is None: # Fallback if font loading truly failed
                 print("Cannot add watermark: No usable font loaded.")
                 return False

            draw.text(position, text, font=watermark_font, fill=font_color)
            return self._save_image(img, output_path, "Text watermark")
        except FileNotFoundError:
            print(f"Error: Input image not found at {self.input_image_path}")
            return False
        except Exception as e:
            print(f"Error adding text watermark: {e}")
            return False

    def object_detection_placeholder(self):
        """
        A placeholder for object detection.
        This is a complex task requiring pre-trained machine learning models
        (e.g., from TensorFlow, PyTorch, or OpenCV's DNN module) and specific setup.
        """
        print("\n--- Object Detection (Conceptual) ---")
        print("Object detection is a more advanced task requiring pre-trained deep learning models and significant setup.")
        print("You would typically use libraries like OpenCV's DNN module, TensorFlow, or PyTorch for this.")
        print("This function serves as a reminder that such capabilities exist but are beyond a simple script function.")
        print("You'd need to download model weights and configuration files (e.g., YOLO, SSD models) to implement this.")
        print("-----------------------------------\n")
        return False

# --- Example Usage ---
if __name__ == "__main__":
    INPUT_IMAGE_NAME = "input_image.jpg"

    # Create a dummy image for testing if one doesn't exist
    if not os.path.exists(INPUT_IMAGE_NAME):
        print(f"Creating a dummy '{INPUT_IMAGE_NAME}' for demonstration purposes.")
        dummy_img = Image.new('RGB', (800, 600), color='lightgray')
        d = ImageDraw.Draw(dummy_img)

        # Load font for dummy image text
        dummy_font_size = 50
        dummy_font = None
        try:
            font_paths_to_try = [
                "arial.ttf",
                "C:/Windows/Fonts/arial.ttf",
                "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
                "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
                "/System/Library/Fonts/Supplemental/Arial.ttf",
                "/Library/Fonts/Arial.ttf",
            ]
            for f_path in font_paths_to_try:
                if os.path.exists(f_path):
                    dummy_font = ImageFont.truetype(f_path, dummy_font_size)
                    break
            
            if dummy_font is None:
                print("Warning: No common TrueType font found for dummy image, using default.")
                dummy_font = ImageFont.load_default()
        except IOError:
            print("Error loading TrueType font for dummy image, using default.")
            dummy_font = ImageFont.load_default()
        except Exception as e:
            print(f"Unexpected error loading font for dummy image: {e}, using default.")
            dummy_font = ImageFont.load_default()
            
        if dummy_font: # Only draw text if a font was successfully loaded
            d.text((20, 20), "Sample Image", fill=(0, 0, 0), font=dummy_font)
        else:
            print("Could not create text for dummy image (font issue).")

        d.rectangle((100, 100, 300, 300), fill="blue")
        d.ellipse((400, 200, 600, 400), fill="red")
        dummy_img.save(INPUT_IMAGE_NAME)
        print(f"Dummy '{INPUT_IMAGE_NAME}' created.")

    processor = ImageProcessor(INPUT_IMAGE_NAME)

    # Use the new flag to check if the input file exists
    if not processor.image_file_exists: # THIS LINE WAS MODIFIED
        print("Exiting: Input image could not be found or created.")
    else:
        print("\n--- Running Image Processing Examples ---")

        # 1. Grayscale
        processor.convert_to_grayscale("output_grayscale.jpg")

        # 2. Blur
        processor.apply_blur("output_blurred_radius_5.jpg", radius=5)

        # 3. Sharpen
        processor.apply_sharpen("output_sharpened.jpg")

        # 4. Edge Detection (Pillow)
        processor.detect_edges_pil("output_edges_pil.jpg")

        # 4b. Edge Detection (OpenCV Canny) - Only if OpenCV is available
        processor.detect_edges_opencv("output_edges_opencv.jpg")

        # 5. Resizing
        processor.resize_image(size=(400, 300), output_path="output_resized_400x300.jpg")
        processor.resize_image(size=(150, 150), output_path="output_resized_150x150.jpg")

        # 6. Cropping
        processor.crop_image(box=(100, 100, 700, 500), output_path="output_cropped_example.jpg")

        # 7. Color Adjustments
        processor.adjust_colors(brightness_factor=1.2, contrast_factor=0.8, color_factor=1.1, output_path="output_bright_low_contrast_high_color.jpg")
        processor.adjust_colors(brightness_factor=0.7, contrast_factor=1.5, output_path="output_dark_high_contrast.jpg")

        # 8. Add Watermark
        # Dynamic position for watermark (bottom right corner)
        try:
            # Re-open the image just to get its dimensions for dynamic placement
            temp_img = Image.open(processor.input_image_path)
            original_img_width, original_img_height = temp_img.size
            temp_img.close() # Close the temporary image object

            watermark_text = "CONFIDENTIAL DRAFT"
            font_size = 25
            
            # Use the helper to load the font for text size calculation too
            watermark_font_obj = processor._load_font_for_drawing(font_size)
            
            # Get text bounding box to calculate accurate position
            # Note: getbbox requires a font object. It returns (left, top, right, bottom)
            if watermark_font_obj:
                # Calculate text size using the font object
                # For Pillow 9.0.0+, use font.getbbox(text)
                # For older versions, font.getsize(text) might be used but is deprecated.
                try:
                    bbox = watermark_font_obj.getbbox(watermark_text)
                    text_width = bbox[2] - bbox[0]
                    text_height = bbox[3] - bbox[1]
                except AttributeError: # Fallback for very old Pillow versions if getbbox isn't available
                    print("Warning: getbbox not available, estimating text size. Consider updating Pillow.")
                    text_width, text_height = watermark_font_obj.getsize(watermark_text)
                
                padding = 15
                pos_x = original_img_width - text_width - padding
                pos_y = original_img_height - text_height - padding
                
                # Ensure positions are not negative
                pos_x = max(0, pos_x)
                pos_y = max(0, pos_y)

                processor.add_text_watermark(
                    text=watermark_text,
                    position=(pos_x, pos_y),
                    font_size=font_size,
                    font_color=(255, 0, 0, 150), # Red, semi-transparent
                    output_path="output_watermarked_confidential.jpg"
                )
            else:
                print("Skipping watermark: Font could not be loaded for text measurement.")
        except Exception as e:
            print(f"Could not add dynamic watermark: {e}")

        # 9. Object Detection (Conceptual)
        processor.object_detection_placeholder()

        print("\n--- All image processing examples completed ---")
        print(f"Check the 'output_*.jpg' files in the directory: {os.getcwd()}")