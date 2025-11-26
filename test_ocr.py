#!/usr/bin/env python3
"""
Simple test script for the ocr_image function.
Creates a test image with text and runs OCR on it.
"""
import os
from dotenv import load_dotenv
from PIL import Image, ImageDraw, ImageFont
from tools import ocr_image

# Load environment variables
load_dotenv()

def create_test_image():
    """Create a simple test image with text"""
    # Create a white image
    img = Image.new('RGB', (400, 200), color='white')
    draw = ImageDraw.Draw(img)
    
    # Add some text
    text = "Hello World!\nThis is a test image\nfor OCR testing."
    
    # Try to use a default font, fallback to basic if not available
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 24)
    except:
        font = ImageFont.load_default()
    
    # Draw text on image
    draw.text((20, 50), text, fill='black', font=font)
    
    # Save the image
    test_image_path = "test_ocr_image.png"
    img.save(test_image_path)
    print(f"Created test image: {test_image_path}")
    
    return test_image_path

def test_ocr():
    """Test the OCR function"""
    print("=" * 50)
    print("Testing OCR functionality")
    print("=" * 50)
    
    # Create test image
    image_path = create_test_image()
    
    # Run OCR
    print(f"\nRunning OCR on {image_path}...")
    result = ocr_image(image_path)
    
    print("\n" + "=" * 50)
    print("OCR Result:")
    print("=" * 50)
    print(result)
    print("=" * 50)
    
    # Cleanup
    if os.path.exists(image_path):
        os.remove(image_path)
        print(f"\nCleaned up test image: {image_path}")
    
    return result

if __name__ == "__main__":
    test_ocr()
