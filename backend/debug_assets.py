#!/usr/bin/env python3
"""
Simple script to analyze test images and debug asset extraction.
"""

import sys
import cv2
import numpy as np
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from asset_extraction.core import AssetExtractor, ExtractionConfig


def analyze_image(image_path: Path, name: str):
    """Analyze an image and print its characteristics."""
    print(f"\n=== Analyzing {name} ===")
    
    if not image_path.exists():
        print(f"Image not found: {image_path}")
        return None
    
    # Load image
    image = cv2.imread(str(image_path))
    if image is None:
        print(f"Failed to load: {image_path}")
        return None
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    h, w, c = image.shape
    
    print(f"Dimensions: {w}x{h} ({w*h} pixels)")
    print(f"Mean intensity: {np.mean(gray):.1f}")
    print(f"Intensity variance: {np.var(gray):.1f}")
    print(f"Intensity range: {np.min(gray)} - {np.max(gray)}")
    
    # Color analysis
    color_var = np.var(image, axis=(0, 1)).mean()
    print(f"Color variance: {color_var:.1f}")
    
    # Edge analysis
    edges = cv2.Canny(gray, 50, 150)
    edge_density = np.sum(edges > 0) / edges.size
    print(f"Edge density: {edge_density:.4f}")
    
    # Gradient analysis
    grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
    print(f"Mean gradient: {np.mean(gradient_magnitude):.1f}")
    
    return image


def test_extraction(image_path: Path, name: str):
    """Test asset extraction on an image."""
    print(f"\n=== Testing Extraction on {name} ===")
    
    # Load image and convert to data URL
    import base64
    with open(image_path, 'rb') as f:
        image_bytes = f.read()
    encoded = base64.b64encode(image_bytes).decode('utf-8')
    data_url = f"data:image/png;base64,{encoded}"
    
    # Test with different configurations
    configs = {
        "Current": ExtractionConfig(),
        "Lenient": ExtractionConfig(min_area=2000, max_assets=12, canny_low=25, canny_high=80),
        "Very Lenient": ExtractionConfig(min_area=1000, max_assets=15, canny_low=20, canny_high=60)
    }
    
    for config_name, config in configs.items():
        print(f"\n--- {config_name} Configuration ---")
        print(f"Min area: {config.min_area}, Max assets: {config.max_assets}")
        
        extractor = AssetExtractor(config)
        try:
            assets = extractor.extract_from_data_url(data_url)
            print(f"Extracted {len(assets)} assets:")
            
            for i, asset in enumerate(assets):
                area = asset.width * asset.height
                print(f"  {i+1}. {asset.asset_type}: {asset.width}x{asset.height} ({area} px) "
                      f"at ({asset.x}, {asset.y}) - conf: {asset.confidence:.3f}")
        
        except Exception as e:
            print(f"Extraction failed: {e}")


def main():
    """Main analysis function."""
    assets_dir = Path(__file__).parent / "tests" / "assets"
    
    test_files = [
        ("test1-subject.png", "Test1 Subject (Gradient)"),
        ("test1-target.png", "Test1 Target (Expected Gradient)"),
        ("test2-subject.png", "Test2 Subject (Photo)"),
        ("test2-target.png", "Test2 Target (Expected Photo)")
    ]
    
    print("🔍 ASSET EXTRACTION ANALYSIS")
    print("=" * 50)
    
    # Analyze each image
    for filename, description in test_files:
        image_path = assets_dir / filename
        analyze_image(image_path, description)
    
    print("\n\n🚀 EXTRACTION TESTING")
    print("=" * 50)
    
    # Test extraction on subject images
    for filename, description in [("test1-subject.png", "Test1 Subject"), ("test2-subject.png", "Test2 Subject")]:
        image_path = assets_dir / filename
        if image_path.exists():
            test_extraction(image_path, description)


if __name__ == "__main__":
    main()
