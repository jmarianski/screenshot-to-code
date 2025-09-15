"""
Asset extraction core functionality.

This module provides computer vision-based asset extraction from screenshots,
identifying and cropping potential UI elements, images, and other reusable components.
"""

import json
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import base64
import io

try:
    import numpy as np
    import cv2
    from PIL import Image
    CV_AVAILABLE = True
except ImportError:
    CV_AVAILABLE = False

try:
    import pytesseract
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False


class AssetExtractionError(Exception):
    """Raised when asset extraction fails."""
    pass


class ExtractedAsset:
    """Represents a single extracted asset."""
    
    def __init__(
        self,
        asset_id: str,
        image_data: bytes,
        x: int,
        y: int,
        width: int,
        height: int,
        confidence: float = 1.0,
        asset_type: str = "image"
    ):
        self.id = asset_id
        self.image_data = image_data
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.confidence = confidence
        self.asset_type = asset_type
    
    def to_dict(self) -> Dict:
        """Convert to dictionary representation."""
        return {
            "id": self.id,
            "x": self.x,
            "y": self.y,
            "width": self.width,
            "height": self.height,
            "confidence": self.confidence,
            "type": self.asset_type
        }
    
    def to_data_url(self) -> str:
        """Convert image data to base64 data URL."""
        # Convert bytes to PIL Image to ensure proper format
        img = Image.open(io.BytesIO(self.image_data))
        
        # Save as PNG for consistency
        img_buffer = io.BytesIO()
        img.save(img_buffer, format='PNG')
        img_buffer.seek(0)
        
        # Encode as base64
        img_base64 = base64.b64encode(img_buffer.getvalue()).decode('utf-8')
        return f"data:image/png;base64,{img_base64}"


class ExtractionConfig:
    """Configuration for asset extraction."""
    
    def __init__(
        self,
        min_area: int = 15000,  # Much higher - avoid small random regions
        max_assets: int = 3,    # Very conservative - only best assets
        canny_low: int = 60,    # Higher sensitivity for cleaner edges
        canny_high: int = 180,  # Higher sensitivity for cleaner edges
        morph_kernel_size: int = 7,
        nms_overlap_threshold: float = 0.3,  # Very aggressive NMS
        merge_iou_threshold: float = 0.2,    # Very aggressive merging
        merge_passes: int = 6,   # More merging passes
        generate_background: bool = False
    ):
        self.min_area = min_area
        self.max_assets = max_assets
        self.canny_low = canny_low
        self.canny_high = canny_high
        self.morph_kernel_size = morph_kernel_size
        self.nms_overlap_threshold = nms_overlap_threshold
        self.merge_iou_threshold = merge_iou_threshold
        self.merge_passes = merge_passes
        self.generate_background = generate_background


class AssetExtractor:
    """Main asset extraction class using computer vision techniques."""
    
    def __init__(self, config: Optional[ExtractionConfig] = None):
        if not CV_AVAILABLE:
            raise AssetExtractionError(
                "OpenCV and numpy are required for asset extraction. "
                "Install with: pip install opencv-python numpy"
            )
        
        self.config = config or ExtractionConfig()
    
    def extract_from_data_url(self, image_data_url: str) -> List[ExtractedAsset]:
        """
        Extract assets from a base64 data URL.
        
        Args:
            image_data_url: Base64 encoded image data URL
            
        Returns:
            List of extracted assets
        """
        # Parse data URL
        if not image_data_url.startswith('data:image/'):
            raise AssetExtractionError("Invalid image data URL format")
        
        try:
            # Extract base64 data
            base64_data = image_data_url.split(',')[1]
            image_bytes = base64.b64decode(base64_data)
            
            # Convert to numpy array for OpenCV
            nparr = np.frombuffer(image_bytes, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if image is None:
                raise AssetExtractionError("Failed to decode image data")
            
            return self._extract_assets(image)
            
        except Exception as e:
            raise AssetExtractionError(f"Failed to extract assets: {str(e)}")
    
    def _extract_assets(self, image: np.ndarray) -> List[ExtractedAsset]:
        """
        Extract assets from OpenCV image array.
        
        Args:
            image: OpenCV image array (BGR format)
            
        Returns:
            List of extracted assets
        """
        # Remove text from the image before any detection occurs
        image_without_text = self._remove_text_from_image(image)
        
        # Detect candidate regions on the text-removed image
        boxes = self._detect_candidates(image_without_text)
        
        # Crop and create asset objects
        assets = []
        for i, (x, y, w, h) in enumerate(boxes, start=1):
            # Crop the region from the text-removed image (not the original!)
            cropped = image_without_text[y:y+h, x:x+w]
            
            # Convert to bytes (PNG format)
            success, img_encoded = cv2.imencode('.png', cropped)
            if not success:
                continue
            
            # Create asset object
            asset = ExtractedAsset(
                asset_id=f"asset_{i:03d}",
                image_data=img_encoded.tobytes(),
                x=x,
                y=y,
                width=w,
                height=h,
                confidence=self._calculate_confidence(cropped),
                asset_type=self._classify_asset_type(cropped, w, h)
            )
            
            assets.append(asset)
        
        return assets
    
    def _remove_text_from_image(self, image: np.ndarray) -> np.ndarray:
        """
        Remove text regions from the image before asset detection.
        Uses OCR to detect text regions and inpaints them with surrounding content.
        
        Args:
            image: OpenCV image array (BGR format)
            
        Returns:
            Image with text regions removed/inpainted
        """
        if not OCR_AVAILABLE:
            # If OCR is not available, return original image
            return image.copy()
        
        try:
            # Convert to RGB for pytesseract
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(image_rgb)
            
            # Get detailed text detection data
            data = pytesseract.image_to_data(pil_image, output_type=pytesseract.Output.DICT)
            
            # Create a mask for text regions
            mask = np.zeros(image.shape[:2], dtype=np.uint8)
            
            # Mark text regions in the mask
            for i in range(len(data['text'])):
                confidence = int(data['conf'][i])
                text = data['text'][i].strip()
                
                # Only process high-confidence text detections
                if confidence > 30 and len(text) > 0:
                    x = data['left'][i]
                    y = data['top'][i]
                    w = data['width'][i]
                    h = data['height'][i]
                    
                    # Expand the region slightly to ensure complete text removal
                    padding = 2
                    x = max(0, x - padding)
                    y = max(0, y - padding)
                    w = min(image.shape[1] - x, w + 2 * padding)
                    h = min(image.shape[0] - y, h + 2 * padding)
                    
                    # Mark this region in the mask
                    cv2.rectangle(mask, (x, y), (x + w, y + h), 255, -1)
            
            # If no text regions found, return original image
            if np.sum(mask) == 0:
                return image.copy()
            
            # Use OpenCV inpainting to fill text regions
            # We'll use TELEA inpainting algorithm which works well for removing text
            result = cv2.inpaint(image, mask, 3, cv2.INPAINT_TELEA)
            
            return result
            
        except Exception as e:
            # If anything fails, return the original image
            print(f"Warning: Text removal failed: {e}")
            return image.copy()
    
    def _detect_candidates(self, image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """
        Detect candidate asset regions using computer vision.
        Focus on visual assets that can't be easily recreated with HTML/CSS.
        
        Args:
            image: OpenCV image array
            
        Returns:
            List of bounding boxes as (x, y, width, height) tuples
        """
        gray = self._to_grayscale(image)
        h, w = gray.shape[:2]
        
        # Multi-approach detection for different types of visual assets
        candidates = []
        
        # 1. Detect image-like regions with rich visual content
        candidates.extend(self._detect_image_regions(gray, image))
        
        # 2. Detect icons and small visual elements
        candidates.extend(self._detect_icon_regions(gray, image))
        
        # 3. Detect complex visual patterns (logos, graphics)
        candidates.extend(self._detect_complex_patterns(gray, image))
        
        # Filter and validate candidates with stricter criteria for gradient images
        validated_boxes = []
        for x, y, width, height in candidates:
            # Ensure box is within image bounds
            x = max(0, x)
            y = max(0, y)
            width = min(w - x, width)
            height = min(h - y, height)
            
            # Much stricter size constraints for cleaner extraction
            area = width * height
            min_gradient_area = max(self.config.min_area, 50000)  # Minimum 50k pixels for gradients
            if area < min_gradient_area or area > (w * h * 0.8):  # Allow larger regions
                continue
                
            # Skip very thin/wide rectangles (likely UI borders/text lines)
            aspect_ratio = width / height
            if aspect_ratio > 6 or aspect_ratio < 0.2:  # More lenient for gradients
                continue
                
            # Require substantial size - no small fragments
            if width < 200 or height < 150:  # Much larger minimum dimensions
                continue
            
            # Validate that this region contains actual visual content and is absolutely not text-heavy
            region = image[y:y+height, x:x+width]
            if (self._is_visual_asset(region) and 
                not self._is_text_heavy(region) and 
                not self._contains_any_text(region)):  # Double-check for text
                validated_boxes.append((x, y, width, height))
        
        # Apply aggressive non-maximum suppression to merge overlapping regions
        validated_boxes = self._apply_aggressive_nms(validated_boxes)
        
        # Merge overlapping boxes more aggressively
        validated_boxes = self._merge_overlapping_boxes_aggressive(validated_boxes)
        
        # Sort by area (largest first) and limit count to 2 for gradient images
        validated_boxes = sorted(validated_boxes, key=lambda b: b[2] * b[3], reverse=True)
        validated_boxes = validated_boxes[:min(2, self.config.max_assets)]
        
        return validated_boxes
    
    def _detect_image_regions(self, gray: np.ndarray, color_image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Detect regions that look like images (rich texture, gradients, photos)."""
        boxes = []
        h, w = gray.shape
        
        # Method 1: Direct large region detection for photos
        # Look for rectangular regions with good photographic characteristics
        step_size = 100  # Much coarser sampling for efficiency
        
        # Only test a few key sizes that match our target photo (~363x229)
        test_sizes = [(350, 220), (400, 250), (300, 200), (500, 300)]
        
        for w_region, h_region in test_sizes:
            if h_region > h or w_region > w:
                continue
            for y in range(0, h - h_region + 1, step_size):
                for x in range(0, w - w_region + 1, step_size):
                    region = gray[y:y+h_region, x:x+w_region]
                    if self._has_photographic_characteristics(region):
                        # Check if it's not mostly text
                        color_region = color_image[y:y+h_region, x:x+w_region]
                        if not self._is_text_heavy(color_region):
                            boxes.append((x, y, w_region, h_region))
        
        # Method 2: Local variance for textured regions
        kernel_size = 15  # Larger kernel for better photo detection
        kernel = np.ones((kernel_size, kernel_size), np.float32) / (kernel_size * kernel_size)
        mean = cv2.filter2D(gray.astype(np.float32), -1, kernel)
        sqr_mean = cv2.filter2D((gray.astype(np.float32))**2, -1, kernel)
        variance = sqr_mean - mean**2
        
        # Use lower percentile threshold to catch more subtle images
        threshold = np.percentile(variance, 75)  # Lower threshold
        _, high_variance = cv2.threshold(variance, threshold, 255, cv2.THRESH_BINARY)
        high_variance = high_variance.astype(np.uint8)
        
        # Morphological operations to connect nearby high-variance regions
        kernel_morph = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 10))
        high_variance = cv2.morphologyEx(high_variance, cv2.MORPH_CLOSE, kernel_morph)
        high_variance = cv2.morphologyEx(high_variance, cv2.MORPH_OPEN, kernel_morph)
        
        contours, _ = cv2.findContours(high_variance, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            x, y, width, height = cv2.boundingRect(contour)
            area = width * height
            
            # Look for substantial regions that could be photos/images
            if area > 8000 and area < (w * h * 0.7):  # Lower minimum, higher maximum
                boxes.append((x, y, width, height))
        
        # Method 2: Color clustering for photos (if color image)
        if len(color_image.shape) == 3:
            # Convert to LAB color space for better color analysis
            lab = cv2.cvtColor(color_image, cv2.COLOR_BGR2LAB)
            
            # Calculate color variance in sliding windows
            window_size = 64
            stride = 32
            
            for y_start in range(0, h - window_size, stride):
                for x_start in range(0, w - window_size, stride):
                    window = lab[y_start:y_start+window_size, x_start:x_start+window_size]
                    
                    # Check if this window has photographic characteristics
                    # Convert LAB window back to BGR first, then to grayscale
                    window_bgr = cv2.cvtColor(window, cv2.COLOR_Lab2BGR)
                    window_gray = cv2.cvtColor(window_bgr, cv2.COLOR_BGR2GRAY)
                    if self._has_photographic_characteristics(window_gray):
                        # Expand the window to find the full region
                        expanded = self._expand_photographic_region(image, x_start, y_start, window_size)
                        if expanded:
                            boxes.append(expanded)
        
        # Method 4: MSER (Maximally Stable Extremal Regions) for blob detection
        try:
            mser = cv2.MSER_create(
                _min_area=8000,      # Minimum area for detected regions
                _max_area=w*h//3,    # Maximum area
                _delta=8,            # Delta for MSER
            )
            regions, _ = mser.detectRegions(gray)
            
            for region in regions:
                if len(region) > 100:  # Sufficient points to form a meaningful region
                    x, y, width, height = cv2.boundingRect(region)
                    area = width * height
                    if 8000 < area < (w * h * 0.6):
                        boxes.append((x, y, width, height))
        except:
            pass  # MSER might not be available in all OpenCV builds
        
        return boxes
    
    def _expand_photographic_region(self, lab_image: np.ndarray, start_x: int, start_y: int, initial_size: int) -> Optional[Tuple[int, int, int, int]]:
        """Expand a seed region to find the full photographic area."""
        h, w = lab_image.shape[:2]
        
        # Simple region growing based on color similarity
        visited = np.zeros((h, w), dtype=bool)
        
        # Get seed characteristics
        seed_region = lab_image[start_y:start_y+initial_size, start_x:start_x+initial_size]
        seed_mean = np.mean(seed_region, axis=(0, 1))
        seed_std = np.std(seed_region, axis=(0, 1))
        
        # Flood fill with color tolerance
        def is_similar(pixel):
            diff = np.abs(pixel - seed_mean)
            return np.all(diff < 2.5 * seed_std + 20)  # Adaptive threshold
        
        # Simple expansion (could be improved with proper flood fill)
        min_x, max_x = start_x, start_x + initial_size
        min_y, max_y = start_y, start_y + initial_size
        
        # Expand bounds by checking neighboring pixels
        expansion_step = 16
        for _ in range(5):  # Max 5 expansion steps
            expanded = False
            
            # Try expanding left
            if min_x - expansion_step >= 0:
                test_region = lab_image[min_y:max_y, min_x-expansion_step:min_x]
                if np.mean([is_similar(pixel) for pixel in test_region.reshape(-1, 3)]) > 0.6:
                    min_x -= expansion_step
                    expanded = True
            
            # Try expanding right  
            if max_x + expansion_step < w:
                test_region = lab_image[min_y:max_y, max_x:max_x+expansion_step]
                if np.mean([is_similar(pixel) for pixel in test_region.reshape(-1, 3)]) > 0.6:
                    max_x += expansion_step
                    expanded = True
            
            # Try expanding up
            if min_y - expansion_step >= 0:
                test_region = lab_image[min_y-expansion_step:min_y, min_x:max_x]
                if np.mean([is_similar(pixel) for pixel in test_region.reshape(-1, 3)]) > 0.6:
                    min_y -= expansion_step
                    expanded = True
            
            # Try expanding down
            if max_y + expansion_step < h:
                test_region = lab_image[max_y:max_y+expansion_step, min_x:max_x]
                if np.mean([is_similar(pixel) for pixel in test_region.reshape(-1, 3)]) > 0.6:
                    max_y += expansion_step
                    expanded = True
            
            if not expanded:
                break
        
        # Return expanded region if it's substantial
        area = (max_x - min_x) * (max_y - min_y)
        if area > 15000:  # Substantial photo-like region
            return (min_x, min_y, max_x - min_x, max_y - min_y)
        
        return None
    
    def _detect_icon_regions(self, gray: np.ndarray, color_image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Detect small icon-like regions."""
        # Use template matching or feature detection for common icon patterns
        # For now, use edge density to find icon-like regions
        
        edges = cv2.Canny(gray, 50, 150)
        
        # Create sliding windows to find regions with high edge density
        window_size = 32
        stride = 16
        boxes = []
        
        h, w = gray.shape
        for y in range(0, h - window_size, stride):
            for x in range(0, w - window_size, stride):
                window = edges[y:y+window_size, x:x+window_size]
                edge_density = np.sum(window > 0) / (window_size * window_size)
                
                # If edge density is high but not too high (not noise), it might be an icon
                if 0.1 < edge_density < 0.5:
                    # Check if it's roughly square (icon-like)
                    boxes.append((x, y, window_size, window_size))
        
        return boxes
    
    def _detect_complex_patterns(self, gray: np.ndarray, color_image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Detect complex visual patterns like logos, graphics."""
        # Use corner detection to find regions with many features
        corners = cv2.cornerHarris(gray.astype(np.float32), 2, 3, 0.04)
        corners = cv2.normalize(corners, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        
        # Threshold corner response
        _, corner_thresh = cv2.threshold(corners, np.percentile(corners, 95), 255, cv2.THRESH_BINARY)
        
        # Find regions with high corner density
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 20))
        dense_corners = cv2.morphologyEx(corner_thresh, cv2.MORPH_CLOSE, kernel)
        
        contours, _ = cv2.findContours(dense_corners, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        boxes = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            area = w * h
            
            # Look for medium-sized regions with complex patterns
            if 1000 < area < (gray.shape[0] * gray.shape[1] * 0.3):
                boxes.append((x, y, w, h))
        
        return boxes
    
    def _is_visual_asset(self, region: np.ndarray) -> bool:
        """Determine if a region contains actual visual content worth extracting.
        Much more restrictive to avoid random background regions."""
        if region.size == 0:
            return False
        
        gray = self._to_grayscale(region)
        h, w = region.shape[:2]
        area = h * w
        
        # First, check if this region is primarily text - if so, reject it
        if self._is_text_heavy(region):
            return False
        
        # MUCH stricter size requirements - avoid tiny snippets and huge backgrounds
        if area < 15000 or area > 300000:  # More restrictive size range
            return False
        
        # Reject extremely elongated regions (likely text areas or UI elements)
        aspect_ratio = max(h, w) / min(h, w)
        if aspect_ratio > 3.0:  # Much stricter aspect ratio
            return False
        
        # Require MUCH higher visual complexity
        complexity_threshold = 0.6 if area < 30000 else 0.5  # Much higher thresholds
        visual_complexity = self._calculate_visual_complexity(region)
        
        if visual_complexity < complexity_threshold:
            return False
        
        # Check for sufficient visual content indicators - ALL must be strong
        visual_scores = []
        
        # 1. Color variety (if color image) - stricter threshold
        if len(region.shape) == 3 and region.shape[2] == 3:
            color_var = np.var(region, axis=(0, 1)).mean()
            visual_scores.append(min(color_var / 500, 1.0))  # Higher threshold
        
        # 2. Texture complexity - much stricter
        variance = np.var(gray)
        visual_scores.append(min(variance / 1500, 1.0))  # Much higher threshold
        
        # 3. Edge density - but not too much (not text)
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / edges.size
        if 0.03 < edge_density < 0.25:  # Tighter sweet spot
            visual_scores.append(edge_density * 10)  # Scale for scoring
        else:
            visual_scores.append(0)  # Fail if outside range
        
        # 4. Gradient magnitude - higher threshold
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        mean_gradient = np.mean(gradient_magnitude)
        visual_scores.append(min(mean_gradient / 30, 1.0))  # Higher threshold
        
        # 5. Check for distinct shapes/objects
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        meaningful_contours = 0
        for contour in contours:
            contour_area = cv2.contourArea(contour)
            if contour_area > 300:  # Larger minimum contour area
                meaningful_contours += 1
        
        if meaningful_contours < 2:  # Need multiple distinct shapes
            return False
        
        # 6. Check for photographic characteristics with stricter requirements
        has_photo_chars = self._has_photographic_characteristics(gray)
        if has_photo_chars:
            visual_scores.append(1.0)
        else:
            visual_scores.append(0.3)  # Partial score if not clearly photographic
        
        # Require STRONG visual content - average score must be high
        if not visual_scores:
            return False
        
        avg_visual_score = sum(visual_scores) / len(visual_scores)
        return avg_visual_score > 0.7  # Much higher threshold for acceptance
    
    def _has_photographic_characteristics(self, gray: np.ndarray) -> bool:
        """Check if region has characteristics typical of photographs."""
        if gray.size < 5000:  # Too small to be meaningful photo
            return False
        
        # Photos typically have:
        # 1. High variance (rich detail, not flat)
        variance = np.var(gray)
        if variance < 1000:  # Relaxed lower bound for more detail
            return False
        
        # 2. Good dynamic range (contrast)
        intensity_range = np.max(gray) - np.min(gray)
        if intensity_range < 80:  # Good contrast
            return False
        
        # 3. Edge density indicates detail
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / edges.size
        if edge_density < 0.01:  # Has some detail/edges
            return False
        
        return True
    
    def _is_text_heavy(self, region: np.ndarray) -> bool:
        """Determine if a region contains primarily text content."""
        if not OCR_AVAILABLE or region.size == 0:
            # Fallback to heuristic-based text detection if OCR not available
            return self._is_text_heavy_heuristic(region)
        
        try:
            # Convert to PIL Image for pytesseract
            if len(region.shape) == 3:
                # Convert BGR to RGB for PIL
                region_rgb = cv2.cvtColor(region, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(region_rgb)
            else:
                pil_image = Image.fromarray(region)
            
            # Use pytesseract to detect text
            # Get confidence scores for detected text
            data = pytesseract.image_to_data(pil_image, output_type=pytesseract.Output.DICT)
            
            # Calculate text coverage
            confidences = [int(conf) for conf in data['conf'] if int(conf) > 30]  # Filter low-confidence detections
            
            if not confidences:
                return False
            
            # Calculate area covered by text
            total_area = region.shape[0] * region.shape[1]
            text_area = 0
            
            for i, conf in enumerate(data['conf']):
                if int(conf) > 30:  # Only consider high-confidence text
                    w = data['width'][i]
                    h = data['height'][i]
                    text_area += w * h
            
            # If text covers more than 40% of the region, consider it text-heavy
            text_coverage = text_area / total_area
            
            # Also check if there's significant amount of readable text
            text_content = pytesseract.image_to_string(pil_image, config='--psm 6').strip()
            word_count = len(text_content.split())
            
            # Consider text-heavy if:
            # 1. High text coverage (>40%) OR
            # 2. Many words detected (>8 words) OR  
            # 3. High average confidence with reasonable coverage (>20%)
            is_text_heavy = (
                text_coverage > 0.4 or
                word_count > 8 or
                (text_coverage > 0.2 and len(confidences) > 0 and sum(confidences) / len(confidences) > 60)
            )
            
            return is_text_heavy
            
        except Exception:
            # Fall back to heuristic if OCR fails
            return self._is_text_heavy_heuristic(region)
    
    def _is_text_heavy_heuristic(self, region: np.ndarray) -> bool:
        """Heuristic-based text detection when OCR is not available."""
        gray = self._to_grayscale(region)
        
        # Text typically has:
        # 1. Horizontal line patterns (for text rows)
        # 2. Regular spacing
        # 3. Moderate edge density in horizontal direction
        # 4. Low variance in vertical strips (consistent character width)
        
        # Check for horizontal line patterns
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 1))
        horizontal_lines = cv2.morphologyEx(gray, cv2.MORPH_OPEN, horizontal_kernel)
        horizontal_score = np.sum(horizontal_lines > 0) / horizontal_lines.size
        
        # Check for regular vertical patterns (character spacing)
        edges = cv2.Canny(gray, 50, 150)
        
        # Analyze horizontal projection to detect text lines
        horizontal_proj = np.sum(edges, axis=1)
        
        # Text typically has regular peaks in horizontal projection
        if len(horizontal_proj) > 10:
            # Look for multiple peaks (text lines) - simple implementation without scipy
            try:
                # Simple peak detection: find local maxima
                threshold = np.max(horizontal_proj) * 0.1
                peaks = []
                for i in range(1, len(horizontal_proj) - 1):
                    if (horizontal_proj[i] > horizontal_proj[i-1] and 
                        horizontal_proj[i] > horizontal_proj[i+1] and 
                        horizontal_proj[i] > threshold):
                        peaks.append(i)
                
                peak_regularity = len(peaks) / len(horizontal_proj) if len(horizontal_proj) > 0 else 0
            except:
                peak_regularity = 0
        else:
            peak_regularity = 0
        
        # Combine metrics
        text_likelihood = horizontal_score * 0.6 + peak_regularity * 0.4
        
        # Consider text-heavy if likelihood > 0.15
        return text_likelihood > 0.15
    
    def _contains_any_text(self, region: np.ndarray) -> bool:
        """Smart text detection that avoids false positives on gradients."""
        if not OCR_AVAILABLE or region.size == 0:
            return False  # If no OCR, assume no text (better than false positive)
        
        try:
            # Convert to PIL Image for pytesseract
            if len(region.shape) == 3:
                # Convert BGR to RGB for PIL
                region_rgb = cv2.cvtColor(region, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(region_rgb)
            else:
                pil_image = Image.fromarray(region)
            
            # Use moderate confidence threshold to avoid gradient false positives
            data = pytesseract.image_to_data(pil_image, output_type=pytesseract.Output.DICT)
            
            # Only consider high-confidence text detections (>40) to avoid gradient noise
            high_conf_text_found = False
            for i, conf in enumerate(data['conf']):
                if int(conf) > 40:  # Higher threshold to avoid false positives
                    text = data['text'][i].strip()
                    # Only consider meaningful text (not single characters or noise)
                    if len(text) >= 2 and text.isalnum():  # At least 2 alphanumeric characters
                        high_conf_text_found = True
                        break
            
            if high_conf_text_found:
                return True
            
            # Also check raw text extraction with strict filtering
            text_content = pytesseract.image_to_string(pil_image, config='--psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789 ').strip()
            # Only consider it text if it has multiple meaningful words
            words = [word for word in text_content.split() if len(word) >= 2]
            if len(words) >= 2:  # At least 2 words of 2+ characters
                return True
                
            return False
            
        except Exception:
            # If OCR fails, assume no text (safer for gradients)
            return False
    
    def _calculate_visual_complexity(self, region: np.ndarray) -> float:
        """Calculate visual complexity score for prioritizing assets."""
        if region.size == 0:
            return 0.0
        
        gray = self._to_grayscale(region)
        
        # Combine multiple complexity metrics
        complexity_score = 0.0
        
        # 1. Variance (texture)
        variance = np.var(gray)
        complexity_score += min(variance / 1000.0, 1.0) * 0.3
        
        # 2. Edge density
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / edges.size
        complexity_score += min(edge_density * 10, 1.0) * 0.3
        
        # 3. Color variety (if color image)
        if len(region.shape) == 3 and region.shape[2] == 3:
            color_var = np.var(region, axis=(0, 1)).mean()
            complexity_score += min(color_var / 1000.0, 1.0) * 0.4
        
        return complexity_score
    
    def _to_grayscale(self, image: np.ndarray) -> np.ndarray:
        """Convert image to grayscale."""
        if len(image.shape) == 2:
            return image
        elif image.shape[2] == 4:
            # BGRA to BGR, then to grayscale
            bgr = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
            return cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        else:
            # BGR to grayscale
            return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    def _apply_nms(self, boxes: List[Tuple[int, int, int, int]]) -> List[Tuple[int, int, int, int]]:
        """Apply non-maximum suppression to remove overlapping boxes."""
        if not boxes:
            return []
        
        boxes_array = np.array(boxes, dtype=float)
        x1 = boxes_array[:, 0]
        y1 = boxes_array[:, 1]
        x2 = boxes_array[:, 0] + boxes_array[:, 2]
        y2 = boxes_array[:, 1] + boxes_array[:, 3]
        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        
        # Sort by bottom coordinate
        order = np.argsort(y2)
        
        keep = []
        while len(order) > 0:
            i = order[-1]
            keep.append(i)
            
            # Calculate intersection
            xx1 = np.maximum(x1[i], x1[order[:-1]])
            yy1 = np.maximum(y1[i], y1[order[:-1]])
            xx2 = np.minimum(x2[i], x2[order[:-1]])
            yy2 = np.minimum(y2[i], y2[order[:-1]])
            
            w = np.maximum(0, xx2 - xx1 + 1)
            h = np.maximum(0, yy2 - yy1 + 1)
            intersection = w * h
            
            # Calculate overlap ratio
            overlap = intersection / areas[order[:-1]]
            
            # Keep boxes with low overlap
            order = order[np.where(overlap <= self.config.nms_overlap_threshold)[0]]
        
        return [tuple(map(int, boxes[i])) for i in keep]
    
    def _merge_overlapping_boxes(self, boxes: List[Tuple[int, int, int, int]]) -> List[Tuple[int, int, int, int]]:
        """Merge boxes that have high IoU overlap."""
        def calculate_iou(box1: Tuple[int, int, int, int], box2: Tuple[int, int, int, int]) -> float:
            x1, y1, w1, h1 = box1
            x2, y2, w2, h2 = box2
            
            # Calculate intersection
            inter_x1 = max(x1, x2)
            inter_y1 = max(y1, y2)
            inter_x2 = min(x1 + w1, x2 + w2)
            inter_y2 = min(y1 + h1, y2 + h2)
            
            inter_w = max(0, inter_x2 - inter_x1)
            inter_h = max(0, inter_y2 - inter_y1)
            intersection = inter_w * inter_h
            
            if intersection == 0:
                return 0.0
            
            # Calculate union
            area1 = w1 * h1
            area2 = w2 * h2
            union = area1 + area2 - intersection
            
            return intersection / max(1.0, union)
        
        current_boxes = [list(box) for box in boxes]
        
        for _ in range(self.config.merge_passes):
            merged_boxes = []
            used = [False] * len(current_boxes)
            
            for i in range(len(current_boxes)):
                if used[i]:
                    continue
                
                current_box = current_boxes[i]
                
                # Check for merges with remaining boxes
                for j in range(i + 1, len(current_boxes)):
                    if used[j]:
                        continue
                    
                    if calculate_iou(current_box, current_boxes[j]) >= self.config.merge_iou_threshold:
                        # Merge boxes by taking union
                        other_box = current_boxes[j]
                        x1 = min(current_box[0], other_box[0])
                        y1 = min(current_box[1], other_box[1])
                        x2 = max(current_box[0] + current_box[2], other_box[0] + other_box[2])
                        y2 = max(current_box[1] + current_box[3], other_box[1] + other_box[3])
                        
                        current_box = [x1, y1, x2 - x1, y2 - y1]
                        used[j] = True
                
                merged_boxes.append(current_box)
                used[i] = True
            
            current_boxes = merged_boxes
        
        return [tuple(map(int, box)) for box in current_boxes]
    
    def _apply_aggressive_nms(self, boxes: List[Tuple[int, int, int, int]]) -> List[Tuple[int, int, int, int]]:
        """Apply more aggressive non-maximum suppression for gradient images."""
        if len(boxes) <= 2:
            return boxes
        
        # Convert to areas for sorting
        areas = [(x, y, w, h, w * h) for x, y, w, h in boxes]
        # Sort by area (largest first)
        areas.sort(key=lambda x: x[4], reverse=True)
        
        # Keep track of which boxes to keep
        keep = []
        used = set()
        
        for i, (x1, y1, w1, h1, area1) in enumerate(areas):
            if i in used:
                continue
                
            keep.append((x1, y1, w1, h1))
            
            # Mark overlapping smaller boxes as used
            for j, (x2, y2, w2, h2, area2) in enumerate(areas[i+1:], i+1):
                if j in used:
                    continue
                
                # Calculate overlap
                overlap_x = max(0, min(x1 + w1, x2 + w2) - max(x1, x2))
                overlap_y = max(0, min(y1 + h1, y2 + h2) - max(y1, y2))
                overlap_area = overlap_x * overlap_y
                
                # If there's significant overlap (>30%), remove the smaller box
                if overlap_area > 0.3 * min(area1, area2):
                    used.add(j)
        
        return keep[:2]  # Maximum 2 assets for gradients
    
    def _merge_overlapping_boxes_aggressive(self, boxes: List[Tuple[int, int, int, int]]) -> List[Tuple[int, int, int, int]]:
        """Aggressively merge overlapping boxes for gradient images."""
        if len(boxes) <= 1:
            return boxes
        
        merged = []
        used = [False] * len(boxes)
        
        for i in range(len(boxes)):
            if used[i]:
                continue
            
            x1, y1, w1, h1 = boxes[i]
            
            # Find all boxes that overlap with this one
            to_merge = [(x1, y1, w1, h1)]
            used[i] = True
            
            for j in range(i + 1, len(boxes)):
                if used[j]:
                    continue
                
                x2, y2, w2, h2 = boxes[j]
                
                # Check for any overlap (even small)
                overlap_x = max(0, min(x1 + w1, x2 + w2) - max(x1, x2))
                overlap_y = max(0, min(y1 + h1, y2 + h2) - max(y1, y2))
                
                if overlap_x > 0 and overlap_y > 0:
                    to_merge.append((x2, y2, w2, h2))
                    used[j] = True
            
            # Merge all overlapping boxes into one large box
            if len(to_merge) > 1:
                min_x = min(x for x, y, w, h in to_merge)
                min_y = min(y for x, y, w, h in to_merge)
                max_x = max(x + w for x, y, w, h in to_merge)
                max_y = max(y + h for x, y, w, h in to_merge)
                
                merged.append((min_x, min_y, max_x - min_x, max_y - min_y))
            else:
                merged.append(to_merge[0])
        
        return merged[:2]  # Maximum 2 assets
    
    def _calculate_confidence(self, cropped_image: np.ndarray) -> float:
        """Calculate confidence score for extracted asset."""
        # Simple heuristic based on image properties
        gray = self._to_grayscale(cropped_image)
        
        # Calculate variance (higher variance = more interesting content)
        variance = np.var(gray)
        
        # Normalize to 0-1 range (rough heuristic)
        confidence = min(1.0, variance / 10000.0)
        
        return max(0.1, confidence)  # Minimum confidence of 0.1
    
    def _classify_asset_type(self, cropped_image: np.ndarray, width: int, height: int) -> str:
        """Classify the type of extracted asset based on visual characteristics."""
        gray = self._to_grayscale(cropped_image)
        aspect_ratio = width / height
        area = width * height
        
        # Analyze visual content
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / edges.size
        variance = np.var(gray)
        
        # Color analysis if available
        has_color_variety = False
        if len(cropped_image.shape) == 3 and cropped_image.shape[2] == 3:
            color_var = np.var(cropped_image, axis=(0, 1)).mean()
            has_color_variety = color_var > 100
        
        # Classification logic based on visual characteristics
        
        # Icons: small, square-ish, moderate edge density, often simple
        if (area < 8000 and 0.4 <= aspect_ratio <= 2.5 and 
            0.02 < edge_density < 0.3 and variance > 200):
            return "icon"
        
        # Logos: medium size, moderate complexity, often with color
        elif (1000 < area < 20000 and 0.3 <= aspect_ratio <= 3.0 and
              (has_color_variety or edge_density > 0.05)):
            return "logo"
        
        # Photos/images: larger, high texture variation, often rectangular
        elif (area > 5000 and variance > 800 and 
              (has_color_variety or edge_density > 0.03)):
            return "image"
        
        # Graphics/illustrations: varied sizes, high edge density or color variety
        elif ((edge_density > 0.1 or has_color_variety) and area > 2000):
            return "graphic"
        
        # Charts/diagrams: often rectangular, moderate complexity
        elif (area > 10000 and (aspect_ratio > 1.5 or aspect_ratio < 0.67) and
              edge_density > 0.02):
            return "chart"
        
        # Default to image if it passed our visual asset test
        return "image"
