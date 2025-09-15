"""
Asset extraction API routes.

This module provides endpoints for extracting assets from screenshots using computer vision.
"""

import logging
from typing import Dict, List, Optional

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from asset_extraction.core import AssetExtractor, AssetExtractionError, ExtractionConfig

logger = logging.getLogger(__name__)

router = APIRouter()


class ExtractionRequest(BaseModel):
    """Request model for asset extraction."""
    
    screenshot_data_url: str = Field(..., description="Base64 encoded screenshot data URL")
    config: Optional[Dict] = Field(None, description="Optional extraction configuration parameters")


class ExtractionResponse(BaseModel):
    """Response model for asset extraction results."""
    
    success: bool = Field(..., description="Whether extraction was successful")
    message: str = Field(..., description="Status message")
    assets: List[Dict] = Field(default_factory=list, description="List of extracted assets")
    stats: Dict = Field(default_factory=dict, description="Extraction statistics")


@router.post("/extract", response_model=ExtractionResponse)
async def extract_assets(request: ExtractionRequest) -> ExtractionResponse:
    """
    Extract assets from a screenshot using computer vision.
    
    This endpoint analyzes a screenshot and identifies potential reusable assets
    like images, icons, and UI elements that can be cropped and used separately.
    
    Args:
        request: Extraction request containing screenshot data and optional config
        
    Returns:
        ExtractionResponse with extracted assets and metadata
        
    Raises:
        HTTPException: If extraction fails or invalid input is provided
    """
    try:
        # Parse extraction configuration
        config = ExtractionConfig()
        if request.config:
            # Update config with provided parameters
            for key, value in request.config.items():
                if hasattr(config, key):
                    setattr(config, key, value)
                else:
                    logger.warning(f"Unknown config parameter: {key}")
        
        # Initialize extractor
        extractor = AssetExtractor(config)
        
        # Extract assets
        logger.info("Starting asset extraction from screenshot")
        extracted_assets = extractor.extract_from_data_url(request.screenshot_data_url)
        
        # Convert assets to response format
        assets_data = []
        for asset in extracted_assets:
            asset_dict = asset.to_dict()
            # Include data URL for immediate use
            asset_dict["data_url"] = asset.to_data_url()
            assets_data.append(asset_dict)
        
        # Generate statistics
        stats = {
            "total_assets": len(extracted_assets),
            "asset_types": _count_asset_types(extracted_assets),
            "extraction_config": {
                "min_area": config.min_area,
                "max_assets": config.max_assets,
                "canny_low": config.canny_low,
                "canny_high": config.canny_high
            }
        }
        
        logger.info(f"Successfully extracted {len(extracted_assets)} assets")
        
        return ExtractionResponse(
            success=True,
            message=f"Successfully extracted {len(extracted_assets)} assets",
            assets=assets_data,
            stats=stats
        )
        
    except AssetExtractionError as e:
        logger.error(f"Asset extraction failed: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Asset extraction failed: {str(e)}")
    
    except Exception as e:
        logger.error(f"Unexpected error during asset extraction: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error during asset extraction")


@router.get("/config/defaults")
async def get_default_config() -> Dict:
    """
    Get the default extraction configuration parameters.
    
    Returns:
        Dictionary containing default configuration values and descriptions
    """
    config = ExtractionConfig()
    
    return {
        "config": {
            "min_area": {
                "value": config.min_area,
                "description": "Minimum area in pixels for detected assets",
                "type": "integer",
                "min": 100,
                "max": 10000
            },
            "max_assets": {
                "value": config.max_assets,
                "description": "Maximum number of assets to extract",
                "type": "integer",
                "min": 1,
                "max": 500
            },
            "canny_low": {
                "value": config.canny_low,
                "description": "Lower threshold for Canny edge detection",
                "type": "integer",
                "min": 10,
                "max": 200
            },
            "canny_high": {
                "value": config.canny_high,
                "description": "Upper threshold for Canny edge detection",
                "type": "integer",
                "min": 50,
                "max": 300
            },
            "nms_overlap_threshold": {
                "value": config.nms_overlap_threshold,
                "description": "Overlap threshold for non-maximum suppression",
                "type": "float",
                "min": 0.1,
                "max": 0.9
            },
            "merge_iou_threshold": {
                "value": config.merge_iou_threshold,
                "description": "IoU threshold for merging overlapping boxes",
                "type": "float",
                "min": 0.1,
                "max": 0.8
            }
        },
        "asset_types": [
            "image",
            "icon", 
            "ui_element"
        ],
        "supported_formats": [
            "image/png",
            "image/jpeg",
            "image/webp"
        ]
    }


@router.get("/health")
async def health_check() -> Dict:
    """
    Check if asset extraction dependencies are available.
    
    Returns:
        Health status and available features
    """
    try:
        # Test if we can create an extractor (checks OpenCV availability)
        AssetExtractor()
        opencv_available = True
        status = "healthy"
        message = "Asset extraction service is ready"
    except AssetExtractionError as e:
        opencv_available = False
        status = "unhealthy"
        message = str(e)
    
    # Check OCR availability
    try:
        import pytesseract
        pytesseract.get_tesseract_version()
        ocr_available = True
    except:
        ocr_available = False
    
    return {
        "status": status,
        "message": message,
        "dependencies": {
            "opencv": opencv_available,
            "numpy": opencv_available,  # If OpenCV works, numpy is also available
            "tesseract": ocr_available
        },
        "features": {
            "asset_detection": opencv_available,
            "edge_detection": opencv_available,
            "contour_analysis": opencv_available,
            "text_filtering": ocr_available,
            "ocr_text_detection": ocr_available
        }
    }


def _count_asset_types(assets) -> Dict[str, int]:
    """Count assets by type for statistics."""
    type_counts = {}
    for asset in assets:
        asset_type = asset.asset_type
        type_counts[asset_type] = type_counts.get(asset_type, 0) + 1
    return type_counts
