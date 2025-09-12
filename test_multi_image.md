# Multi-Image Upload with Asset Tokens

## What was implemented:

### 1. **Token-Based Asset System** (Performance Optimization):
   - AI uses short tokens like `ASSET_IMAGE_1`, `ASSET_IMAGE_2` instead of full base64 data
   - Post-processing replaces tokens with actual data URLs
   - **Massive performance improvement** - prompts are now much shorter and faster

### 2. **System Prompts Enhanced**: Updated all multi-image system prompts to instruct the AI:
   - Use asset tokens (e.g., `ASSET_IMAGE_1`) in src attributes  
   - DO NOT use filenames or descriptions in src
   - Example: `<img src="ASSET_IMAGE_1" alt="Logo" />`

### 3. **Instruction Text Optimized**: Modified `create_multi_image_instruction()` to:
   - Provide asset tokens instead of full data URLs
   - Give clear mapping: "1. Company Logo → Token: ASSET_IMAGE_1"
   - Much cleaner and faster prompt generation

### 4. **Post-Processing Pipeline**: Added `replace_asset_tokens()` to all generation stages:
   - `ParallelGenerationStage` - for normal multi-variant generation
   - `MockResponseStage` - for testing/development
   - `VideoGenerationStage` - for video-based generation
   - Replaces tokens with actual base64 data URLs after AI generation

## How to test:

1. Upload multiple images (screenshot + assets like logos/photos)
2. Designate one as main screenshot, others as assets
3. Add descriptions to asset files  
4. Generate code and verify:
   - AI generation is much faster (shorter prompts)
   - Generated HTML has proper base64 data URLs in src attributes
   - No token strings remain in final output

## Expected behavior:

**AI receives**: `<img src="ASSET_IMAGE_1" alt="Logo" />`
**User gets**: `<img src="data:image/png;base64,iVBORw0KGgoAAAANS..." alt="Logo" />`

## Performance Impact:

- **Before**: Prompts contained full base64 strings (10K+ characters each)
- **After**: Prompts use short tokens (12 characters each)  
- **Result**: Dramatically faster LLM processing and lower costs
