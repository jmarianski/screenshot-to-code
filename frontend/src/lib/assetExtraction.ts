/**
 * Asset extraction service functions
 */

import { HTTP_BACKEND_URL } from '../config';

export interface ExtractedAsset {
    id: string;
    x: number;
    y: number;
    width: number;
    height: number;
    confidence: number;
    type: string;
    data_url: string;
}

export interface AssetExtractionResponse {
    success: boolean;
    message: string;
    assets: ExtractedAsset[];
    stats: {
        total_assets: number;
        asset_types: Record<string, number>;
        extraction_config: Record<string, unknown>;
    };
}

export interface AssetExtractionConfig {
    min_area?: number;
    max_assets?: number;
    canny_low?: number;
    canny_high?: number;
    nms_overlap_threshold?: number;
    merge_iou_threshold?: number;
}

/**
 * Extract assets from a screenshot using computer vision
 */
export async function extractAssets(
    screenshotDataUrl: string,
    config?: AssetExtractionConfig
): Promise<AssetExtractionResponse> {
    try {
        const response = await fetch(`${HTTP_BACKEND_URL}/asset-extraction/extract`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                screenshot_data_url: screenshotDataUrl,
                config: config || {},
            }),
        });

        if (!response.ok) {
            const errorData = await response.json().catch(() => ({ detail: 'Unknown error' }));
            throw new Error(errorData.detail || `HTTP ${response.status}`);
        }

        const result: AssetExtractionResponse = await response.json();
        return result;
    } catch (error) {
        console.error('Asset extraction failed:', error);
        throw error;
    }
}

/**
 * Get default extraction configuration
 */
export async function getExtractionConfig(): Promise<Record<string, unknown>> {
    try {
        const response = await fetch(`${HTTP_BACKEND_URL}/asset-extraction/config/defaults`);

        if (!response.ok) {
            throw new Error(`HTTP ${response.status}`);
        }

        return await response.json();
    } catch (error) {
        console.error('Failed to get extraction config:', error);
        throw error;
    }
}

/**
 * Check if asset extraction service is available
 */
export async function checkAssetExtractionHealth(): Promise<boolean> {
    try {
        const response = await fetch(`${HTTP_BACKEND_URL}/asset-extraction/health`);

        if (!response.ok) {
            return false;
        }

        const result = await response.json();
        return result.status === 'healthy';
    } catch (error) {
        console.error('Asset extraction health check failed:', error);
        return false;
    }
}

/**
 * Convert extracted assets to UploadedFile format for use in the app
 */
export function assetsToUploadedFiles(assets: ExtractedAsset[]): Array<{
    id: string;
    dataUrl: string;
    name: string;
    type: 'asset';
    preview: string;
    file: File;
}> {
    return assets.map((asset, index) => {
        // Convert data URL to File object for consistency
        const byteString = atob(asset.data_url.split(',')[1]);
        const mimeString = asset.data_url.split(',')[0].split(':')[1].split(';')[0];
        const ab = new ArrayBuffer(byteString.length);
        const ia = new Uint8Array(ab);

        for (let i = 0; i < byteString.length; i++) {
            ia[i] = byteString.charCodeAt(i);
        }

        const blob = new Blob([ab], { type: mimeString });
        const file = new File([blob], `${asset.id}.png`, { type: 'image/png' });

        return {
            id: asset.id,
            dataUrl: asset.data_url,
            name: `${asset.type}_${index + 1}.png`,
            type: 'asset' as const,
            preview: asset.data_url,
            file,
        };
    });
}
