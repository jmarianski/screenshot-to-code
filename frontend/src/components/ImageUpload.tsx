import { useState, useEffect } from 'react';
import { useDropzone } from 'react-dropzone';
import { toast } from 'react-hot-toast';
import { URLS } from '../urls';
import ScreenRecorder from './recording/ScreenRecorder';
import { ScreenRecorderState, UploadedFile } from '../types';
import ModelSelection from './ModelSelection';

// Modern Tailwind-based styling - removing inline styles for better maintainability

// TODO: Move to a separate file
function fileToDataURL(file: File) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(reader.result);
    reader.onerror = (error) => reject(error);
    reader.readAsDataURL(file);
  });
}

type FileWithPreview = {
  preview: string;
} & File;

interface Props {
  setReferenceImages: (
    referenceImages: string[] | UploadedFile[],
    inputMode: 'image' | 'video',
    customModels?: string[],
    customVariantCount?: number,
  ) => void;
}

function ImageUpload({ setReferenceImages }: Props) {
  const [files, setFiles] = useState<FileWithPreview[]>([]);
  const [uploadedFiles, setUploadedFiles] = useState<UploadedFile[]>([]);
  const [mainScreenshotId, setMainScreenshotId] = useState<string | null>(null);
  const [dataUrls, setDataUrls] = useState<string[]>([]);
  const [inputMode, setInputMode] = useState<'image' | 'video'>('image');
  const [selectedModels, setSelectedModels] = useState<string[]>([]);
  const [variantCount, setVariantCount] = useState<number>(1);
  const [showModelSelection, setShowModelSelection] = useState<boolean>(false);
  // TODO: Switch to Zustand
  const [screenRecorderState, setScreenRecorderState] =
    useState<ScreenRecorderState>(ScreenRecorderState.INITIAL);

  const { getRootProps, getInputProps, isFocused, isDragAccept, isDragReject } =
    useDropzone({
      maxFiles: 10,
      maxSize: 1024 * 1024 * 20, // 20 MB
      accept: {
        // Image formats
        'image/png': ['.png'],
        'image/jpeg': ['.jpeg'],
        'image/jpg': ['.jpg'],
        // Video formats
        'video/quicktime': ['.mov'],
        'video/mp4': ['.mp4'],
        'video/webm': ['.webm'],
      },
      onDrop: (acceptedFiles) => {
        // Keep old format for backward compatibility
        setFiles(
          acceptedFiles.map((file: File) =>
            Object.assign(file, {
              preview: URL.createObjectURL(file),
            }),
          ) as FileWithPreview[],
        );

        // Create new UploadedFile format
        Promise.all(acceptedFiles.map((file) => fileToDataURL(file)))
          .then((urls) => {
            setDataUrls(urls as string[]);
            
            // Create UploadedFile objects
            const newUploadedFiles: UploadedFile[] = acceptedFiles.map((file, index) => ({
              id: `file-${Date.now()}-${index}`,
              dataUrl: urls[index] as string,
              name: file.name,
              type: index === 0 && uploadedFiles.length === 0 ? 'screenshot' : 'asset', // First file is screenshot by default
              file,
              preview: URL.createObjectURL(file),
            }));

            // Add to existing files
            const allFiles = [...uploadedFiles, ...newUploadedFiles];
            setUploadedFiles(allFiles);

            // Set main screenshot if none selected
            if (!mainScreenshotId && newUploadedFiles.length > 0) {
              setMainScreenshotId(newUploadedFiles[0].id);
            }

            if (urls.length > 0) {
              setInputMode(
                (urls[0] as string).startsWith('data:video')
                  ? 'video'
                  : 'image',
              );
            }
          })
          .catch((error) => {
            toast.error('Error reading files' + error);
            console.error('Error reading files:', error);
          });
      },
      onDropRejected: (rejectedFiles) => {
        toast.error(rejectedFiles[0].errors[0].message);
      },
    });

  // Handle paste event for images from clipboard
  useEffect(() => {
    function handlePaste(event: ClipboardEvent) {
      if (screenRecorderState !== ScreenRecorderState.INITIAL) return;
      const clipboardData = event.clipboardData;
      if (!clipboardData) return;
      const items = clipboardData.items;
      const imageFiles: File[] = [];
      for (let i = 0; i < items.length; i++) {
        const file = items[i].getAsFile();
        if (file && file.type.startsWith('image/')) {
          imageFiles.push(file);
        }
      }
      if (imageFiles.length > 0) {
        Promise.all(imageFiles.map((file) => fileToDataURL(file)))
          .then((urls) => {
            // Add to existing files using functional update to avoid closure issues
            setUploadedFiles(prevFiles => {
              // Create UploadedFile objects for pasted images using current state
              const newUploadedFiles: UploadedFile[] = imageFiles.map((file, index) => ({
                id: `paste-${Date.now()}-${index}`,
                dataUrl: urls[index] as string,
                name: file.name || `Pasted Image ${index + 1}`,
                description: '',
                type: (index === 0 && prevFiles.length === 0 ? 'screenshot' : 'asset') as 'screenshot' | 'asset',
                file,
                preview: URL.createObjectURL(file),
              }));

              const allFiles = [...prevFiles, ...newUploadedFiles];
              
              // Update legacy arrays for backward compatibility
              const newDataUrls = allFiles.map(f => f.dataUrl);
              setDataUrls(newDataUrls);
              const newLegacyFiles = allFiles.map(f => f.file as FileWithPreview);
              setFiles(newLegacyFiles);
              
              // Set main screenshot if none selected
              setMainScreenshotId(prevMainId => {
                if (!prevMainId && newUploadedFiles.length > 0) {
                  return newUploadedFiles[0].id;
                }
                return prevMainId;
              });
              
              return allFiles;
            });

            setInputMode('image');
            toast.success(`Pasted ${imageFiles.length} image${imageFiles.length > 1 ? 's' : ''} from clipboard`);
          })
          .catch((error) => {
            toast.error('Error reading pasted image: ' + error);
            console.error('Error reading pasted image:', error);
          });
      }
    }
    window.addEventListener(
      'paste',
      handlePaste as (event: ClipboardEvent) => void,
    );
    return () =>
      window.removeEventListener(
        'paste',
        handlePaste as (event: ClipboardEvent) => void,
      );
  }, [screenRecorderState]);

  useEffect(() => {
    return () => {
      files.forEach((file) => URL.revokeObjectURL(file.preview));
      uploadedFiles.forEach((file) => URL.revokeObjectURL(file.preview));
    };
  }, [files, uploadedFiles]);

  // Helper functions for file management
  const removeFile = (fileId: string) => {
    const fileToRemove = uploadedFiles.find(f => f.id === fileId);
    if (fileToRemove) {
      URL.revokeObjectURL(fileToRemove.preview);
      const newFiles = uploadedFiles.filter(f => f.id !== fileId);
      setUploadedFiles(newFiles);
      
      // If removed file was the main screenshot, select another one
      if (mainScreenshotId === fileId) {
        const screenshotFile = newFiles.find(f => f.type === 'screenshot');
        setMainScreenshotId(screenshotFile?.id || (newFiles.length > 0 ? newFiles[0].id : null));
      }
      
      // Update legacy arrays
      const newDataUrls = newFiles.map(f => f.dataUrl);
      setDataUrls(newDataUrls);
      const newLegacyFiles = newFiles.map(f => f.file as FileWithPreview);
      setFiles(newLegacyFiles);
    }
  };

  const setFileType = (fileId: string, type: 'screenshot' | 'asset') => {
    const newFiles = uploadedFiles.map(file => 
      file.id === fileId ? { ...file, type } : file
    );
    setUploadedFiles(newFiles);
    
    if (type === 'screenshot') {
      setMainScreenshotId(fileId);
      // Ensure only one screenshot
      const otherFiles = newFiles.map(file => 
        file.id !== fileId && file.type === 'screenshot' 
          ? { ...file, type: 'asset' as const }
          : file
      );
      setUploadedFiles(otherFiles);
    }
  };

  const setFileDescription = (fileId: string, description: string) => {
    const newFiles = uploadedFiles.map(file => 
      file.id === fileId ? { ...file, description } : file
    );
    setUploadedFiles(newFiles);
  };

  // Dynamic Tailwind classes based on dropzone state
  const dropzoneClasses = `
    flex flex-col items-center justify-center w-full min-h-[300px] p-8
    border-2 border-dashed rounded-2xl transition-all duration-300 cursor-pointer
    bg-gradient-to-br from-slate-50 to-blue-50 dark:from-slate-800 dark:to-slate-700
    hover:from-blue-50 hover:to-purple-50 dark:hover:from-slate-700 dark:hover:to-slate-600
    ${isFocused 
      ? 'border-blue-400 shadow-lg ring-4 ring-blue-100 dark:ring-blue-900' 
      : isDragAccept
        ? 'border-green-400 bg-green-50 dark:bg-green-900/20'
        : isDragReject
          ? 'border-red-400 bg-red-50 dark:bg-red-900/20'
          : 'border-slate-300 dark:border-slate-600'
    }
  `.trim();

  return (
    <div className="space-y-6">
      {screenRecorderState === ScreenRecorderState.INITIAL && (
        <div {...getRootProps({ className: dropzoneClasses })}>
          <input {...getInputProps()} className="hidden" />
          <div className="text-center space-y-4">
            <div className="w-16 h-16 mx-auto bg-gradient-to-r from-blue-500 to-purple-500 rounded-2xl flex items-center justify-center shadow-lg">
              <svg className="w-8 h-8 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12" />
              </svg>
            </div>
            <div className="space-y-2">
              <p className="text-lg font-medium text-slate-700 dark:text-slate-200">
                Drop your screenshot here
              </p>
              <p className="text-slate-500 dark:text-slate-400">
                or <span className="text-blue-600 dark:text-blue-400 font-medium">click to browse</span> files
              </p>
              <p className="text-sm text-slate-400 dark:text-slate-500 font-medium">
                ✨ You can also paste from clipboard (Ctrl+V)
              </p>
            </div>
          </div>
        </div>
      )}
      {/* File Management Section */}
      {uploadedFiles.length > 0 && (
        <div className="bg-white dark:bg-slate-800 rounded-xl p-6 border border-slate-200 dark:border-slate-700">
          <h3 className="text-lg font-semibold text-slate-800 dark:text-slate-200 mb-4">
            Uploaded Files ({uploadedFiles.length})
          </h3>
          
          {uploadedFiles.length > 1 && (
            <div className="mb-4 p-3 bg-blue-50 dark:bg-blue-900/20 border border-blue-200 dark:border-blue-700/50 rounded-lg">
              <p className="text-sm text-blue-800 dark:text-blue-200 font-medium">
                💡 Select which image is the main screenshot to recreate, others will be used as assets
              </p>
            </div>
          )}

          <div className="grid gap-4">
            {uploadedFiles.map((file) => (
              <div
                key={file.id}
                className={`flex gap-4 p-4 rounded-lg border-2 transition-colors ${
                  file.type === 'screenshot'
                    ? 'border-blue-500 bg-blue-50 dark:bg-blue-900/20'
                    : 'border-slate-200 dark:border-slate-600 bg-slate-50 dark:bg-slate-700/50'
                }`}
              >
                {/* Image Preview */}
                <div className="flex-shrink-0">
                  <img
                    src={file.preview}
                    alt={file.name}
                    className="w-20 h-20 object-cover rounded-lg border border-slate-200 dark:border-slate-600"
                  />
                </div>

                {/* File Info and Controls */}
                <div className="flex-1 space-y-3">
                  <div className="flex items-start justify-between">
                    <div>
                      <p className="font-medium text-slate-800 dark:text-slate-200 truncate">
                        {file.name}
                      </p>
                      <p className="text-xs text-slate-500 dark:text-slate-400">
                        {(file.file.size / 1024 / 1024).toFixed(1)} MB
                      </p>
                    </div>
                    <button
                      onClick={() => removeFile(file.id)}
                      className="p-1 text-slate-400 hover:text-red-500 dark:hover:text-red-400 transition-colors"
                      title="Remove file"
                    >
                      <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                      </svg>
                    </button>
                  </div>

                  {/* File Type Selection */}
                  <div className="flex gap-2">
                    <label className="flex items-center gap-2 cursor-pointer">
                      <input
                        type="radio"
                        name="screenshot"
                        checked={file.type === 'screenshot'}
                        onChange={() => setFileType(file.id, 'screenshot')}
                        className="w-4 h-4 text-blue-600 dark:text-blue-400"
                      />
                      <span className="text-sm font-medium text-slate-700 dark:text-slate-300">
                        Main Screenshot
                      </span>
                    </label>
                    <label className="flex items-center gap-2 cursor-pointer">
                      <input
                        type="radio"
                        name={`type-${file.id}`}
                        checked={file.type === 'asset'}
                        onChange={() => setFileType(file.id, 'asset')}
                        className="w-4 h-4 text-green-600 dark:text-green-400"
                      />
                      <span className="text-sm font-medium text-slate-700 dark:text-slate-300">
                        Asset File
                      </span>
                    </label>
                  </div>

                  {/* Description Input for Assets */}
                  {file.type === 'asset' && (
                    <input
                      type="text"
                      placeholder="Describe this asset (e.g., 'Company logo', 'Profile photo')..."
                      value={file.description || ''}
                      onChange={(e) => setFileDescription(file.id, e.target.value)}
                      className="w-full px-3 py-1.5 text-sm border border-slate-200 dark:border-slate-600 rounded bg-white dark:bg-slate-800 text-slate-700 dark:text-slate-300 placeholder-slate-400 dark:placeholder-slate-500 focus:ring-2 focus:ring-blue-500 dark:focus:ring-blue-400 focus:border-transparent"
                    />
                  )}
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Advanced Options */}
      <div className="space-y-4">
        <button
          className="flex items-center justify-center gap-2 px-4 py-2.5 bg-slate-100 hover:bg-slate-200 dark:bg-slate-700 dark:hover:bg-slate-600 text-slate-700 dark:text-slate-200 rounded-lg transition-colors font-medium text-sm border border-slate-200 dark:border-slate-600"
          onClick={() => setShowModelSelection(!showModelSelection)}
        >
          <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10.325 4.317c.426-1.756 2.924-1.756 3.35 0a1.724 1.724 0 002.573 1.066c1.543-.94 3.31.826 2.37 2.37a1.724 1.724 0 001.065 2.572c1.756.426 1.756 2.924 0 3.35a1.724 1.724 0 00-1.066 2.573c.94 1.543-.826 3.31-2.37 2.37a1.724 1.724 0 00-2.572 1.065c-.426 1.756-2.924 1.756-3.35 0a1.724 1.724 0 00-2.573-1.066c-1.543.94-3.31-.826-2.37-2.37a1.724 1.724 0 00-1.065-2.572c-1.756-.426-1.756-2.924 0-3.35a1.724 1.724 0 001.066-2.573c-.94-1.543.826-3.31 2.37-2.37.996.608 2.296.07 2.572-1.065z" />
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" />
          </svg>
          {showModelSelection ? 'Hide' : 'Show'} Advanced Options
        </button>

        {showModelSelection && (
          <div className="bg-slate-50 dark:bg-slate-800 rounded-xl p-4 border border-slate-200 dark:border-slate-700">
            <ModelSelection
              selectedModels={selectedModels}
              onModelsChange={setSelectedModels}
              onVariantCountChange={setVariantCount}
            />
          </div>
        )}

        <button
          className="w-full px-6 py-4 bg-gradient-to-r from-blue-600 to-purple-600 hover:from-blue-700 hover:to-purple-700 text-white rounded-xl transition-all duration-200 font-semibold text-lg shadow-lg hover:shadow-xl disabled:opacity-50 disabled:cursor-not-allowed disabled:hover:shadow-lg"
          onClick={() => {
            // Use new format if we have uploadedFiles, otherwise fallback to legacy
            if (uploadedFiles.length > 0) {
              // Ensure we have at least one screenshot
              const hasScreenshot = uploadedFiles.some(f => f.type === 'screenshot');
              if (!hasScreenshot && uploadedFiles.length > 0) {
                // Auto-designate first file as screenshot if none selected
                const firstFile = uploadedFiles[0];
                setFileType(firstFile.id, 'screenshot');
                const updatedFiles = uploadedFiles.map((f, i) => 
                  i === 0 ? { ...f, type: 'screenshot' as const } : f
                );
                setReferenceImages(
                  updatedFiles,
                  inputMode,
                  selectedModels.length > 0 ? selectedModels : undefined,
                  variantCount > 1 ? variantCount : undefined,
                );
              } else {
                setReferenceImages(
                  uploadedFiles,
                  inputMode,
                  selectedModels.length > 0 ? selectedModels : undefined,
                  variantCount > 1 ? variantCount : undefined,
                );
              }
            } else if (dataUrls.length > 0) {
              // Fallback to legacy format
              setReferenceImages(
                dataUrls,
                inputMode,
                selectedModels.length > 0 ? selectedModels : undefined,
                variantCount > 1 ? variantCount : undefined,
              );
            }
          }}
          disabled={uploadedFiles.length === 0 && dataUrls.length === 0}
        >
          <div className="flex items-center justify-center gap-2">
            <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" />
            </svg>
            Generate Code
          </div>
        </button>
      </div>
      {screenRecorderState === ScreenRecorderState.INITIAL && (
        <div className="bg-amber-50 dark:bg-amber-900/20 border border-amber-200 dark:border-amber-700/50 rounded-xl p-4 text-center">
          <div className="flex items-center justify-center gap-2 mb-2">
            <svg className="w-5 h-5 text-amber-600 dark:text-amber-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 10l4.553-2.276A1 1 0 0121 8.618v6.764a1 1 0 01-1.447.894L15 14M5 18h8a2 2 0 002-2V8a2 2 0 00-2-2H5a2 2 0 00-2 2v8a2 2 0 002 2z" />
            </svg>
            <span className="font-medium text-amber-800 dark:text-amber-200">Video Recording</span>
          </div>
          <p className="text-sm text-amber-700 dark:text-amber-300 mb-3">
            Upload a screen recording (.mp4, .mov) or record your screen to clone a whole app
          </p>
          <a
            className="inline-flex items-center gap-1 text-sm text-amber-600 dark:text-amber-400 hover:text-amber-700 dark:hover:text-amber-300 underline font-medium"
            href={URLS['intro-to-video']}
            target='_blank'
          >
            Learn more about video features
            <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10 6H6a2 2 0 00-2 2v10a2 2 0 002 2h10a2 2 0 002-2v-4M14 4h6m0 0v6m0-6L10 14" />
            </svg>
          </a>
        </div>
      )}
      <ScreenRecorder
        screenRecorderState={screenRecorderState}
        setScreenRecorderState={setScreenRecorderState}
        generateCode={setReferenceImages}
      />
    </div>
  );
}

export default ImageUpload;
