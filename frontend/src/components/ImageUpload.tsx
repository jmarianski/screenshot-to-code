import { useState, useEffect, useMemo } from 'react';
import { useDropzone } from 'react-dropzone';
import { toast } from 'react-hot-toast';
import { URLS } from '../urls';
import ScreenRecorder from './recording/ScreenRecorder';
import { ScreenRecorderState } from '../types';
import ModelSelection from './ModelSelection';

const baseStyle = {
  flex: 1,
  width: '80%',
  margin: '0 auto',
  minHeight: '400px',
  display: 'flex',
  flexDirection: 'column',
  alignItems: 'center',
  justifyContent: 'center',
  padding: '20px',
  borderWidth: 2,
  borderRadius: 2,
  borderColor: '#eeeeee',
  borderStyle: 'dashed',
  backgroundColor: '#fafafa',
  color: '#bdbdbd',
  outline: 'none',
  transition: 'border .24s ease-in-out',
};

const focusedStyle = {
  borderColor: '#2196f3',
};

const acceptStyle = {
  borderColor: '#00e676',
};

const rejectStyle = {
  borderColor: '#ff1744',
};

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
    referenceImages: string[],
    inputMode: 'image' | 'video',
    customModels?: string[],
    customVariantCount?: number,
  ) => void;
}

function ImageUpload({ setReferenceImages }: Props) {
  const [files, setFiles] = useState<FileWithPreview[]>([]);
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
      maxFiles: 1,
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
        setFiles(
          acceptedFiles.map((file: File) =>
            Object.assign(file, {
              preview: URL.createObjectURL(file),
            }),
          ) as FileWithPreview[],
        );
        Promise.all(acceptedFiles.map((file) => fileToDataURL(file)))
          .then((urls) => {
            setDataUrls(urls as string[]);
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
        setFiles(
          imageFiles.map((file: File) =>
            Object.assign(file, {
              preview: URL.createObjectURL(file),
            }),
          ) as FileWithPreview[],
        );
        Promise.all(imageFiles.map((file) => fileToDataURL(file)))
          .then((urls) => {
            setDataUrls(urls as string[]);
            setInputMode('image');
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
    return () => files.forEach((file) => URL.revokeObjectURL(file.preview));
  }, [files]); // Added files as a dependency

  const style = useMemo(
    () => ({
      ...baseStyle,
      ...(isFocused ? focusedStyle : {}),
      ...(isDragAccept ? acceptStyle : {}),
      ...(isDragReject ? rejectStyle : {}),
    }),
    [isFocused, isDragAccept, isDragReject],
  );

  return (
    <section className='container'>
      {screenRecorderState === ScreenRecorderState.INITIAL && (
        /* eslint-disable-next-line @typescript-eslint/no-explicit-any */
        <div {...getRootProps({ style: style as any })}>
          <input {...getInputProps()} className='file-input' />
          <p className='text-slate-700 text-lg'>
            Drag & drop a screenshot here,
            <br />
            click to upload,
            <br />
            <b>or paste a screenshot from your clipboard</b>
          </p>
        </div>
      )}
      <div className='my-4 flex flex-col items-center'>
        {files.length > 0 && (
          <div className='flex gap-4 mb-2'>
            {files.map((file, idx) => (
              <img
                key={idx}
                src={file.preview}
                alt={`preview-${idx}`}
                className='max-h-40 rounded shadow border'
              />
            ))}
          </div>
        )}
        <div className='space-y-4'>
          <button
            className='px-4 py-2 bg-gray-600 text-white rounded hover:bg-gray-700 transition text-sm'
            onClick={() => setShowModelSelection(!showModelSelection)}
          >
            {showModelSelection ? 'Hide' : 'Show'} Advanced Options
          </button>

          {showModelSelection && (
            <ModelSelection
              selectedModels={selectedModels}
              onModelsChange={setSelectedModels}
              onVariantCountChange={setVariantCount}
            />
          )}

          <button
            className='px-6 py-3 bg-blue-600 text-white rounded hover:bg-blue-700 transition font-medium'
            onClick={() => {
              if (dataUrls.length > 0) {
                setReferenceImages(
                  dataUrls,
                  inputMode,
                  selectedModels.length > 0 ? selectedModels : undefined,
                  variantCount > 1 ? variantCount : undefined,
                );
              }
            }}
            disabled={dataUrls.length === 0}
          >
            Generate Code
          </button>
        </div>
      </div>
      {screenRecorderState === ScreenRecorderState.INITIAL && (
        <div className='text-center text-sm text-slate-800 mt-4'>
          Upload a screen recording (.mp4, .mov) or record your screen to clone
          a whole app (experimental).{' '}
          <a
            className='underline'
            href={URLS['intro-to-video']}
            target='_blank'
          >
            Learn more.
          </a>
        </div>
      )}
      <ScreenRecorder
        screenRecorderState={screenRecorderState}
        setScreenRecorderState={setScreenRecorderState}
        generateCode={setReferenceImages}
      />
    </section>
  );
}

export default ImageUpload;
