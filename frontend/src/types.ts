import { Stack } from "./lib/stacks";
import { CodeGenerationModel } from "./lib/models";

export enum EditorTheme {
  ESPRESSO = "espresso",
  COBALT = "cobalt",
}

export interface Settings {
  openAiApiKey: string | null;
  openAiBaseURL: string | null;
  screenshotOneApiKey: string | null;
  isImageGenerationEnabled: boolean;
  editorTheme: EditorTheme;
  generatedCodeConfig: Stack;
  codeGenerationModel: CodeGenerationModel;
  // Only relevant for hosted version
  isTermOfServiceAccepted: boolean;
  anthropicApiKey: string | null; // Added property for anthropic API key
}

export enum AppState {
  INITIAL = "INITIAL",
  CODING = "CODING",
  CODE_READY = "CODE_READY",
}

export enum ScreenRecorderState {
  INITIAL = "initial",
  RECORDING = "recording",
  FINISHED = "finished",
}

export interface UploadedFile {
  id: string;
  dataUrl: string;
  name: string;
  type: 'screenshot' | 'asset';
  description?: string;
  file: File;
  preview: string;
}

export interface PromptContent {
  text: string;
  images: string[]; // Array of data URLs - will be deprecated in favor of uploadedFiles
  uploadedFiles?: UploadedFile[]; // New multi-file support
}

export interface CodeGenerationParams {
  generationType: "create" | "update";
  inputMode: "image" | "video" | "text";
  prompt: PromptContent;
  history?: PromptContent[];
  isImportedFromCode?: boolean;
  customModels?: string[];
  customVariantCount?: number;
}

export type FullGenerationSettings = CodeGenerationParams & Settings;
