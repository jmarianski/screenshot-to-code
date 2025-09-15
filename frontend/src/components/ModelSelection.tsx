import { useState, useEffect } from 'react';
import { HTTP_BACKEND_URL } from '../config';

interface Model {
  id: string;
  name: string;
  provider: string;
  display_name: string;
}

interface AvailableModelsResponse {
  models: Model[];
  providers: string[];
  has_models: boolean;
}

interface ModelSelectionProps {
  selectedModels: string[];
  onModelsChange: (models: string[]) => void;
  onVariantCountChange: (count: number) => void;
}

export default function ModelSelection({
  selectedModels,
  onModelsChange,
  onVariantCountChange,
}: ModelSelectionProps) {
  const [availableModels, setAvailableModels] = useState<Model[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    fetchAvailableModels();
  }, []);

  const fetchAvailableModels = async () => {
    try {
      const response = await fetch(`${HTTP_BACKEND_URL}/available-models`);
      if (!response.ok) {
        throw new Error('Failed to fetch available models');
      }
      const data: AvailableModelsResponse = await response.json();
      setAvailableModels(data.models);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Unknown error');
    } finally {
      setLoading(false);
    }
  };

  const handleModelToggle = (modelId: string) => {
    let newSelectedModels: string[];
    if (selectedModels.includes(modelId)) {
      newSelectedModels = selectedModels.filter((id) => id !== modelId);
    } else {
      newSelectedModels = [...selectedModels, modelId];
    }

    onModelsChange(newSelectedModels);

    // Always set variant count to match number of selected models
    // If no models selected, default to 1 for auto-selection
    onVariantCountChange(newSelectedModels.length || 1);
  };

  const groupedModels = availableModels.reduce((acc, model) => {
    if (!acc[model.provider]) {
      acc[model.provider] = [];
    }
    acc[model.provider].push(model);
    return acc;
  }, {} as Record<string, Model[]>);

  if (loading) {
    return (
      <div className='text-sm text-slate-600 dark:text-slate-400'>
        Loading available models...
      </div>
    );
  }

  if (error) {
    return (
      <div className='text-sm text-red-600 dark:text-red-400'>
        Error: {error}
      </div>
    );
  }

  if (availableModels.length === 0) {
    return (
      <div className='text-sm text-amber-600 dark:text-amber-400'>
        No models available. Please check your API keys in the backend/.env
        file.
      </div>
    );
  }

  return (
    <div className='bg-gradient-to-br from-blue-50 to-indigo-50 dark:from-blue-950/30 dark:to-indigo-950/30 border border-blue-200 dark:border-blue-700/50 rounded-xl p-6 shadow-sm'>
      <div className='flex items-center gap-2 mb-6'>
        <div className='w-2 h-2 bg-blue-500 dark:bg-blue-400 rounded-full'></div>
        <h3 className='text-lg font-semibold text-slate-800 dark:text-slate-200'>
          AI Model Selection
        </h3>
      </div>

      <div className='space-y-5'>
        <p className='text-sm text-slate-600 dark:text-slate-300 bg-white/60 dark:bg-slate-800/60 rounded-lg p-3 border border-slate-200 dark:border-slate-600'>
          💡 <strong>Tip:</strong> Select specific models to compare their
          outputs, or leave empty for smart auto-selection
        </p>

        <div className='grid gap-4'>
          {Object.entries(groupedModels).map(([provider, models]) => (
            <div
              key={provider}
              className='bg-white dark:bg-slate-800 rounded-lg p-4 border border-slate-200 dark:border-slate-600 shadow-sm'
            >
              <div className='flex items-center gap-2 mb-3'>
                <div
                  className={`w-3 h-3 rounded-full ${
                    provider === 'openai'
                      ? 'bg-green-500 dark:bg-green-400'
                      : provider === 'anthropic'
                      ? 'bg-orange-500 dark:bg-orange-400'
                      : provider === 'gemini'
                      ? 'bg-blue-500 dark:bg-blue-400'
                      : 'bg-slate-500 dark:bg-slate-400'
                  }`}
                ></div>
                <h4 className='font-semibold text-slate-800 dark:text-slate-200 capitalize text-base'>
                  {provider === 'openai'
                    ? 'OpenAI'
                    : provider === 'anthropic'
                    ? 'Anthropic (Claude)'
                    : provider === 'gemini'
                    ? 'Google Gemini'
                    : provider}
                </h4>
              </div>
              <div className='grid grid-cols-1 gap-2'>
                {models.map((model) => (
                  <label
                    key={model.id}
                    className='group flex items-center space-x-3 p-2 rounded-lg hover:bg-slate-50 dark:hover:bg-slate-700 cursor-pointer transition-colors'
                  >
                    <input
                      type='checkbox'
                      checked={selectedModels.includes(model.id)}
                      onChange={() => handleModelToggle(model.id)}
                      className='w-4 h-4 text-blue-600 dark:text-blue-400 rounded border-slate-300 dark:border-slate-600 focus:ring-blue-500 dark:focus:ring-blue-400 focus:ring-2 bg-white dark:bg-slate-700'
                    />
                    <span className='text-sm text-slate-700 dark:text-slate-300 group-hover:text-slate-900 dark:group-hover:text-slate-100 font-medium'>
                      {model.display_name}
                    </span>
                  </label>
                ))}
              </div>
            </div>
          ))}
        </div>

        {selectedModels.length > 0 && (
          <div className='bg-blue-100 dark:bg-blue-900/30 border border-blue-300 dark:border-blue-700/50 rounded-lg p-4'>
            <div className='flex items-center gap-2'>
              <div className='w-5 h-5 bg-blue-500 dark:bg-blue-400 rounded-full flex items-center justify-center'>
                <span className='text-white dark:text-slate-900 text-xs font-bold'>
                  {selectedModels.length}
                </span>
              </div>
              <span className='text-blue-800 dark:text-blue-200 font-medium'>
                Will generate {selectedModels.length} variant
                {selectedModels.length === 1 ? '' : 's'} using your selected
                models
              </span>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
