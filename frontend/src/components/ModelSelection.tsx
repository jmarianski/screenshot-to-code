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
      <div className='text-sm text-gray-600'>Loading available models...</div>
    );
  }

  if (error) {
    return <div className='text-sm text-red-600'>Error: {error}</div>;
  }

  if (availableModels.length === 0) {
    return (
      <div className='text-sm text-yellow-600'>
        No models available. Please check your API keys in the backend/.env
        file.
      </div>
    );
  }

  return (
    <div className='space-y-4 p-4 bg-gray-50 rounded-lg'>
      <h3 className='text-lg font-semibold'>Model Selection</h3>

      <div className='space-y-3'>
        <div className='text-sm text-gray-600'>
          Select models to use (leave empty for automatic selection):
        </div>

        {Object.entries(groupedModels).map(([provider, models]) => (
          <div key={provider} className='space-y-2'>
            <h4 className='font-medium capitalize text-sm'>
              {provider} Models
            </h4>
            <div className='grid grid-cols-1 gap-2 ml-4'>
              {models.map((model) => (
                <label
                  key={model.id}
                  className='flex items-center space-x-2 text-sm'
                >
                  <input
                    type='checkbox'
                    checked={selectedModels.includes(model.id)}
                    onChange={() => handleModelToggle(model.id)}
                    className='rounded'
                  />
                  <span>{model.display_name}</span>
                </label>
              ))}
            </div>
          </div>
        ))}
      </div>

      {selectedModels.length > 0 && (
        <div className='text-sm text-blue-600 font-medium'>
          Will generate {selectedModels.length} variant
          {selectedModels.length === 1 ? '' : 's'} using selected models
        </div>
      )}
    </div>
  );
}
