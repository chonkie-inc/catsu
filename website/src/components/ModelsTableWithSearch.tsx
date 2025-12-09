import { useState, useEffect } from 'react';
import { ModelsTable } from './ModelsTable';

interface Model {
  name: string;
  provider: string;
  dimensions: number;
  max_input_tokens: number;
  cost_per_million_tokens: number;
  supports_batching: boolean;
  supports_input_type: boolean;
  supports_dimensions: boolean;
}

interface ModelsTableWithSearchProps {
  models: Model[];
}

export function ModelsTableWithSearch({ models }: ModelsTableWithSearchProps) {
  const [searchValue, setSearchValue] = useState('');
  const [rowCount, setRowCount] = useState(models.length);

  // Inject search bar into navbar
  useEffect(() => {
    const navbarSearch = document.getElementById('navbar-search');
    const navbarModelCount = document.getElementById('navbar-model-count');

    if (navbarSearch && !navbarSearch.querySelector('input')) {
      const searchInput = document.createElement('input');
      searchInput.type = 'text';
      searchInput.placeholder = 'Filter by model';
      searchInput.className = 'w-48 px-3 py-1.5 text-xs border border-gray-300 dark:border-gray-700 rounded bg-white dark:bg-gray-900 text-gray-900 dark:text-gray-100 placeholder-gray-400 focus:outline-none focus:ring-1 focus:ring-gray-400 dark:focus:ring-gray-600';
      searchInput.value = searchValue;
      searchInput.addEventListener('input', (e) => {
        setSearchValue((e.target as HTMLInputElement).value);
      });
      navbarSearch.appendChild(searchInput);
    }

    // Update model count in navbar
    if (navbarModelCount) {
      navbarModelCount.textContent = rowCount.toString();
    }
  }, [rowCount, searchValue]);

  return (
    <div>
      <ModelsTable
        models={models}
        onFilterChange={setSearchValue}
        onRowCountChange={setRowCount}
      />
    </div>
  );
}
