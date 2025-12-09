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

  // Inject search bar into navbar (only once on mount)
  useEffect(() => {
    const navbarSearch = document.getElementById('navbar-search');

    if (navbarSearch && !navbarSearch.querySelector('input')) {
      const searchInput = document.createElement('input');
      searchInput.type = 'text';
      searchInput.placeholder = 'Filter';
      searchInput.className = 'w-24 md:w-48 px-2 md:px-3 py-1.5 text-xs border border-gray-300 dark:border-gray-700 rounded bg-white dark:bg-gray-900 text-gray-900 dark:text-gray-100 placeholder-gray-400 focus:outline-none focus:ring-1 focus:ring-gray-400 dark:focus:ring-gray-600';
      searchInput.addEventListener('input', (e) => {
        setSearchValue((e.target as HTMLInputElement).value);
      });
      navbarSearch.appendChild(searchInput);
    }
  }, []);

  // Update model count in navbar
  useEffect(() => {
    const navbarModelCount = document.getElementById('navbar-model-count');
    if (navbarModelCount) {
      navbarModelCount.textContent = rowCount.toString();
    }
  }, [rowCount]);

  return (
    <div>
      <ModelsTable
        models={models}
        filterValue={searchValue}
        onFilterChange={setSearchValue}
        onRowCountChange={setRowCount}
      />
    </div>
  );
}
