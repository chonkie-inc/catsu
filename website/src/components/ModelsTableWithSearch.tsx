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
  const [providerCount, setProviderCount] = useState(
    new Set(models.map((m) => m.provider)).size
  );

  // Inject search bar into navbar (only once on mount)
  useEffect(() => {
    const navbarSearch = document.getElementById('navbar-search');

    if (navbarSearch && !navbarSearch.querySelector('input')) {
      // Create wrapper for input and keyboard hint
      const wrapper = document.createElement('div');
      wrapper.className = 'relative inline-flex items-center';

      const searchInput = document.createElement('input');
      searchInput.type = 'text';
      searchInput.placeholder = 'Filter';
      searchInput.className = 'w-24 md:w-48 px-2 md:px-3 py-1.5 pr-12 text-xs border border-gray-300 dark:border-gray-700 rounded bg-white dark:bg-gray-900 text-gray-900 dark:text-gray-100 placeholder-gray-400 focus:outline-none focus:ring-1 focus:ring-gray-400 dark:focus:ring-gray-600';
      searchInput.addEventListener('input', (e) => {
        setSearchValue((e.target as HTMLInputElement).value);
      });

      // Create keyboard shortcut hint
      const isMac = navigator.platform.toUpperCase().indexOf('MAC') >= 0;
      const shortcutHint = document.createElement('kbd');
      shortcutHint.textContent = isMac ? 'cmd + k' : 'ctrl + k';
      shortcutHint.className = 'hidden md:inline-flex absolute right-2 top-1/2 -translate-y-1/2 px-1.5 py-0.5 text-[10px] font-medium text-gray-400 dark:text-gray-500 bg-gray-100 dark:bg-gray-800 border border-gray-200 dark:border-gray-700 rounded pointer-events-none';

      wrapper.appendChild(searchInput);
      wrapper.appendChild(shortcutHint);
      navbarSearch.appendChild(wrapper);

      // Add Cmd+K / Ctrl+K keyboard shortcut
      const handleKeyDown = (e: KeyboardEvent) => {
        if ((e.metaKey || e.ctrlKey) && e.key === 'k') {
          e.preventDefault();
          searchInput.focus();
        }
      };

      document.addEventListener('keydown', handleKeyDown);

      // Cleanup
      return () => {
        document.removeEventListener('keydown', handleKeyDown);
      };
    }
  }, []);

  // Update model count in navbar
  useEffect(() => {
    const navbarModelCount = document.getElementById('navbar-model-count');
    if (navbarModelCount) {
      navbarModelCount.textContent = rowCount.toString();
    }
  }, [rowCount]);

  // Update provider count in navbar
  useEffect(() => {
    const navbarProviderCount = document.getElementById('navbar-provider-count');
    if (navbarProviderCount) {
      navbarProviderCount.textContent = providerCount.toString();
    }
  }, [providerCount]);

  return (
    <div className="table-container">
      <ModelsTable
        models={models}
        filterValue={searchValue}
        onRowCountChange={setRowCount}
        onProviderCountChange={setProviderCount}
      />
    </div>
  );
}
