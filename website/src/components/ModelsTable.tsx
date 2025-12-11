import { useState, useMemo, useEffect } from 'react';

interface Model {
  name: string;
  provider: string;
  dimensions: number;
  max_input_tokens: number;
  cost_per_million_tokens: number;
  modalities: string[];
  quantizations?: string[];
  supports_batching: boolean;
  supports_input_type: boolean;
  supports_dimensions: boolean;
  mteb_score: number | null;
  rteb_score: number | null;
  release_date: string | null;
}

type SortDirection = 'asc' | 'desc' | null;
type SortKey = keyof Model | null;

interface ModelsTableProps {
  models: Model[];
  filterValue?: string;
  onRowCountChange?: (count: number) => void;
  onProviderCountChange?: (count: number) => void;
}

const modalityBadge: Record<string, string> = {
  text: 'T',
  image: 'I',
};

const quantizationBadge: Record<string, string> = {
  float: 'F',
  int8: 'I',
  binary: 'B',
};

const formatDate = (dateStr: string | null) => {
  if (!dateStr) return '—';
  const date = new Date(dateStr);
  return new Intl.DateTimeFormat('en-US', {
    year: 'numeric',
    month: 'short'
  }).format(date);
};

export function ModelsTable({ models, filterValue = '', onRowCountChange, onProviderCountChange }: ModelsTableProps) {
  const [sortKey, setSortKey] = useState<SortKey>(null);
  const [sortDirection, setSortDirection] = useState<SortDirection>(null);

  // Filter models based on search
  const filteredModels = useMemo(() => {
    if (!filterValue.trim()) return models;
    const search = filterValue.toLowerCase();
    return models.filter(model =>
      model.name.toLowerCase().includes(search) ||
      model.provider.toLowerCase().includes(search)
    );
  }, [models, filterValue]);

  // Sort filtered models
  const sortedModels = useMemo(() => {
    if (!sortKey || !sortDirection) return filteredModels;

    return [...filteredModels].sort((a, b) => {
      const aVal = a[sortKey];
      const bVal = b[sortKey];

      // Handle null values - always push them to the bottom
      if (aVal === null && bVal === null) return 0;
      if (aVal === null) return 1;  // a goes to bottom
      if (bVal === null) return -1; // b goes to bottom

      // Compare values
      if (typeof aVal === 'string' && typeof bVal === 'string') {
        const cmp = aVal.localeCompare(bVal);
        return sortDirection === 'asc' ? cmp : -cmp;
      }

      if (typeof aVal === 'number' && typeof bVal === 'number') {
        return sortDirection === 'asc' ? aVal - bVal : bVal - aVal;
      }

      if (typeof aVal === 'boolean' && typeof bVal === 'boolean') {
        const cmp = (aVal ? 1 : 0) - (bVal ? 1 : 0);
        return sortDirection === 'asc' ? cmp : -cmp;
      }

      return 0;
    });
  }, [filteredModels, sortKey, sortDirection]);

  // Handle column header click for sorting
  const handleSort = (key: SortKey) => {
    if (sortKey === key) {
      // Cycle through: desc -> asc -> null
      if (sortDirection === 'desc') {
        setSortDirection('asc');
      } else if (sortDirection === 'asc') {
        setSortDirection(null);
        setSortKey(null);
      } else {
        setSortDirection('desc');
      }
    } else {
      setSortKey(key);
      setSortDirection('desc');
    }
  };

  // Get sort direction for data attribute
  const getSortDirection = (key: SortKey) => {
    if (sortKey !== key) return undefined;
    return sortDirection || undefined;
  };

  // Notify parent of counts
  const rowCount = sortedModels.length;
  const providerCount = new Set(sortedModels.map(m => m.provider)).size;

  useEffect(() => {
    onRowCountChange?.(rowCount);
  }, [rowCount, onRowCountChange]);

  useEffect(() => {
    onProviderCountChange?.(providerCount);
  }, [providerCount, onProviderCountChange]);

  return (
    <table>
      <thead>
        <tr>
          <th className="sortable" onClick={() => handleSort('provider')} data-sort={getSortDirection('provider')}>
            Provider
          </th>
          <th className="sortable" onClick={() => handleSort('name')} data-sort={getSortDirection('name')}>
            Model
          </th>
          <th className="sortable" onClick={() => handleSort('dimensions')} data-sort={getSortDirection('dimensions')}>
            Dimensions
          </th>
          <th className="sortable" onClick={() => handleSort('max_input_tokens')} data-sort={getSortDirection('max_input_tokens')}>
            Max Tokens
          </th>
          <th className="sortable" onClick={() => handleSort('cost_per_million_tokens')} data-sort={getSortDirection('cost_per_million_tokens')}>
            Cost/1M
          </th>
          <th className="sortable" onClick={() => handleSort('mteb_score')} data-sort={getSortDirection('mteb_score')}>
            MTEB
          </th>
          <th className="sortable" onClick={() => handleSort('rteb_score')} data-sort={getSortDirection('rteb_score')}>
            RTEB
          </th>
          <th>Modality</th>
          <th>Quant</th>
          <th className="sortable" onClick={() => handleSort('supports_input_type')} data-sort={getSortDirection('supports_input_type')}>
            Input Type
          </th>
          <th className="sortable" onClick={() => handleSort('supports_dimensions')} data-sort={getSortDirection('supports_dimensions')}>
            Config Dims
          </th>
          <th className="sortable" onClick={() => handleSort('release_date')} data-sort={getSortDirection('release_date')}>
            Released
          </th>
        </tr>
      </thead>
      <tbody>
        {sortedModels.map((model, idx) => (
          <tr key={`${model.provider}-${model.name}-${idx}`}>
            <td>
              <span className="provider-name">{model.provider}</span>
            </td>
            <td>{model.name}</td>
            <td>{model.dimensions.toLocaleString()}</td>
            <td>{model.max_input_tokens.toLocaleString()}</td>
            <td>${model.cost_per_million_tokens.toFixed(2)}</td>
            <td>{model.mteb_score !== null ? model.mteb_score.toFixed(2) : '—'}</td>
            <td>{model.rteb_score !== null ? model.rteb_score.toFixed(2) : '—'}</td>
            <td>
              <div className="badges">
                {(model.modalities || ['text']).map((modality) => (
                  <span key={modality} className="badge" title={modality}>
                    {modalityBadge[modality] || modality[0].toUpperCase()}
                  </span>
                ))}
              </div>
            </td>
            <td>
              <div className="badges">
                {(model.quantizations || ['float']).map((quant) => (
                  <span key={quant} className="badge" title={quant}>
                    {quantizationBadge[quant] || quant[0].toUpperCase()}
                  </span>
                ))}
              </div>
            </td>
            <td>{model.supports_input_type ? 'Yes' : 'No'}</td>
            <td>{model.supports_dimensions ? 'Yes' : 'No'}</td>
            <td>{formatDate(model.release_date)}</td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}
