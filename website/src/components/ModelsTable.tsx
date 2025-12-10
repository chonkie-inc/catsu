import { useState, useMemo, useEffect } from 'react';
import {
  useReactTable,
  getCoreRowModel,
  getFilteredRowModel,
  getSortedRowModel,
  type ColumnDef,
  type SortingState,
  flexRender,
} from '@tanstack/react-table';

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

const modalityBadge: Record<string, string> = {
  text: 'T',
  image: 'I',
};

const quantizationBadge: Record<string, string> = {
  float: 'F',
  int8: 'I',
  binary: 'B',
};

interface ModelsTableProps {
  models: Model[];
  filterValue?: string;
  onFilterChange?: (filter: string) => void;
  onRowCountChange?: (count: number) => void;
  onProviderCountChange?: (count: number) => void;
}

export function ModelsTable({ models, filterValue = '', onFilterChange, onRowCountChange, onProviderCountChange }: ModelsTableProps) {
  const [globalFilter, setGlobalFilter] = useState(filterValue);
  const [sorting, setSorting] = useState<SortingState>([]);

  // Sync external filterValue with internal state
  useEffect(() => {
    setGlobalFilter(filterValue);
  }, [filterValue]);

  const columns = useMemo<ColumnDef<Model>[]>(
    () => [
      {
        accessorKey: 'provider',
        header: 'Provider',
        size: 100,
        cell: (info) => (
          <span className="font-medium uppercase">
            {info.getValue<string>()}
          </span>
        ),
      },
      {
        accessorKey: 'name',
        header: 'Model',
        size: 300,
        cell: (info) => info.getValue<string>(),
      },
      {
        accessorKey: 'dimensions',
        header: 'Dimensions',
        size: 110,
        cell: (info) => (
          <span className="tabular-nums">
            {info.getValue<number>().toLocaleString()}
          </span>
        ),
      },
      {
        accessorKey: 'max_input_tokens',
        header: 'Max Tokens',
        size: 120,
        cell: (info) => (
          <span className="tabular-nums">
            {info.getValue<number>().toLocaleString()}
          </span>
        ),
      },
      {
        accessorKey: 'cost_per_million_tokens',
        header: 'Cost/1M',
        size: 100,
        cell: (info) => (
          <span className="tabular-nums">
            ${info.getValue<number>().toFixed(2)}
          </span>
        ),
      },
      {
        accessorKey: 'mteb_score',
        header: 'MTEB',
        size: 80,
        cell: (info) => {
          const score = info.getValue<number | null>();
          return (
            <span className="tabular-nums">
              {score !== null ? score.toFixed(2) : '—'}
            </span>
          );
        },
      },
      {
        accessorKey: 'rteb_score',
        header: 'RTEB',
        size: 80,
        cell: (info) => {
          const score = info.getValue<number | null>();
          return (
            <span className="tabular-nums">
              {score !== null ? score.toFixed(2) : '—'}
            </span>
          );
        },
      },
      {
        accessorKey: 'modalities',
        header: 'Modality',
        size: 100,
        cell: (info) => {
          const modalities = info.getValue<string[]>() || ['text'];
          return (
            <div className="flex gap-1">
              {modalities.map((modality) => (
                <span
                  key={modality}
                  className="inline-flex items-center justify-center w-5 h-5 text-[10px] font-semibold border border-gray-300 dark:border-gray-600 rounded"
                  title={modality}
                >
                  {modalityBadge[modality] || modality[0].toUpperCase()}
                </span>
              ))}
            </div>
          );
        },
      },
      {
        accessorKey: 'quantizations',
        header: 'Quant',
        size: 90,
        cell: (info) => {
          const quantizations = info.getValue<string[]>() || ['float'];
          return (
            <div className="flex gap-1">
              {quantizations.map((quant) => (
                <span
                  key={quant}
                  className="inline-flex items-center justify-center w-5 h-5 text-[10px] font-semibold border border-gray-300 dark:border-gray-600 rounded"
                  title={quant}
                >
                  {quantizationBadge[quant] || quant[0].toUpperCase()}
                </span>
              ))}
            </div>
          );
        },
      },
      {
        accessorKey: 'supports_batching',
        header: 'Batching',
        size: 80,
        cell: (info) => (info.getValue<boolean>() ? 'YES' : 'NO'),
      },
      {
        accessorKey: 'supports_input_type',
        header: 'Input Type',
        size: 100,
        cell: (info) => (info.getValue<boolean>() ? 'YES' : 'NO'),
      },
      {
        accessorKey: 'supports_dimensions',
        header: 'Config Dims',
        size: 110,
        cell: (info) => (info.getValue<boolean>() ? 'YES' : 'NO'),
      },
      {
        accessorKey: 'release_date',
        header: 'Released',
        size: 100,
        cell: (info) => {
          const dateStr = info.getValue<string | null>();
          if (!dateStr) return <span className="text-gray-400 dark:text-gray-500">—</span>;

          const date = new Date(dateStr);
          const formatter = new Intl.DateTimeFormat('en-US', {
            year: 'numeric',
            month: 'short'
          });

          return formatter.format(date);
        },
      },
    ],
    []
  );

  const table = useReactTable({
    data: models,
    columns,
    state: {
      globalFilter,
      sorting,
    },
    onGlobalFilterChange: (value) => {
      setGlobalFilter(value);
      onFilterChange?.(value as string);
    },
    onSortingChange: setSorting,
    getCoreRowModel: getCoreRowModel(),
    getFilteredRowModel: getFilteredRowModel(),
    getSortedRowModel: getSortedRowModel(),
  });

  // Notify parent of row count and provider count changes
  const rows = table.getRowModel().rows;
  const rowCount = rows.length;
  const providerCount = new Set(rows.map(row => row.original.provider)).size;

  useEffect(() => {
    if (onRowCountChange) {
      onRowCountChange(rowCount);
    }
  }, [rowCount, onRowCountChange]);

  useEffect(() => {
    if (onProviderCountChange) {
      onProviderCountChange(providerCount);
    }
  }, [providerCount, onProviderCountChange]);

  return (
    <div className="w-full">
      <table className="w-full text-xs border-collapse">
        <thead>
          {table.getHeaderGroups().map((headerGroup) => (
            <tr key={headerGroup.id}>
              {headerGroup.headers.map((header) => (
                <th
                  key={header.id}
                  className="sticky top-[57px] z-10 px-4 py-3 text-left text-xs font-semibold uppercase tracking-wide cursor-pointer select-none
                    text-gray-600 dark:text-gray-300
                    border-y border-gray-200 dark:border-gray-800
                    bg-gray-100/80 dark:bg-gray-800/80
                    backdrop-blur-md
                    hover:bg-gray-200/80 dark:hover:bg-gray-700/80
                    transition-colors"
                  style={{ width: header.getSize() }}
                  onClick={header.column.getToggleSortingHandler()}
                >
                  <div className="flex items-center gap-1">
                    <span>
                      {flexRender(
                        header.column.columnDef.header,
                        header.getContext()
                      )}
                    </span>
                    {header.column.getIsSorted() && (
                      <span>
                        {header.column.getIsSorted() === 'asc' ? '↑' : '↓'}
                      </span>
                    )}
                  </div>
                </th>
              ))}
            </tr>
          ))}
        </thead>
        <tbody className="text-gray-700 dark:text-gray-300">
          {rows.map((row) => (
            <tr
              key={row.id}
              className="border-b border-gray-100 dark:border-gray-800/50 hover:bg-gray-50 dark:hover:bg-gray-900/30 transition-colors"
            >
              {row.getVisibleCells().map((cell) => (
                <td
                  key={cell.id}
                  className="px-4 py-3 whitespace-nowrap"
                  style={{ width: cell.column.getSize() }}
                >
                  {flexRender(cell.column.columnDef.cell, cell.getContext())}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

// Separate search component that can be used in the header
export function ModelsSearch({ value, onChange }: { value: string; onChange: (value: string) => void }) {
  return (
    <input
      type="text"
      value={value}
      onChange={(e) => onChange(e.target.value)}
      placeholder="Filter by model"
      className="w-64 px-3 py-1.5 text-sm border border-gray-300 dark:border-gray-700 rounded bg-white dark:bg-gray-900 text-gray-900 dark:text-gray-100 placeholder-gray-400 focus:outline-none focus:ring-1 focus:ring-gray-400 dark:focus:ring-gray-600"
    />
  );
}
