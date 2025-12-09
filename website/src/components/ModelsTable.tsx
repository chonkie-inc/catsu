import { useState, useMemo } from 'react';
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
  supports_batching: boolean;
  supports_input_type: boolean;
  supports_dimensions: boolean;
}

interface ModelsTableProps {
  models: Model[];
  onFilterChange?: (filter: string) => void;
  onRowCountChange?: (count: number) => void;
}

export function ModelsTable({ models, onFilterChange, onRowCountChange }: ModelsTableProps) {
  const [globalFilter, setGlobalFilter] = useState('');
  const [sorting, setSorting] = useState<SortingState>([]);

  const columns = useMemo<ColumnDef<Model>[]>(
    () => [
      {
        accessorKey: 'provider',
        header: 'PROVIDER',
        cell: (info) => (
          <span className="text-xs font-medium uppercase text-gray-900 dark:text-gray-100">
            {info.getValue<string>()}
          </span>
        ),
      },
      {
        accessorKey: 'name',
        header: 'MODEL',
        cell: (info) => (
          <span className="text-xs text-gray-700 dark:text-gray-300">
            {info.getValue<string>()}
          </span>
        ),
      },
      {
        accessorKey: 'dimensions',
        header: 'DIMENSIONS',
        cell: (info) => (
          <span className="text-xs tabular-nums text-gray-700 dark:text-gray-300">
            {info.getValue<number>().toLocaleString()}
          </span>
        ),
      },
      {
        accessorKey: 'max_input_tokens',
        header: 'MAX TOKENS',
        cell: (info) => (
          <span className="text-xs tabular-nums text-gray-700 dark:text-gray-300">
            {info.getValue<number>().toLocaleString()}
          </span>
        ),
      },
      {
        accessorKey: 'cost_per_million_tokens',
        header: 'COST/1M',
        cell: (info) => (
          <span className="text-xs tabular-nums text-gray-700 dark:text-gray-300">
            ${info.getValue<number>().toFixed(2)}
          </span>
        ),
      },
      {
        accessorKey: 'supports_batching',
        header: 'BATCHING',
        cell: (info) => (
          <span className="text-xs text-gray-700 dark:text-gray-300">
            {info.getValue<boolean>() ? 'YES' : 'NO'}
          </span>
        ),
      },
      {
        accessorKey: 'supports_input_type',
        header: 'INPUT TYPE',
        cell: (info) => (
          <span className="text-xs text-gray-700 dark:text-gray-300">
            {info.getValue<boolean>() ? 'YES' : 'NO'}
          </span>
        ),
      },
      {
        accessorKey: 'supports_dimensions',
        header: 'CONFIG DIMS',
        cell: (info) => (
          <span className="text-xs text-gray-700 dark:text-gray-300">
            {info.getValue<boolean>() ? 'YES' : 'NO'}
          </span>
        ),
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

  // Notify parent of row count changes
  const rowCount = table.getRowModel().rows.length;
  if (onRowCountChange) {
    onRowCountChange(rowCount);
  }

  return (
    <div className="w-full">
      {/* Table */}
      <div className="border-t border-gray-200 dark:border-gray-800">
        <table className="w-full text-xs">
          <thead className="sticky top-[49px] z-40 bg-gray-50 dark:bg-gray-900">
              {table.getHeaderGroups().map((headerGroup) => (
                <tr key={headerGroup.id} className="border-b border-gray-200 dark:border-gray-800">
                  {headerGroup.headers.map((header) => (
                    <th
                      key={header.id}
                      className="px-4 py-2.5 text-left text-xs font-semibold text-gray-700 dark:text-gray-300 uppercase tracking-wide cursor-pointer hover:bg-gray-100 dark:hover:bg-gray-800 bg-gray-50 dark:bg-gray-900"
                      onClick={header.column.getToggleSortingHandler()}
                    >
                      <div className="flex items-center space-x-1">
                        <span>
                          {flexRender(
                            header.column.columnDef.header,
                            header.getContext()
                          )}
                        </span>
                        {header.column.getIsSorted() && (
                          <span className="text-gray-900 dark:text-gray-100">
                            {header.column.getIsSorted() === 'asc' ? '↑' : '↓'}
                          </span>
                        )}
                      </div>
                    </th>
                  ))}
                </tr>
              ))}
            </thead>
            <tbody className="bg-white dark:bg-gray-950 divide-y divide-gray-100 dark:divide-gray-800">
              {table.getRowModel().rows.map((row) => (
                <tr
                  key={row.id}
                  className="hover:bg-gray-50 dark:hover:bg-gray-900/30 transition-colors"
                >
                  {row.getVisibleCells().map((cell) => (
                    <td key={cell.id} className="px-4 py-2.5">
                      {flexRender(cell.column.columnDef.cell, cell.getContext())}
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
      </div>
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
