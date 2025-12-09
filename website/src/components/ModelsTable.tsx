import { useState, useMemo, useEffect, useRef, useCallback } from 'react';
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
        header: 'PROVIDER',
        size: 100,
        cell: (info) => (
          <span className="text-xs font-medium uppercase text-gray-900 dark:text-gray-100">
            {info.getValue<string>()}
          </span>
        ),
      },
      {
        accessorKey: 'name',
        header: 'MODEL',
        size: 470,
        cell: (info) => (
          <span className="text-xs text-gray-700 dark:text-gray-300">
            {info.getValue<string>()}
          </span>
        ),
      },
      {
        accessorKey: 'dimensions',
        header: 'DIMENSIONS',
        size: 110,
        cell: (info) => (
          <span className="text-xs tabular-nums text-gray-700 dark:text-gray-300">
            {info.getValue<number>().toLocaleString()}
          </span>
        ),
      },
      {
        accessorKey: 'max_input_tokens',
        header: 'MAX TOKENS',
        size: 140,
        cell: (info) => (
          <span className="text-xs tabular-nums text-gray-700 dark:text-gray-300">
            {info.getValue<number>().toLocaleString()}
          </span>
        ),
      },
      {
        accessorKey: 'cost_per_million_tokens',
        header: 'COST/1M',
        size: 120,
        cell: (info) => (
          <span className="text-xs tabular-nums text-gray-700 dark:text-gray-300">
            ${info.getValue<number>().toFixed(2)}
          </span>
        ),
      },
      {
        accessorKey: 'mteb_score',
        header: 'MTEB',
        size: 90,
        cell: (info) => {
          const score = info.getValue<number | null>();
          return (
            <span className="text-xs tabular-nums text-gray-700 dark:text-gray-300">
              {score !== null ? score.toFixed(2) : '—'}
            </span>
          );
        },
      },
      {
        accessorKey: 'rteb_score',
        header: 'RTEB',
        size: 90,
        cell: (info) => {
          const score = info.getValue<number | null>();
          return (
            <span className="text-xs tabular-nums text-gray-700 dark:text-gray-300">
              {score !== null ? score.toFixed(2) : '—'}
            </span>
          );
        },
      },
      {
        accessorKey: 'modalities',
        header: 'MODALITY',
        size: 100,
        cell: (info) => {
          const modalities = info.getValue<string[]>() || ['text'];
          return (
            <div className="flex gap-1">
              {modalities.map((modality) => (
                <span
                  key={modality}
                  className="inline-flex items-center justify-center w-5 h-5 text-[10px] font-semibold border border-gray-300 dark:border-gray-600 rounded text-gray-700 dark:text-gray-300"
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
        header: 'QUANT',
        size: 90,
        cell: (info) => {
          const quantizations = info.getValue<string[]>() || ['float'];
          return (
            <div className="flex gap-1">
              {quantizations.map((quant) => (
                <span
                  key={quant}
                  className="inline-flex items-center justify-center w-5 h-5 text-[10px] font-semibold border border-gray-300 dark:border-gray-600 rounded text-gray-700 dark:text-gray-300"
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
        header: 'BATCHING',
        size: 80,
        cell: (info) => (
          <span className="text-xs text-gray-700 dark:text-gray-300">
            {info.getValue<boolean>() ? 'YES' : 'NO'}
          </span>
        ),
      },
      {
        accessorKey: 'supports_input_type',
        header: 'INPUT TYPE',
        size: 110,
        cell: (info) => (
          <span className="text-xs text-gray-700 dark:text-gray-300">
            {info.getValue<boolean>() ? 'YES' : 'NO'}
          </span>
        ),
      },
      {
        accessorKey: 'supports_dimensions',
        header: 'CONFIG DIMS',
        size: 120,
        cell: (info) => (
          <span className="text-xs text-gray-700 dark:text-gray-300">
            {info.getValue<boolean>() ? 'YES' : 'NO'}
          </span>
        ),
      },
      {
        accessorKey: 'release_date',
        header: 'RELEASED',
        size: 100,
        cell: (info) => {
          const dateStr = info.getValue<string | null>();
          if (!dateStr) return <span className="text-xs text-gray-400 dark:text-gray-500">—</span>;

          // Parse date string (YYYY-MM-DD format)
          const date = new Date(dateStr);
          const formatter = new Intl.DateTimeFormat('en-US', {
            year: 'numeric',
            month: 'short'
          });

          return (
            <span className="text-xs text-gray-700 dark:text-gray-300">
              {formatter.format(date)}
            </span>
          );
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

  // Refs for sticky header functionality
  const scrollContainerRef = useRef<HTMLDivElement>(null);
  const theadRef = useRef<HTMLTableSectionElement>(null);
  const [showStickyHeader, setShowStickyHeader] = useState(false);
  const [scrollLeft, setScrollLeft] = useState(0);

  // Navbar height (py-3 = 24px + h-8 = 32px content = ~56px, but measure dynamically)
  const navbarHeight = 57;

  // Handle scroll to show/hide sticky header
  useEffect(() => {
    const handleScroll = () => {
      if (theadRef.current) {
        const rect = theadRef.current.getBoundingClientRect();
        // Show sticky header when original header scrolls past navbar
        setShowStickyHeader(rect.top < navbarHeight);
      }
    };

    window.addEventListener('scroll', handleScroll, { passive: true });
    handleScroll(); // Check initial state

    return () => window.removeEventListener('scroll', handleScroll);
  }, [navbarHeight]);

  // Sync horizontal scroll
  const handleHorizontalScroll = useCallback((e: React.UIEvent<HTMLDivElement>) => {
    setScrollLeft(e.currentTarget.scrollLeft);
  }, []);

  // Get column widths for consistent sizing
  const getColumnStyle = (size: number) => ({
    width: size,
    minWidth: size,
    maxWidth: size,
  });

  // Render header row content (shared between original and sticky)
  const renderHeaderContent = () => (
    <>
      {table.getHeaderGroups().map((headerGroup) => (
        <tr key={headerGroup.id} className="border-b border-gray-200 dark:border-gray-800">
          {headerGroup.headers.map((header) => (
            <th
              key={header.id}
              className="px-4 py-2.5 text-left text-xs font-semibold text-gray-700 dark:text-gray-300 uppercase tracking-wide cursor-pointer hover:bg-gray-100 dark:hover:bg-gray-800 bg-gray-50 dark:bg-gray-900"
              style={header.getSize() !== 150 ? getColumnStyle(header.getSize()) : undefined}
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
    </>
  );

  return (
    <div className="w-full">
      {/* Sticky header that appears when scrolling */}
      {showStickyHeader && (
        <div
          className="fixed left-0 right-0 z-40 overflow-hidden border-b border-gray-200 dark:border-gray-800 bg-gray-50 dark:bg-gray-900"
          style={{ top: navbarHeight }}
        >
          <div
            style={{ transform: `translateX(-${scrollLeft}px)` }}
          >
            <table className="text-xs" style={{ tableLayout: 'fixed' }}>
              <thead className="bg-gray-50 dark:bg-gray-900">
                {renderHeaderContent()}
              </thead>
            </table>
          </div>
        </div>
      )}

      {/* Main scrollable container */}
      <div
        ref={scrollContainerRef}
        className="w-full overflow-x-auto"
        onScroll={handleHorizontalScroll}
      >
        {/* Table */}
        <div className="border-t border-gray-200 dark:border-gray-800">
          <table className="text-xs" style={{ tableLayout: 'fixed' }}>
            <thead ref={theadRef} className="bg-gray-50 dark:bg-gray-900">
              {renderHeaderContent()}
            </thead>
            <tbody className="bg-white dark:bg-gray-950 divide-y divide-gray-100 dark:divide-gray-800">
              {table.getRowModel().rows.map((row) => (
                <tr
                  key={row.id}
                  className="hover:bg-gray-50 dark:hover:bg-gray-900/30 transition-colors"
                >
                  {row.getVisibleCells().map((cell) => (
                    <td
                      key={cell.id}
                      className="px-4 py-2.5"
                      style={cell.column.getSize() !== 150 ? getColumnStyle(cell.column.getSize()) : undefined}
                    >
                      {flexRender(cell.column.columnDef.cell, cell.getContext())}
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
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
