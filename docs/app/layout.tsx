import { RootProvider } from 'fumadocs-ui/provider/next';
import './global.css';
import { Inter } from 'next/font/google';
import type { Metadata } from 'next';

const inter = Inter({
  subsets: ['latin'],
});

export const metadata: Metadata = {
  title: {
    default: 'Catsu - Unified Embedding API Client',
    template: '%s | Catsu',
  },
  description: 'A unified Python client for embedding APIs across 11 providers with automatic retry logic, cost tracking, and type safety.',
  icons: {
    icon: '/catsu-icon.png',
  },
  openGraph: {
    title: 'Catsu Documentation',
    description: 'A unified Python client for embedding APIs',
    type: 'website',
  },
};

export default function Layout({ children }: LayoutProps<'/'>) {
  return (
    <html lang="en" className={inter.className} suppressHydrationWarning>
      <body className="flex flex-col min-h-screen">
        <RootProvider>{children}</RootProvider>
      </body>
    </html>
  );
}
