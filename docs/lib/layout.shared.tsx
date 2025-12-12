import type { BaseLayoutProps } from 'fumadocs-ui/layouts/shared';
import Image from 'next/image';
import { LayoutGrid } from 'lucide-react';

export function baseOptions(): BaseLayoutProps {
  return {
    nav: {
      title: (
        <div className="flex items-center gap-2">
          <Image
            src="/catsu-icon.png"
            alt="Catsu"
            width={24}
            height={24}
          />
          <span className="font-semibold">Catsu Docs</span>
        </div>
      ),
    },
    githubUrl: 'https://github.com/chonkie-inc/catsu',
    links: [
      {
        icon: <LayoutGrid />,
        text: 'Models Catalog',
        url: 'https://catsu.dev',
        external: true,
        secondary: false,
      },
    ],
  };
}
