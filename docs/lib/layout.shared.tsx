import type { BaseLayoutProps } from 'fumadocs-ui/layouts/shared';
import Image from 'next/image';

export function baseOptions(): BaseLayoutProps {
  return {
    nav: {
      title: (
        <div className="flex items-center gap-2">
          <Image
            src="/catsu-logo.png"
            alt="Catsu"
            width={120}
            height={40}
            className="dark:invert"
          />
        </div>
      ),
    },
    links: [
      {
        text: 'Models Catalog',
        url: 'https://catsu.dev',
        external: true,
      },
      {
        text: 'GitHub',
        url: 'https://github.com/chonkie-inc/catsu',
        external: true,
      },
    ],
  };
}
