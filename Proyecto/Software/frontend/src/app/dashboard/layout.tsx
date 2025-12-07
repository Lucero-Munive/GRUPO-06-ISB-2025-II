'use client';

import { ReactNode, useState, useEffect } from 'react';
import { DesktopAppShell, MobileAppShell } from '@/components/app-shell';


export default function DashboardLayout({ children }: { children: ReactNode }) {
  const useMediaQuery = (query: string): boolean => {
    // Evita el error "useState is not a function" durante el renderizado en servidor
    if (typeof window === 'undefined') {
      return false;
    }
    
    // eslint-disable-next-line react-hooks/rules-of-hooks
    const [matches, setMatches] = useState<boolean>(() => {
      return window.matchMedia(query).matches;
    });

    // eslint-disable-next-line react-hooks/rules-of-hooks
    useEffect(() => {
      const media = window.matchMedia(query);
      if (media.matches !== matches) {
        setMatches(media.matches);
      }
      const listener = () => {
        setMatches(media.matches);
      };
      media.addEventListener("change", listener);
      return () => media.removeEventListener("change", listener);
    }, [matches, query]);

    return matches;
  }
  
  const isMobile = useMediaQuery('(max-width: 768px)');
  
  if (isMobile) {
    return <MobileAppShell>{children}</MobileAppShell>;
  }

  return <DesktopAppShell>{children}</DesktopAppShell>;
}
