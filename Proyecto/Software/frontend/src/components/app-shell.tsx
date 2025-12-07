"use client";

import Link from 'next/link';
import { usePathname } from 'next/navigation';
import { cn } from '@/lib/utils';
import { Button } from '@/components/ui/button';
import {
  LayoutDashboard,
  HeartPulse,
  History,
  User,
} from 'lucide-react';
import { CardioCalmLogo } from './icons';
import { Avatar, AvatarFallback, AvatarImage } from './ui/avatar';

const navItems = [
  { href: '/dashboard/analyze', label: 'Analizar', icon: HeartPulse, aliases: ['/dashboard/live-ecg', '/dashboard/emotion'] },
  { href: '/dashboard', label: 'Dashboard', icon: LayoutDashboard },
  { href: '/dashboard/history', label: 'Historial', icon: History },
  { href: '/dashboard/profile', label: 'Perfil', icon: User },
];

export function DesktopAppShell({ children }: { children: React.ReactNode }) {
  const pathname = usePathname();

  const isNavItemActive = (item: typeof navItems[0]) => {
    if (pathname === item.href) return true;
    if (item.aliases?.includes(pathname)) return true;
    // Special case for dashboard root
    if (item.href === '/dashboard' && pathname === '/dashboard') return true;
    if (item.href === '/dashboard' && (pathname.startsWith('/dashboard/') && !navItems.some(nav => nav.href !== '/dashboard' && pathname.startsWith(nav.href)))) return false;
    if (item.href !== '/dashboard' && pathname.startsWith(item.href)) return true;

    return false;
  };


  return (
    <div className="flex min-h-screen w-full">
      <aside className="w-64 flex-shrink-0 border-r bg-card p-4 flex flex-col gap-8">
        <div className="flex items-center gap-3 px-2">
          <CardioCalmLogo className="h-8 w-8 text-primary" />
          <h1 className="text-xl font-bold text-primary font-headline">
            CardioCalm
          </h1>
        </div>
        <nav className="flex-grow">
          <ul className="space-y-2">
            {navItems.map((item) => (
              <li key={item.href}>
                <Button
                  asChild
                  variant={isNavItemActive(item) ? 'default' : 'ghost'}
                  className="w-full justify-start text-base"
                >
                  <Link href={item.href}>
                    <item.icon className="mr-3 h-5 w-5" />
                    {item.label}
                  </Link>
                </Button>
              </li>
            ))}
          </ul>
        </nav>
        <div className="mt-auto">
           <div className="flex items-center gap-3">
              <Avatar>
                <AvatarImage src="https://picsum.photos/seed/user/40/40" data-ai-hint="person face" />
                <AvatarFallback>JD</AvatarFallback>
              </Avatar>
              <div>
                <p className="font-semibold">John Doe</p>
                <p className="text-sm text-muted-foreground">john.doe@email.com</p>
              </div>
            </div>
        </div>
      </aside>
      <main className="flex-1 p-6 lg:p-8 bg-background overflow-auto">
        {children}
      </main>
    </div>
  );
}

export function MobileAppShell({ children }: { children: React.ReactNode }) {
  const pathname = usePathname();

  const isNavItemActive = (item: typeof navItems[0]) => {
    if (pathname === item.href) return true;
    if (item.aliases?.includes(pathname)) return true;
    if (item.href === '/dashboard' && pathname === '/dashboard') return true;
    if (item.href === '/dashboard' && (pathname.startsWith('/dashboard/') && !navItems.some(nav => nav.href !== '/dashboard' && pathname.startsWith(nav.href)))) return false;
    if (item.href !== '/dashboard' && pathname.startsWith(item.href)) return true;

    return false;
  };

  return (
    <div className="min-h-screen w-full bg-background">
      <main className="p-4 sm:p-6 lg:p-8 pb-24">{children}</main>
      <footer className="fixed bottom-0 left-0 right-0 bg-card border-t z-10">
        <nav className="grid grid-cols-4 items-center justify-items-center h-20">
          {navItems.map((item) => (
            <Link
              key={item.href}
              href={item.href}
              className={cn(
                'flex flex-col items-center gap-1 p-2 rounded-md transition-colors w-full',
                isNavItemActive(item)
                  ? 'text-primary'
                  : 'text-muted-foreground hover:text-primary'
              )}
            >
              <item.icon className="h-6 w-6" />
              <span className="text-xs font-medium">{item.label}</span>
            </Link>
          ))}
        </nav>
      </footer>
    </div>
  );
}
