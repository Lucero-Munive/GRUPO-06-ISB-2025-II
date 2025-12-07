'use client';

import { cn } from '@/lib/utils';

type AnxietyGaugeProps = {
  value: number; // 0-100
  level: 'Bajo' | 'Moderado' | 'Alto' | string;
  size?: number;
  strokeWidth?: number;
  className?: string;
};

const levelColors = {
  Bajo: 'var(--chart-2)',
  Moderado: 'var(--chart-4)',
  Alto: 'var(--destructive)',
};

export function AnxietyGauge({
  value,
  level,
  size = 250,
  strokeWidth = 20,
  className,
}: AnxietyGaugeProps) {
  const radius = (size - strokeWidth) / 2;
  const circumference = 2 * Math.PI * radius;
  const offset = circumference - (value / 100) * circumference;

  const color = levelColors[level as keyof typeof levelColors] || 'hsl(var(--muted))';

  return (
    <div
      className={cn('relative flex items-center justify-center', className)}
      style={{ width: size, height: size }}
    >
      <svg width={size} height={size} className="transform -rotate-90">
        <circle
          cx={size / 2}
          cy={size / 2}
          r={radius}
          stroke="hsl(var(--muted))"
          strokeWidth={strokeWidth}
          fill="transparent"
          strokeDasharray={circumference}
          strokeLinecap="round"
        />
        <circle
          cx={size / 2}
          cy={size / 2}
          r={radius}
          stroke={color}
          strokeWidth={strokeWidth}
          fill="transparent"
          strokeDasharray={circumference}
          strokeDashoffset={offset}
          strokeLinecap="round"
          style={{ transition: 'stroke-dashoffset 0.5s ease-out' }}
        />
      </svg>
      <div className="absolute flex flex-col items-center text-center">
        <span className="text-5xl font-bold font-headline" style={{ color }}>
          {level}
        </span>
        <span className="text-sm text-muted-foreground mt-1">Nivel de Ansiedad</span>
      </div>
    </div>
  );
}
