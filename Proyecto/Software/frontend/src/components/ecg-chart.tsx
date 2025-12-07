'use client';

import { Line, LineChart, XAxis, YAxis, CartesianGrid, ResponsiveContainer, Tooltip } from 'recharts';
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from './ui/card';

type EcgChartProps = {
  data: { time: number; value: number }[];
  title?: string;
  description?: string;
  isLive?: boolean;
};

export function EcgChart({ data, title, description, isLive = false }: EcgChartProps) {

  return (
    <Card className="rounded-2xl shadow-sm">
      <CardHeader>
        {title && <CardTitle>{title}</CardTitle>}
        {description && <CardDescription>{description}</CardDescription>}
      </CardHeader>
      <CardContent>
        <div className="h-[250px] w-full">
          <ResponsiveContainer width="100%" height="100%">
            <LineChart
              data={data}
              margin={{ top: 5, right: 20, left: -20, bottom: 5 }}
            >
              <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" />
              <XAxis
                dataKey="time"
                type="number"
                domain={isLive ? ['dataMin', 'dataMax'] : [0, 30]}
                tickFormatter={(time) => time.toFixed(0)}
                axisLine={false}
                label={{ value: "Tiempo (s)", position: "insideBottom", dy: 10, fill: 'hsl(var(--muted-foreground))' }}
              />
              <YAxis 
                domain={[-0.5, 1.5]}
                tickFormatter={(value) => `${value.toFixed(1)} mV`}
                axisLine={false}
                tickLine={false}
                tick={{ fill: 'hsl(var(--muted-foreground))', fontSize: 12 }}
              />
              <Tooltip 
                labelFormatter={(label) => `Tiempo: ${label.toFixed(2)}s`}
                formatter={(value: number) => [`${value.toFixed(3)} mV`, 'ECG']}
              />
              <Line
                type="monotone"
                dataKey="value"
                stroke="hsl(var(--primary))"
                strokeWidth={2}
                dot={false}
                isAnimationActive={isLive ? false : true}
              />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </CardContent>
    </Card>
  );
}
