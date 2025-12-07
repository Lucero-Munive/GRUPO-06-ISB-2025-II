'use client';

import { useState, useEffect, useMemo } from 'react';
import { db } from "../../../lib/firebase"; 
import { collection, query, orderBy, onSnapshot } from "firebase/firestore";
import { Button } from '@/components/ui/button';
import { Card, CardContent } from '@/components/ui/card';
import { ArrowUp, ArrowDown, FileDown } from 'lucide-react';
import { cn } from '@/lib/utils';
import {
  AreaChart,
  Area,
  XAxis,
  ResponsiveContainer,
  Tooltip,
} from 'recharts';
import { format, subDays, isAfter } from "date-fns";
import { es } from "date-fns/locale";

// --- TIPOS DE DATOS ---
type TimeRange = 'semana' | 'mes' | 'año';

interface SessionData {
  id: string;
  prediction: string;
  confidence: number;
  timestamp: any;
}

// Estructura que espera tu UI
interface HistoryData {
  label: string;
  anxietyTrend: { level: string; change: number };
  anxietyData: any[];
  hrvTrend: { average: number; change: number };
  hrvData: any[];
  recentSessions: any[];
}

// --- CONFIGURACIÓN VISUAL ---
const levelColors: Record<string, string> = {
  Bajo: 'text-green-500',
  Moderado: 'text-yellow-500',
  Alto: 'text-red-500',
};

const levelBgColors: Record<string, string> = {
  Bajo: 'bg-green-500',
  Moderado: 'bg-yellow-500',
  Alto: 'bg-red-500',
};

export default function HistoryPage() {
  const [timeRange, setTimeRange] = useState<TimeRange>('semana');
  const [sessions, setSessions] = useState<SessionData[]>([]);
  const [loading, setLoading] = useState(true);

  // 1. Cargar Datos Reales de Firebase
  useEffect(() => {
    const q = query(collection(db, "sessions"), orderBy("timestamp", "desc"));
    const unsubscribe = onSnapshot(q, (snapshot) => {
      const data = snapshot.docs.map(doc => ({
        id: doc.id,
        ...doc.data()
      })) as SessionData[];
      setSessions(data);
      setLoading(false);
    });
    return () => unsubscribe();
  }, []);

  // 2. Procesar Datos para los Gráficos (Memoizado)
  const activeData: HistoryData = useMemo(() => {
    if (sessions.length === 0) return getEmptyData();

    // Filtrar por rango de tiempo
    const now = new Date();
    const daysToSubtract = timeRange === 'semana' ? 7 : timeRange === 'mes' ? 30 : 365;
    const cutoffDate = subDays(now, daysToSubtract);

    const filteredSessions = sessions.filter(s => {
      const date = s.timestamp?.toDate ? s.timestamp.toDate() : new Date();
      return isAfter(date, cutoffDate);
    });

    // Si no hay datos en el rango, devolver vacío
    if (filteredSessions.length === 0) return getEmptyData();

    // Transformar para Gráficos
    // Nota: Como no guardamos HRV real en DB aun, simulamos HRV basado en Stress (Stress=Bajo HRV)
    // Stress = 100 score, Baseline = 20 score
    const chartData = filteredSessions.map(s => {
      const date = s.timestamp?.toDate ? s.timestamp.toDate() : new Date();
      const isStress = s.prediction === 'Stress';
      return {
        day: format(date, timeRange === 'semana' ? 'EEE' : 'dd/MM', { locale: es }),
        fullDate: date,
        score: isStress ? 80 + (Math.random()*10) : 20 + (Math.random()*10), // Simulación visual
        hrv: isStress ? 40 + (Math.random()*10) : 80 + (Math.random()*20),  // Simulación visual
        original: s
      };
    }).reverse(); // Para que el gráfico vaya de izquierda (viejo) a derecha (nuevo)

    // Calcular Promedios y Tendencias
    const lastSession = chartData[chartData.length - 1];
    const prevSession = chartData.length > 1 ? chartData[chartData.length - 2] : lastSession;

    const avgAnxiety = chartData.reduce((acc, curr) => acc + curr.score, 0) / chartData.length;
    const avgHrv = chartData.reduce((acc, curr) => acc + curr.hrv, 0) / chartData.length;

    // Determinar Nivel Texto
    let level = "Bajo";
    if (avgAnxiety > 40) level = "Moderado";
    if (avgAnxiety > 70) level = "Alto";

    return {
      label: timeRange,
      anxietyTrend: {
        level: level,
        change: Math.round(((lastSession.score - prevSession.score) / prevSession.score) * 100) || 0
      },
      anxietyData: chartData,
      hrvTrend: {
        average: Math.round(avgHrv),
        change: Math.round(((lastSession.hrv - prevSession.hrv) / prevSession.hrv) * 100) || 0
      },
      hrvData: chartData.map(d => ({ day: d.day, value: d.hrv })),
      recentSessions: filteredSessions.slice(0, 5).map(s => ({
        id: s.id,
        level: s.prediction === 'Stress' ? 'Alto' : 'Bajo',
        timestamp: s.timestamp?.toDate ? s.timestamp.toDate() : new Date(),
        hrv: s.prediction === 'Stress' ? 45 : 85 // Valor referencial
      }))
    };
  }, [sessions, timeRange]);

  const handleExport = () => {
    alert('Exportando datos...');
  };

  if (loading) return <div className="p-8 text-center">Cargando historial...</div>;

  return (
    <div className="pb-24 p-4 md:p-8">
      <header className="flex items-center justify-between mb-6">
        <h1 className="text-3xl font-bold font-headline text-foreground">
          Historial
        </h1>
      </header>

      <div className="mb-6">
        <TimeRangeSelector
          selected={timeRange}
          onSelect={setTimeRange}
        />
      </div>

      <div className="space-y-6">
        <AnxietyTrendCard data={activeData} />
        <HrvCard data={activeData} />
        <RecentLogs sessions={activeData.recentSessions} />
      </div>

      <Button
        onClick={handleExport}
        className="fixed bottom-24 right-6 lg:bottom-10 lg:right-10 rounded-full h-16 w-16 shadow-lg z-20"
      >
        <FileDown className="h-6 w-6" />
      </Button>
    </div>
  );
}

// --- COMPONENTES UI (Tus componentes originales adaptados) ---

function TimeRangeSelector({ selected, onSelect }: { selected: TimeRange; onSelect: (range: TimeRange) => void }) {
  const ranges: TimeRange[] = ['semana', 'mes', 'año'];
  return (
    <div className="inline-flex items-center p-1 bg-muted rounded-full">
      {ranges.map((range) => (
        <Button
          key={range}
          variant={selected === range ? 'default' : 'ghost'}
          onClick={() => onSelect(range)}
          className={cn(
            'capitalize rounded-full transition-colors duration-300',
            selected === range ? 'bg-background shadow text-primary' : 'text-muted-foreground'
          )}
        >
          {range}
        </Button>
      ))}
    </div>
  );
}

function AnxietyTrendCard({ data }: { data: HistoryData }) {
  const isPositiveChange = data.anxietyTrend.change >= 0;
  return (
    <Card className="rounded-2xl shadow-sm">
      <CardContent className="p-6">
        <p className="text-sm text-muted-foreground">Tendencia de Ansiedad</p>
        <p className={`text-3xl font-bold ${levelColors[data.anxietyTrend.level] || 'text-gray-500'}`}>
          {data.anxietyTrend.level}
        </p>
        <div className="flex items-center text-sm text-muted-foreground gap-1">
          <span>Últimos {data.label}</span>
          <div className={`flex items-center ${isPositiveChange ? 'text-red-500' : 'text-green-500'}`}>
            {isPositiveChange ? <ArrowUp className="h-4 w-4" /> : <ArrowDown className="h-4 w-4" />}
            <span>{Math.abs(data.anxietyTrend.change)}%</span>
          </div>
        </div>
        <div className="h-32 w-full mt-4">
          <ResponsiveContainer>
            <AreaChart data={data.anxietyData}>
              <defs>
                <linearGradient id="anxietyColor" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor="hsl(var(--chart-2))" stopOpacity={0.4}/>
                  <stop offset="95%" stopColor="hsl(var(--chart-2))" stopOpacity={0}/>
                </linearGradient>
              </defs>
              <Tooltip contentStyle={{ borderRadius: '8px' }} labelFormatter={(l) => `Fecha: ${l}`} />
              <XAxis dataKey="day" axisLine={false} tickLine={false} tick={{ fill: '#888', fontSize: 12 }} />
              <Area type="monotone" dataKey="score" stroke="hsl(var(--chart-2))" fill="url(#anxietyColor)" strokeWidth={2} />
            </AreaChart>
          </ResponsiveContainer>
        </div>
      </CardContent>
    </Card>
  );
}

function HrvCard({ data }: { data: HistoryData }) {
    const isPositiveChange = data.hrvTrend.change >= 0;
    return (
        <Card className="rounded-2xl shadow-sm">
            <CardContent className="p-6">
                <p className="text-sm text-muted-foreground">Variabilidad (HRV)</p>
                <p className="text-4xl font-bold text-foreground">{data.hrvTrend.average} ms</p>
                 <div className="flex items-center text-sm text-muted-foreground gap-1">
                    <span>Últimos {data.label}</span>
                     <div className={`flex items-center ${isPositiveChange ? 'text-green-500' : 'text-red-500'}`}>
                        {isPositiveChange ? <ArrowUp className="h-4 w-4" /> : <ArrowDown className="h-4 w-4" />}
                        <span>{Math.abs(data.hrvTrend.change)}%</span>
                    </div>
                </div>
                <div className="h-32 w-full mt-4">
                  <ResponsiveContainer>
                    <AreaChart data={data.hrvData}>
                        <defs>
                          <linearGradient id="hrvColor" x1="0" y1="0" x2="0" y2="1">
                            <stop offset="5%" stopColor="hsl(var(--primary))" stopOpacity={0.4}/>
                            <stop offset="95%" stopColor="hsl(var(--primary))" stopOpacity={0}/>
                          </linearGradient>
                        </defs>
                      <Tooltip contentStyle={{ borderRadius: '8px' }} />
                      <XAxis dataKey="day" axisLine={false} tickLine={false} tick={{ fontSize: 12 }} />
                      <Area type="monotone" dataKey="value" stroke="hsl(var(--primary))" fill="url(#hrvColor)" strokeWidth={2} />
                    </AreaChart>
                  </ResponsiveContainer>
                </div>
            </CardContent>
        </Card>
    )
}

function RecentLogs({ sessions }: { sessions: any[] }) {
    return (
        <div>
            <h2 className="text-xl font-bold text-foreground mb-4">Registros Recientes</h2>
            <div className="space-y-3">
                {sessions.length === 0 && <p className="text-muted-foreground">No hay registros aún.</p>}
                {sessions.map(session => (
                    <Card key={session.id} className="rounded-xl shadow-sm cursor-pointer hover:bg-muted/50">
                        <CardContent className="p-4 flex items-center justify-between">
                            <div className="flex items-center gap-4">
                                <div className={cn("h-4 w-4 rounded-full", levelBgColors[session.level] || 'bg-gray-300')}></div>
                                <div>
                                    <p className="font-semibold">Nivel: {session.level}</p>
                                    <p className="text-sm text-muted-foreground">
                                        {format(session.timestamp, "dd MMM yyyy - HH:mm", { locale: es })}
                                    </p>
                                </div>
                            </div>
                            <p className="text-sm font-medium text-muted-foreground">HRV: {session.hrv} ms</p>
                        </CardContent>
                    </Card>
                ))}
            </div>
        </div>
    )
}

// Helper para datos vacíos (evita crash si no hay sesiones)
function getEmptyData(): HistoryData {
    return {
        label: 'semana',
        anxietyTrend: { level: 'Sin datos', change: 0 },
        anxietyData: [],
        hrvTrend: { average: 0, change: 0 },
        hrvData: [],
        recentSessions: []
    };
}