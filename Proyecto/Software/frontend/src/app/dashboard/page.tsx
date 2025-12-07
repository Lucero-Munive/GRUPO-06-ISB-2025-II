'use client';

import {
  ArrowLeft,
  Download,
  MoreVertical,
  Save,
} from 'lucide-react';
import { Button } from '@/components/ui/button';
import {
  Card,
  CardContent,
  CardHeader,
  CardTitle,
} from '@/components/ui/card';
import { CardioCalmLogo } from '@/components/icons';
import { useRouter } from 'next/navigation';
import { Progress } from '@/components/ui/progress';
import { useToast } from '@/hooks/use-toast';

// Simula los datos que vendrían del backend de análisis
const analysisData = {
  result: 'Ansiedad Moderada',
  confidence: 92.3,
  summary:
    'Tu variabilidad de la frecuencia cardiaca sugiere patrones consistentes con un nivel moderado de estrés o ansiedad.',
  metrics: {
    hr: 82,
    sdrr: 45,
    rmssd: 28,
    pnn50: 7,
    approxEntropy: 1.12,
    sampleEntropy: 1.35,
  },
  factors: {
    sdrr: { label: 'Baja SDRR', value: 80, level: 'Alta' },
    hr: { label: 'Alta HR', value: 60, level: 'Media' },
    pnn50: { label: 'Baja pNN50', value: 30, level: 'Baja' },
  },
};

const factorColors = {
    Alta: 'hsl(var(--destructive))',
    Media: 'hsl(var(--chart-2))',
    Baja: 'hsl(var(--primary))'
}

export default function DashboardPage() {
  const router = useRouter();
  const { toast } = useToast();

  const handleSave = () => {
    // Aquí iría la lógica para guardar en Firestore
    toast({
      title: 'Análisis Guardado',
      description: 'Tus resultados han sido guardados en tu historial.',
    });
  };

  const handleExport = () => {
    // Aquí iría la lógica para generar y descargar un PDF/JSON
    const dataStr =
      'data:text/json;charset=utf-8,' +
      encodeURIComponent(JSON.stringify(analysisData, null, 2));
    const downloadAnchorNode = document.createElement('a');
    downloadAnchorNode.setAttribute('href', dataStr);
    downloadAnchorNode.setAttribute('download', 'analisis_cardiocalm.json');
    document.body.appendChild(downloadAnchorNode);
    downloadAnchorNode.click();
    downloadAnchorNode.remove();
    toast({
      title: 'Exportación Completa',
      description: 'Se ha descargado el archivo con tus resultados.',
    });
  };

  return (
    <div className="flex flex-col h-full bg-background/50 pb-24">
      {/* --- Header --- */}
      <header className="p-4 flex items-center justify-between sticky top-0 bg-background/80 backdrop-blur-sm z-10 -mx-4 px-4">
        <Button variant="ghost" size="icon" onClick={() => router.back()}>
          <ArrowLeft />
        </Button>
        <div className="flex items-center gap-2">
          <CardioCalmLogo className="h-6 w-6 text-primary" />
          <span className="text-lg font-bold text-primary font-headline">
            CardioCalm AI
          </span>
        </div>
        <Button variant="ghost" size="icon">
          <MoreVertical />
        </Button>
      </header>

      <main className="flex-grow flex flex-col items-center px-4 space-y-6">
        {/* --- Resultado del Análisis --- */}
        <Card className="w-full rounded-2xl shadow-lg">
          <CardContent className="p-6 text-center">
            <p className="text-sm text-muted-foreground">
              Resultado del Análisis
            </p>
            <p className="text-4xl font-bold text-yellow-500 my-2">
              {analysisData.result}
            </p>
            <p className="text-xs text-muted-foreground mb-4">
              (Confianza: {analysisData.confidence}%)
            </p>
            <p className="text-sm text-foreground max-w-xs mx-auto">
              {analysisData.summary}
            </p>
          </CardContent>
        </Card>

        {/* --- Métricas Clave --- */}
        <Card className="w-full rounded-2xl shadow-lg">
          <CardHeader>
            <CardTitle>Métricas Clave</CardTitle>
          </CardHeader>
          <CardContent className="grid grid-cols-2 gap-x-8 gap-y-6">
            <MetricItem
              label="Frecuencia Cardíaca (HR)"
              value={analysisData.metrics.hr}
              unit="lpm"
            />
            <MetricItem
              label="SDRR"
              value={analysisData.metrics.sdrr}
              unit="ms"
            />
            <MetricItem
              label="rMSSD"
              value={analysisData.metrics.rmssd}
              unit="ms"
            />
            <MetricItem
              label="pNN50"
              value={analysisData.metrics.pnn50}
              unit="%"
            />
            <MetricItem
              label="Entropía Aproximada"
              value={analysisData.metrics.approxEntropy}
            />
            <MetricItem
              label="Entropía Muestral"
              value={analysisData.metrics.sampleEntropy}
            />
          </CardContent>
        </Card>

        {/* --- Factores de Influencia --- */}
        <Card className="w-full rounded-2xl shadow-lg">
          <CardHeader>
            <CardTitle>Factores de Influencia</CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            {Object.values(analysisData.factors).map((factor) => (
              <InfluenceFactor
                key={factor.label}
                label={factor.label}
                value={factor.value}
                level={factor.level as keyof typeof factorColors}
              />
            ))}
          </CardContent>
        </Card>
      </main>

       {/* --- Action Buttons --- */}
      <footer className="p-4 grid grid-cols-2 gap-4 sticky bottom-20">
          <Button size="lg" className="font-bold text-lg h-14 rounded-xl" onClick={handleSave}>
            <Save className="mr-2"/> Guardar
          </Button>
          <Button size="lg" variant="outline" className="font-bold text-lg h-14 rounded-xl bg-card" onClick={handleExport}>
            <Download className="mr-2"/> Exportar
          </Button>
        </footer>

    </div>
  );
}

function MetricItem({
  label,
  value,
  unit,
}: {
  label: string;
  value: number;
  unit?: string;
}) {
  return (
    <div>
      <p className="text-sm text-muted-foreground">{label}</p>
      <p className="text-2xl font-bold text-foreground">
        {value} <span className="text-lg font-medium">{unit}</span>
      </p>
    </div>
  );
}

function InfluenceFactor({
    label,
    value,
    level
}: {
    label: string;
    value: number;
    level: keyof typeof factorColors
}) {
    const color = factorColors[level];
    return (
        <div>
            <div className="flex justify-between items-center mb-1 text-sm">
                <span className="text-muted-foreground">{label}</span>
                <span className="font-semibold" style={{ color: color }}>{level}</span>
            </div>
            <div className="relative h-2 w-full overflow-hidden rounded-full bg-secondary">
                 <div
                    className="absolute h-full transition-all"
                    style={{ width: `${value}%`, backgroundColor: color }}
                />
            </div>
        </div>
    )
}
