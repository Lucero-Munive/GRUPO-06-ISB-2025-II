'use client';

import { useState, useEffect, useRef } from 'react';
import { useRouter } from 'next/navigation';
import { ArrowLeft, Loader2, Camera, AlertCircle } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Card, CardContent } from '@/components/ui/card';
import { CardioCalmLogo } from '@/components/icons';
import { useToast } from '@/hooks/use-toast';

// URL de tu API local
const API_URL = "https://cardiocalm-api-65187920779.us-central1.run.app/predict/emotion";

export default function EmotionDetectionPage() {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const router = useRouter();
  const { toast } = useToast();

  const [hasPermission, setHasPermission] = useState<boolean | null>(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);

  // Estado del resultado
  const [result, setResult] = useState<{
    emotion: string;
    confidence: number;
    message: string;
  } | null>(null);

  // Iniciar cámara al cargar
  useEffect(() => {
    startCamera();
    return () => stopCamera();
  }, []);

  // Loop de análisis (Si está activo, manda foto cada 1s)
  useEffect(() => {
    let interval: NodeJS.Timeout;
    if (isAnalyzing && hasPermission) {
      interval = setInterval(captureAndAnalyze, 1000); // 1 FPS para no saturar
    }
    return () => clearInterval(interval);
  }, [isAnalyzing, hasPermission]);

  const startCamera = async () => {
    if (!navigator.mediaDevices?.getUserMedia) {
      setHasPermission(false);
      return;
    }
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: 'user', width: 640, height: 480 },
      });
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
      }
      setHasPermission(true);
    } catch (err) {
      console.error("Error cámara:", err);
      setHasPermission(false);
      toast({
        variant: 'destructive',
        title: 'Acceso denegado',
        description: 'Necesitamos acceso a tu cámara para analizar emociones.',
      });
    }
  };

  const stopCamera = () => {
    if (videoRef.current && videoRef.current.srcObject) {
      const stream = videoRef.current.srcObject as MediaStream;
      stream.getTracks().forEach((track) => track.stop());
      videoRef.current.srcObject = null;
    }
  };

  const captureAndAnalyze = async () => {
    if (!videoRef.current || !canvasRef.current) return;

    const context = canvasRef.current.getContext('2d');
    if (!context) return;

    // 1. Configurar tamaño reducido para envío rápido (480px ancho)
    const targetWidth = 480;
    const scale = targetWidth / videoRef.current.videoWidth;
    const targetHeight = videoRef.current.videoHeight * scale;

    canvasRef.current.width = targetWidth;
    canvasRef.current.height = targetHeight;

    // 2. Dibujar frame redimensionado
    context.drawImage(videoRef.current, 0, 0, targetWidth, targetHeight);

    // 3. Obtener Base64 (JPG calidad 0.8 es suficiente para IA)
    const imageBase64 = canvasRef.current.toDataURL('image/jpeg', 0.8);

    // 3. Enviar a la API
    try {
      const response = await fetch(API_URL, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ image: imageBase64 }),
      });

      const data = await response.json();

      if (data.status === "success") {
        setResult({
          emotion: data.emotion,
          confidence: data.confidence,
          message: getEmotionMessage(data.emotion)
        });
      } else if (data.status === "no_face") {
        // Opcional: Avisar que no se ve cara
      }

    } catch (error) {
      console.error("Error API Emociones:", error);
    }
  };

  // Mensajes personalizados según emoción
  const getEmotionMessage = (emotion: string) => {
    const msgs: Record<string, string> = {
      "Feliz": "¡Esa sonrisa es contagiosa! Sigue así.",
      "Neutral": "Te ves en calma y equilibrio.",
      "Triste": "Parece un momento difícil. Respira profundo.",
      "Enojado": "Detectamos tensión. ¿Quizás una pausa ayudaría?",
      "Sorpresa": "¡Vaya! Algo ha llamado tu atención.",
      "Miedo": "Todo va a estar bien. Estamos contigo.",
      "Disgustado": "Algo no te agrada. Es válido sentirlo."
    };
    return msgs[emotion] || "Analizando expresión...";
  };

  const toggleAnalysis = () => {
    setIsAnalyzing(!isAnalyzing);
    if (!isAnalyzing) setResult(null); // Limpiar al reiniciar
  };

  return (
    <div className="flex flex-col h-full items-center px-4 space-y-4 pb-24 pt-6">
      <div className="w-full max-w-md">
        <Button variant="ghost" onClick={() => router.push('/dashboard/analyze')} className="-ml-4">
          <ArrowLeft className="mr-2 h-4 w-4" />
          Volver
        </Button>
      </div>

      <header className="text-center">
        <div className="flex items-center justify-center gap-2">
          <CardioCalmLogo className="h-6 w-6 text-primary" />
          <span className="text-lg font-bold text-primary font-headline">
            CardioCalm Vision
          </span>
        </div>
        <p className="text-muted-foreground mt-2">Análisis facial en tiempo real</p>
      </header>

      {/* Contenedor de Video */}
      <div className="w-full max-w-md aspect-square rounded-2xl border-2 border-dashed border-border flex items-center justify-center overflow-hidden bg-black relative shadow-xl">

        {hasPermission === null && <Loader2 className="h-8 w-8 animate-spin text-white" />}

        {hasPermission === false && (
          <div className="text-center p-6">
            <AlertCircle className="h-10 w-10 text-red-500 mx-auto mb-2" />
            <p className="text-white">Se requiere acceso a la cámara.</p>
          </div>
        )}

        {/* Video Element */}
        <video
          ref={videoRef}
          className={`w-full h-full object-cover scale-x-[-1] transition-opacity ${hasPermission ? 'opacity-100' : 'opacity-0'}`}
          autoPlay
          playsInline
          muted
        />

        {/* Canvas Oculto para capturas */}
        <canvas ref={canvasRef} width="640" height="480" className="hidden" />

        {/* Overlay de Análisis Activo */}
        {isAnalyzing && (
          <div className="absolute top-4 right-4">
            <span className="flex h-3 w-3 relative">
              <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-red-400 opacity-75"></span>
              <span className="relative inline-flex rounded-full h-3 w-3 bg-red-500"></span>
            </span>
          </div>
        )}
      </div>

      {/* Tarjeta de Resultados */}
      <div className="w-full max-w-md min-h-[140px]">
        {!isAnalyzing && !result && (
          <Card className="rounded-2xl bg-muted/30 border-dashed">
            <CardContent className="p-6 flex flex-col items-center justify-center text-center h-full">
              <Camera className="h-8 w-8 text-muted-foreground mb-2" />
              <p className="text-muted-foreground text-sm">Presiona "Iniciar Análisis" para comenzar</p>
            </CardContent>
          </Card>
        )}

        {isAnalyzing && !result && (
          <Card className="rounded-2xl bg-background shadow-sm animate-pulse">
            <CardContent className="p-6 flex flex-col items-center text-center">
              <Loader2 className="h-6 w-6 animate-spin text-primary mb-2" />
              <p className="text-sm text-muted-foreground">Escaneando rostro...</p>
            </CardContent>
          </Card>
        )}

        {result && (
          <Card className="rounded-2xl shadow-md border-l-4 border-primary transition-all duration-300">
            <CardContent className="p-6 text-left">
              <div className="flex justify-between items-center mb-1">
                <p className="text-2xl font-bold text-primary">{result.emotion}</p>
                <span className="text-xs font-mono bg-primary/10 text-primary px-2 py-1 rounded">
                  {result.confidence}%
                </span>
              </div>
              <div className="w-full bg-secondary h-2 rounded-full overflow-hidden mb-3">
                <div
                  className="bg-primary h-full transition-all duration-500 ease-out"
                  style={{ width: `${result.confidence}%` }}
                />
              </div>
              <p className="text-foreground text-sm">{result.message}</p>
            </CardContent>
          </Card>
        )}
      </div>

      <div className="w-full max-w-md pt-2">
        <Button
          size="lg"
          className={`w-full font-bold h-14 rounded-xl transition-colors ${isAnalyzing ? 'bg-destructive hover:bg-destructive/90' : ''}`}
          onClick={toggleAnalysis}
          disabled={!hasPermission}
        >
          {isAnalyzing ? 'Detener Análisis' : 'Iniciar Análisis'}
        </Button>
      </div>
    </div>
  );
}