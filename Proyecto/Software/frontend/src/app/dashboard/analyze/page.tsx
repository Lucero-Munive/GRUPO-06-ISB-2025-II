'use client';

import {
  ArrowLeft,
  ArrowRight,
  FileUp,
  HeartPulse,
  Smile,
  Loader2,
  AlertTriangle,
} from 'lucide-react';
import Link from 'next/link';
import { useRouter, useSearchParams } from 'next/navigation';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/card';
import { FileUpload } from '@/components/file-upload';
import { useState, useEffect, useRef, Suspense } from 'react';
import { useToast } from '@/hooks/use-toast';
import { Alert, AlertDescription, AlertTitle } from '@/components/ui/alert';

function AnalyzePageContent() {
  const router = useRouter();
  const searchParams = useSearchParams();
  const tab = searchParams.get('tab');
  const [isProcessing, setIsProcessing] = useState(false);
  const { toast } = useToast();
  const videoRef = useRef<HTMLVideoElement>(null);

  const handleFileAnalysis = (content: string, fileName: string) => {
    console.log(`Analyzing file ${fileName}`);
    setIsProcessing(true);
    toast({
      title: 'Analizando archivo...',
      description: 'Tu archivo ECG está siendo procesado por CardioCalm AI.',
    });
    // Simulate analysis delay
    setTimeout(() => {
      setIsProcessing(false);
      router.push('/dashboard');
    }, 2500);
  };
  
  const handleEmotionAnalysis = () => {
    // Stop camera stream first
    if (videoRef.current && videoRef.current.srcObject) {
      const stream = videoRef.current.srcObject as MediaStream;
      stream.getTracks().forEach(track => track.stop());
      videoRef.current.srcObject = null;
    }

    setIsProcessing(true);
    toast({
      title: 'Analizando emociones...',
      description: 'Estamos analizando la expresión facial.',
    });
    // Simulate analysis delay and result
    setTimeout(() => {
      setIsProcessing(false);
      const simulatedEmotion = 'Feliz'; // Simulate AI result
      router.push(`/dashboard/analyze?tab=emotion-result&emotion=${simulatedEmotion}`);
    }, 2500);
  };


  if (isProcessing) {
    return (
      <div className="flex flex-col h-full items-center justify-center text-center">
        <Loader2 className="h-16 w-16 text-primary animate-spin mb-4" />
        <h1 className="text-2xl font-bold text-foreground">Procesando tu análisis</h1>
        <p className="text-muted-foreground">Esto tomará solo un momento...</p>
      </div>
    );
  }


  if (tab === 'upload') {
    return <UploadView onFileAccepted={handleFileAnalysis} />;
  }

  if (tab === 'camera') {
    return <CameraView videoRef={videoRef} onStartAnalysis={handleEmotionAnalysis} />;
  }
  
  if (tab === 'emotion-result') {
    return <EmotionResultView />;
  }

  return <MainAnalyzeView />;
}

export default function AnalyzePage() {
  return (
    <Suspense fallback={<div>Cargando...</div>}>
      <AnalyzePageContent />
    </Suspense>
  );
}


function MainAnalyzeView() {
  const router = useRouter();
  return (
    <div className="flex flex-col h-full">
      <main className="flex-grow flex flex-col items-center justify-center text-center">
        <h1 className="text-3xl md:text-4xl font-bold font-headline text-foreground mb-2">
          Conecta tu Salud
        </h1>
        <p className="text-muted-foreground max-w-xs mx-auto mb-8">
          Empieza a monitorear tu bienestar emocional y físico.
        </p>

        <div className="w-full max-w-sm space-y-4 mb-8">
           <ActionCard
            title="Subir archivo ECG"
            description="Sube un archivo desde tu dispositivo."
            icon={FileUp}
            onClick={() => router.push('/dashboard/analyze?tab=upload')}
          />
          <ActionCard
            title="Conectar Wearable"
            description="Sincroniza con tu smartwatch o monitor."
            icon={HeartPulse}
            onClick={() => router.push('/dashboard/live-ecg')}
          />
          <ActionCard
            title="Detectar Emociones"
            description="Usa tu cámara para analizar expresiones."
            icon={Smile}
            onClick={() => router.push('/dashboard/emotion')}
          />
        </div>

        <div className="w-full max-w-sm space-y-4">
            <Button size="lg" className="w-full font-semibold" onClick={() => router.push('/dashboard')}>Empezar</Button>
            <Button variant="link" className="w-full text-muted-foreground" onClick={() => router.push('/dashboard')}>Hacerlo más tarde</Button>
        </div>
      </main>
    </div>
  );
}

function ActionCard({
  title,
  description,
  icon: Icon,
  onClick,
}: {
  title: string;
  description: string;
  icon: React.ElementType;
  onClick: () => void;
}) {
  return (
    <Card
      className="rounded-2xl shadow-sm hover:bg-muted/50 transition-colors cursor-pointer"
      onClick={onClick}
    >
      <CardContent className="p-4 flex items-center justify-between">
        <div className="flex items-center gap-4">
          <div className="p-2 bg-primary/10 rounded-lg">
            <Icon className="h-6 w-6 text-primary" />
          </div>
          <div className="text-left">
            <p className="font-semibold">{title}</p>
            <p className="text-sm text-muted-foreground">{description}</p>
          </div>
        </div>
        <ArrowRight className="h-5 w-5 text-muted-foreground" />
      </CardContent>
    </Card>
  );
}

function UploadView({ onFileAccepted }: { onFileAccepted: (content: string, fileName: string) => void }) {
  const router = useRouter();
  const [fileContent, setFileContent] = useState<string | null>(null);

  return (
    <div>
      <Button variant="ghost" onClick={() => router.push('/dashboard/analyze')} className="mb-4">
        <ArrowLeft className="mr-2 h-4 w-4" />
        Volver
      </Button>
      <Card className="rounded-2xl">
        <CardHeader>
          <CardTitle>Subir Archivo ECG</CardTitle>
          <CardDescription>Sube un archivo en formato .csv o .txt para que sea analizado.</CardDescription>
        </CardHeader>
        <CardContent className="space-y-6">
          <FileUpload onFileAccepted={(content, fileName) => {
            setFileContent(content);
            onFileAccepted(content, fileName);
          }} />
          <Button 
            size="lg" 
            className="w-full" 
            disabled={!fileContent}
            onClick={() => {
              if(fileContent) {
                // The onFileAccepted callback is already called by FileUpload,
                // this is just to show the user the button is active.
                // The processing logic is handled in the parent component.
              }
            }}
          >
            Empezar Análisis
          </Button>
        </CardContent>
      </Card>
    </div>
  );
}


function CameraView({ videoRef, onStartAnalysis }: { videoRef: React.RefObject<HTMLVideoElement>, onStartAnalysis: () => void }) {
  const router = useRouter();
  const [hasCameraPermission, setHasCameraPermission] = useState<boolean | null>(null);
  const { toast } = useToast();

  useEffect(() => {
    let stream: MediaStream | null = null;
    const getCameraPermission = async () => {
      if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
        setHasCameraPermission(false);
        toast({
          variant: 'destructive',
          title: 'Cámara no Soportada',
          description: 'Tu navegador no soporta el acceso a la cámara.',
        });
        return;
      }

      try {
        stream = await navigator.mediaDevices.getUserMedia({ video: true });
        setHasCameraPermission(true);

        if (videoRef.current) {
          videoRef.current.srcObject = stream;
        }
      } catch (error) {
        console.error('Error accessing camera:', error);
        setHasCameraPermission(false);
        toast({
          variant: 'destructive',
          title: 'Acceso a la Cámara Denegado',
          description: 'Por favor, habilita los permisos de cámara en tu navegador para usar esta función.',
        });
      }
    };

    getCameraPermission();
    
    return () => {
        // Cleanup: stop the camera stream when component unmounts
        if (stream) {
            stream.getTracks().forEach(track => track.stop());
        }
        if (videoRef.current && videoRef.current.srcObject) {
            const currentStream = videoRef.current.srcObject as MediaStream;
            currentStream.getTracks().forEach(track => track.stop());
            videoRef.current.srcObject = null;
        }
    }
  }, [toast, videoRef]);

  return (
    <div className="max-w-2xl mx-auto">
      <Button variant="ghost" onClick={() => router.push('/dashboard/analyze')} className="mb-4 -ml-4">
        <ArrowLeft className="mr-2 h-4 w-4" />
        Volver
      </Button>
      <Card className="rounded-2xl">
        <CardHeader>
          <CardTitle>Detectar Emociones</CardTitle>
          <CardDescription>Usa tu cámara para un análisis de expresión facial en tiempo real.</CardDescription>
        </CardHeader>
        <CardContent className="space-y-6">
          <div className="relative aspect-video w-full bg-secondary rounded-lg overflow-hidden flex items-center justify-center">
            {hasCameraPermission === null && (
              <div className="flex flex-col items-center gap-2 text-muted-foreground">
                <Loader2 className="h-8 w-8 animate-spin" />
                <p>Iniciando cámara...</p>
              </div>
            )}
             <video ref={videoRef} className="w-full h-full object-cover" autoPlay playsInline muted />
             {hasCameraPermission === false && (
                <div className="absolute inset-0 flex flex-col items-center justify-center bg-background/80 p-4">
                    <AlertTriangle className="h-10 w-10 text-destructive mb-4" />
                    <Alert variant="destructive" className="text-center">
                        <AlertTitle>Se requiere acceso a la cámara</AlertTitle>
                        <AlertDescription>
                            Por favor, permite el acceso a la cámara en la configuración de tu navegador.
                        </AlertDescription>
                    </Alert>
                </div>
            )}
          </div>
          <Button 
            size="lg" 
            className="w-full" 
            disabled={!hasCameraPermission}
            onClick={onStartAnalysis}
          >
            <Smile className="mr-2 h-5 w-5" />
            Empezar Análisis
          </Button>
        </CardContent>
      </Card>
    </div>
  );
}

function EmotionResultView() {
  const router = useRouter();
  const searchParams = useSearchParams();
  const emotion = searchParams.get('emotion') || 'desconocida';

  const emotionEmojis: { [key: string]: string } = {
    Feliz: '😊',
    Triste: '😢',
    Enojado: '😠',
    Sorprendido: '😮',
    Neutral: '😐',
    Ansioso: '😟',
    desconocida: '🤔',
  };

  return (
    <div className="flex flex-col h-full items-center justify-center text-center">
      <Card className="w-full max-w-sm rounded-2xl shadow-lg">
        <CardHeader>
          <CardTitle>Resultado del Análisis</CardTitle>
          <CardDescription>Este es el estado de ánimo que hemos detectado.</CardDescription>
        </CardHeader>
        <CardContent className="flex flex-col items-center gap-4">
          <div className="text-8xl">{emotionEmojis[emotion] || '🤔'}</div>
          <p className="text-4xl font-bold text-primary">{emotion}</p>
        </CardContent>
      </Card>
      <Button 
        onClick={() => router.push('/dashboard/analyze')} 
        className="mt-8"
        size="lg"
      >
        <ArrowLeft className="mr-2 h-5 w-5" />
        Volver a Analizar
      </Button>
    </div>
  );
}
