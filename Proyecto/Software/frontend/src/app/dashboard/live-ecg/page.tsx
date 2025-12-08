"use client";

import { useState, useEffect, useRef } from "react";
import { LineChart, Line, YAxis, ResponsiveContainer, XAxis } from "recharts";
import { Button } from '@/components/ui/button';
import { useToast } from '@/hooks/use-toast';
import { useAuth } from "../../../context/AuthContext";
// Iconos corregidos: Power para Desconexión
import { Zap, Plug, Loader2, Heart, Power } from 'lucide-react';

// --- CONFIGURACIÓN BLE Y API ---
// ¡Importante! Estas UUIDs deben coincidir con las del ESP32 firmware
const SERVICE_UUID = '4fafc201-1fb5-459e-8fcc-c9c0f99421b1';
const ECG_CHARACTERISTIC_UUID = 'beb5483e-36e1-4688-b7f5-ea07361b26a8';
const WINDOW_SIZE = 300; // 300 puntos para la gráfica (6 segundos a ~50Hz)

// URL de tu API en la nube (para enviar el bloque de 60s)
const CLOUD_API_PREDICT = "https://cardiocalm-api-65187920779.us-central1.run.app/predict"; // <--- ¡TU URL DE CLOUD RUN!

// Declaramos las interfaces de Web Bluetooth para evitar errores de TypeScript

export default function LiveECGPage() {
  const { toast } = useToast();
  const { user } = useAuth(); // Obtenemos el usuario logueado
  const [isConnected, setIsConnected] = useState(false);
  const [isCapturing, setIsCapturing] = useState(false);
  // REF PARA EVITAR STALE CLOSURES EN EL CALLBACK DE BLUETOOTH
  const isCapturingRef = useRef(false);

  const [chartData, setChartData] = useState<{ val: number }[]>([]);
  const dataBuffer = useRef<number[]>(new Array(WINDOW_SIZE).fill(0));
  const fullEcgData = useRef<number[]>([]);
  const serverRef = useRef<BluetoothDevice | null>(null);
  const [captureProgress, setCaptureProgress] = useState(0);

  // Buffer de renderizado suave
  useEffect(() => {
    const interval = setInterval(() => {
      const snapshot = dataBuffer.current.map((val) => ({ val }));
      setChartData(snapshot);
    }, 50);

    return () => clearInterval(interval);
  }, []);

  // Manejo de datos entrantes desde BLE
  const handleCharacteristicValueChanged = (event: Event) => {
    // Usamos 'as' para ignorar el error de tipo una vez que hemos añadido las interfaces globales
    const value = (event.target as BluetoothRemoteGATTCharacteristic).value;

    const view = new DataView(value!.buffer);
    const ecgValue = view.getFloat64(0, true);

    // 1. Streaming (Gráfica en vivo)
    dataBuffer.current.shift();
    dataBuffer.current.push(ecgValue);

    // 2. Captura para IA (60s) - USAMOS LA REF AQUÍ
    if (isCapturingRef.current) {
      fullEcgData.current.push(ecgValue);
      const currentSeconds = fullEcgData.current.length / 250;
      setCaptureProgress(Math.min(100, Math.round((currentSeconds / 60) * 100)));
    }
  };

  // Función para conectar
  const connectToBLE = async () => {
    // Corregimos la comprobación de tipo
    if (!('bluetooth' in navigator)) {
      toast({ variant: "destructive", title: "Error", description: "Tu navegador no soporta Web Bluetooth." });
      return;
    }
    try {
      toast({ title: "Buscando...", description: "Inicia la búsqueda BLE del Wearable...", duration: 2000 });

      const device = await navigator.bluetooth.requestDevice({
        filters: [{ name: 'CardioCalm-Wearable' } as any],
        optionalServices: [SERVICE_UUID],
      });

      if (!device.gatt) throw new Error("GATT no soportado.");

      serverRef.current = device;
      const server = await device.gatt.connect();
      const service = await server.getPrimaryService(SERVICE_UUID);
      const characteristic = await service.getCharacteristic(ECG_CHARACTERISTIC_UUID);

      await characteristic.startNotifications();
      characteristic.addEventListener('characteristicvaluechanged', handleCharacteristicValueChanged);

      setIsConnected(true);
      toast({ title: "Conexión Exitosa", description: `Conectado a ${device.name}`, duration: 3000 });

    } catch (error: any) {
      console.error("Error BLE:", error);
      toast({ variant: "destructive", title: "Error de Conexión", description: error.message || "Asegúrate de que el ESP32 esté encendido y BLE esté activo." });
    }
  };

  const startCapture = () => {
    if (!isConnected || !user) {
      toast({ variant: "destructive", title: "Error", description: "Inicia sesión y conecta el dispositivo primero." });
      return;
    }

    fullEcgData.current = [];
    setCaptureProgress(0);
    setIsCapturing(true);
    isCapturingRef.current = true; // ACTIVAR REF

    toast({ title: "Grabación Iniciada", description: "Capturando 60 segundos de ECG...", duration: 60000 });

    setTimeout(async () => {
      setIsCapturing(false);
      isCapturingRef.current = false; // DESACTIVAR REF

      toast({ title: "Grabación Completa", description: "Enviando datos a la IA en la nube...", duration: 5000 });

      const dataToSend = fullEcgData.current;
      const fs = Math.round(dataToSend.length / 60);

      try {
        const response = await fetch(CLOUD_API_PREDICT, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            data: dataToSend,
            fs: fs,
            userId: user.uid // ENVIAMOS EL UID DEL USUARIO LOGUEADO
          }),
        });

        const result = await response.json();

        if (result.status === "success") {
          toast({
            title: "Análisis Finalizado",
            description: `Predicción: ${result.prediction} con ${(result.confidence * 100).toFixed(1)}% de confianza.`,
            duration: 8000
          });
        } else {
          throw new Error(result.detail || "Error desconocido en el servidor.");
        }
      } catch (error) {
        console.error("Error en el análisis de IA:", error);
        toast({ variant: "destructive", title: "Error de IA", description: "Fallo al conectar o procesar en Cloud Run." });
      }

    }, 60000); // 60 segundos
  };

  const disconnectBLE = () => {
    if (serverRef.current && serverRef.current.gatt && serverRef.current.gatt.connected) {
      serverRef.current.gatt.disconnect();
      setIsConnected(false);
      setIsCapturing(false);
      isCapturingRef.current = false;
      toast({ title: "Desconectado", description: "Dispositivo BLE liberado.", duration: 3000 });
    }
  }

  return (
    <div className="p-4 space-y-6 max-w-4xl mx-auto">
      <header className="flex justify-between items-center bg-white p-4 rounded-xl shadow-sm border border-gray-100">
        <div>
          <h1 className="text-2xl font-bold text-gray-800">Monitor ECG en Vivo (BLE)</h1>
          <p className="text-sm text-gray-500">Conexión inalámbrica a Wearable ESP32</p>
        </div>
        <div className="flex items-center gap-2">
          <span className={`h-3 w-3 rounded-full ${isConnected ? "bg-green-500 animate-pulse" : "bg-red-500"}`}></span>
          <span className="text-sm font-medium text-gray-600">
            {isConnected ? "Conectado" : "Desconectado"}
          </span>
        </div>
      </header>

      {/* Tarjeta del Gráfico */}
      <div className="bg-white p-4 rounded-xl shadow-md border border-gray-100 h-96 relative">
        <div className="absolute top-4 right-4 z-10 bg-black/5 p-2 rounded text-xs">
          ECG Real-time
        </div>

        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={chartData}>
            <YAxis domain={['auto', 'auto']} hide={true} />
            <XAxis hide={true} />
            <Line
              type="monotone"
              dataKey="val"
              stroke="#2563eb"
              strokeWidth={2}
              dot={false}
              isAnimationActive={false}
            />
          </LineChart>
        </ResponsiveContainer>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">

        <Button
          onClick={isConnected ? disconnectBLE : connectToBLE}
          disabled={isCapturing}
          className={`w-full h-14 font-bold ${isConnected ? 'bg-red-500 hover:bg-red-600' : 'bg-blue-600 hover:bg-blue-700'}`}
        >
          {isConnected ? <><Power className="mr-2 h-5 w-5" /> Desconectar</> : <><Plug className="mr-2 h-5 w-5" /> Conectar Wearable</>}
        </Button>

        <Button
          onClick={startCapture}
          disabled={!isConnected || isCapturing}
          className={`w-full h-14 font-bold col-span-2 ${isCapturing ? 'bg-yellow-500 hover:bg-yellow-600' : 'bg-green-600 hover:bg-green-700'}`}
        >
          {isCapturing ? (
            <div className="flex items-center justify-center w-full">
              <Loader2 className="mr-2 h-5 w-5 animate-spin" />
              Grabando {Math.floor(fullEcgData.current.length / 250)}/60s
              <div className="ml-4 w-1/3 bg-gray-200 rounded-full h-2.5">
                <div className="bg-green-700 h-2.5 rounded-full" style={{ width: `${captureProgress}%` }}></div>
              </div>
            </div>
          ) : (
            <><Zap className="mr-2 h-5 w-5" /> Iniciar Análisis de 60s</>
          )}
        </Button>
      </div>
    </div>
  );
}