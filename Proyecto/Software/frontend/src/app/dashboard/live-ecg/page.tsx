"use client";

import { useState, useEffect, useRef } from "react";
import { LineChart, Line, YAxis, ResponsiveContainer, XAxis } from "recharts";

// --- CONFIGURACIÓN ---
// Si vas a probar en el celular, cambia 'localhost' por la IP de tu laptop (ej. 192.168.1.15)
const WS_URL = "ws://localhost:8000/ws/live-ecg"; 
const WINDOW_SIZE = 300; // Cuántos puntos mostramos en pantalla (aprox 6 segundos)

export default function LiveECGPage() {
  const [isConnected, setIsConnected] = useState(false);
  const [chartData, setChartData] = useState<{ val: number }[]>([]);
  const [bpm, setBpm] = useState("--");
  
  // Usamos refs para manejar los datos sin renderizar a lo loco
  const dataBuffer = useRef<number[]>(new Array(WINDOW_SIZE).fill(0));
  const ws = useRef<WebSocket | null>(null);

  useEffect(() => {
    // 1. Iniciar conexión WebSocket
    connectToStream();

    // 2. Loop de renderizado suave (30 FPS)
    // Desacopla la recepción de datos (muy rápida) del dibujo (más lento)
    const interval = setInterval(() => {
      if (ws.current?.readyState === WebSocket.OPEN) {
        // Actualizamos el estado de React con lo que hay en el buffer
        const snapshot = dataBuffer.current.map((val, idx) => ({ val }));
        setChartData(snapshot);
      }
    }, 33);

    return () => {
      clearInterval(interval);
      if (ws.current) ws.current.close();
    };
  }, []);

  const connectToStream = () => {
    console.log("Conectando a WS:", WS_URL);
    const socket = new WebSocket(WS_URL);
    ws.current = socket;

    socket.onopen = () => {
      setIsConnected(true);
      console.log("WebSocket Conectado");
    };

    socket.onclose = () => {
      setIsConnected(false);
      console.log("WebSocket Desconectado");
      // Reintento automático en 3s
      setTimeout(connectToStream, 3000);
    };

    socket.onmessage = (event) => {
      try {
        // Formato esperado: { "ecg": 123.45 }
        const msg = JSON.parse(event.data);
        
        if (msg.ecg !== undefined) {
          // Push al buffer circular
          dataBuffer.current.shift(); // Sacar el viejo
          dataBuffer.current.push(msg.ecg); // Meter el nuevo
        }
      } catch (e) {
        // Ignorar errores de parseo
      }
    };
  };

  return (
    <div className="p-4 space-y-4 max-w-4xl mx-auto">
      <header className="flex justify-between items-center bg-white p-4 rounded-xl shadow-sm border border-gray-100">
        <div>
          <h1 className="text-2xl font-bold text-gray-800">Monitor ECG en Vivo</h1>
          <p className="text-sm text-gray-500">Conexión WebSocket directa</p>
        </div>
        <div className="flex items-center gap-2">
          <span className={`h-3 w-3 rounded-full ${isConnected ? "bg-green-500 animate-pulse" : "bg-red-500"}`}></span>
          <span className="text-sm font-medium text-gray-600">
            {isConnected ? "En Línea" : "Desconectado"}
          </span>
        </div>
      </header>

      {/* Tarjeta del Gráfico */}
      <div className="bg-white p-4 rounded-xl shadow-md border border-gray-100 h-96 relative">
        <div className="absolute top-4 right-4 z-10 bg-black/5 p-2 rounded text-xs">
          Lead II (Filtrada)
        </div>
        
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={chartData}>
            <YAxis domain={['auto', 'auto']} hide={true} />
            <XAxis hide={true} />
            <Line 
              type="monotone" 
              dataKey="val" 
              stroke="#2563eb" // Azul médico
              strokeWidth={2} 
              dot={false} 
              isAnimationActive={false} // CRÍTICO: Desactivar animación para rendimiento real-time
            />
          </LineChart>
        </ResponsiveContainer>
      </div>

      <div className="grid grid-cols-2 gap-4">
        <div className="bg-blue-50 p-6 rounded-xl border border-blue-100 text-center">
          <h3 className="text-blue-500 font-medium mb-1">Estado del Sistema</h3>
          <p className="text-2xl font-bold text-blue-900">
            {isConnected ? "Capturando..." : "Esperando..."}
          </p>
        </div>
        <div className="bg-green-50 p-6 rounded-xl border border-green-100 text-center">
          <h3 className="text-green-500 font-medium mb-1">IA Backend</h3>
          <p className="text-sm text-green-800">
            Análisis corriendo en segundo plano...
          </p>
        </div>
      </div>
    </div>
  );
}