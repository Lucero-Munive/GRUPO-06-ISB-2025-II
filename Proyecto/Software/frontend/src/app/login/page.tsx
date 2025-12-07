'use client';

import { useState } from 'react'; // <--- Nuevo
import Link from 'next/link';
import { useRouter } from 'next/navigation';
import { useAuth } from '../../context/AuthContext'; // <--- Importamos nuestro contexto
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from '@/components/ui/card';
import {
  CardioCalmLogo,
  GoogleIcon,
  AppleIcon,
} from '@/components/icons';
import { Mail, Lock, EyeOff } from 'lucide-react';

export default function LoginPage() {
  const router = useRouter();
  const { login } = useAuth(); // <--- Usamos la función real

  // Estados para manejar los datos del formulario
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [error, setError] = useState('');
  const [isSubmitting, setIsSubmitting] = useState(false);

  const handleLogin = async (e: React.FormEvent) => {
    e.preventDefault();
    setError(''); // Limpiar errores previos
    setIsSubmitting(true); // Bloquear botón

    try {
      // Intentamos loguear con Firebase
      await login(email, password);
      // Si funciona, redirigimos
      router.push('/dashboard/analyze');
    } catch (err: any) {
      console.error("Error de Login:", err);
      // Traducir errores comunes de Firebase
      if (err.code === 'auth/invalid-credential' || err.code === 'auth/user-not-found' || err.code === 'auth/wrong-password') {
        setError('Correo o contraseña incorrectos.');
      } else if (err.code === 'auth/too-many-requests') {
        setError('Demasiados intentos fallidos. Intenta más tarde.');
      } else {
        setError('Ocurrió un error al iniciar sesión.');
      }
    } finally {
      setIsSubmitting(false); // Desbloquear botón
    }
  };

  return (
    <div className="flex items-center justify-center min-h-screen bg-background p-4">
      <div className="w-full max-w-md">
        <div className="flex justify-center mb-8">
          <Link href="/" className="flex items-center gap-3">
            <CardioCalmLogo className="h-10 w-10 text-primary" />
            <h1 className="text-3xl font-bold text-primary font-headline">
              CardioCalm AI
            </h1>
          </Link>
        </div>
        <Card className="rounded-2xl shadow-lg">
          <CardHeader className="text-center">
            <CardTitle className="text-2xl font-bold">
              Bienvenido de nuevo
            </CardTitle>
            <CardDescription>
              Accede a tu cuenta para continuar.
            </CardDescription>
          </CardHeader>
          <CardContent>
            <form onSubmit={handleLogin} className="space-y-6">
              
              {/* Mensaje de Error (Solo se muestra si hay error) */}
              {error && (
                <div className="bg-red-50 text-red-600 text-sm p-3 rounded-md border border-red-200 text-center">
                  {error}
                </div>
              )}

              <div className="space-y-2">
                <Label htmlFor="email">Email</Label>
                <div className="relative">
                  <Mail className="absolute left-3 top-1/2 -translate-y-1/2 h-5 w-5 text-muted-foreground" />
                  <Input
                    id="email"
                    type="email"
                    placeholder="Ingresa tu email"
                    // Quitamos defaultValue y ponemos value/onChange
                    value={email}
                    onChange={(e) => setEmail(e.target.value)}
                    className="pl-10"
                    required
                    autoComplete="email"
                  />
                </div>
              </div>
              <div className="space-y-2">
                <Label htmlFor="password">Contraseña</Label>
                <div className="relative">
                  <Lock className="absolute left-3 top-1/2 -translate-y-1/2 h-5 w-5 text-muted-foreground" />
                  <Input
                    id="password"
                    type="password"
                    placeholder="Ingresa tu contraseña"
                    // Quitamos defaultValue y ponemos value/onChange
                    value={password}
                    onChange={(e) => setPassword(e.target.value)}
                    className="pl-10 pr-10"
                    required
                    autoComplete="current-password"
                  />
                  <EyeOff className="absolute right-3 top-1/2 -translate-y-1/2 h-5 w-5 text-muted-foreground cursor-pointer" />
                </div>
                <div className="text-right">
                  <Link
                    href="#"
                    className="text-sm font-medium text-primary hover:underline"
                  >
                    ¿Olvidé mi contraseña?
                  </Link>
                </div>
              </div>
              
              <Button 
                type="submit" 
                className="w-full font-semibold" 
                size="lg"
                disabled={isSubmitting} // Deshabilitar mientras carga
              >
                {isSubmitting ? "Iniciando..." : "Iniciar Sesión"}
              </Button>
            </form>

            <div className="relative my-6">
              <div className="absolute inset-0 flex items-center">
                <span className="w-full border-t" />
              </div>
              <div className="relative flex justify-center text-xs uppercase">
                <span className="bg-card px-2 text-muted-foreground">
                  O continúa con
                </span>
              </div>
            </div>

            <div className="space-y-4">
              <Button variant="outline" className="w-full font-medium" type="button">
                <GoogleIcon className="mr-2 h-5 w-5" />
                Iniciar sesión con Google
              </Button>
              <Button variant="outline" className="w-full font-medium" type="button">
                <AppleIcon className="mr-2 h-5 w-5" />
                Iniciar sesión con Apple
              </Button>
            </div>

            <div className="mt-6 text-center text-sm">
              ¿No tienes cuenta?{' '}
              <Link href="#" className="font-medium text-primary hover:underline">
                Regístrate
              </Link>
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  );
}