'use client';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Button } from '@/components/ui/button';
import { Avatar, AvatarFallback, AvatarImage } from '@/components/ui/avatar';
import { useAuth } from '@/context/AuthContext';

export default function ProfilePage() {
  const { user } = useAuth();

  return (
    <div>
      <h1 className="text-3xl font-bold mb-6 font-headline">Perfil</h1>
      <Card className="max-w-2xl mx-auto rounded-2xl">
        <CardHeader>
          <CardTitle>Información de Usuario</CardTitle>
          <CardDescription>Gestiona tus datos personales y configuración.</CardDescription>
        </CardHeader>
        <CardContent className="space-y-6">
          <div className="flex items-center gap-4">
            <Avatar className="h-20 w-20">
              <AvatarImage src={user?.photoURL || "https://picsum.photos/seed/user/80/80"} />
              <AvatarFallback>{user?.displayName?.charAt(0) || "U"}</AvatarFallback>
            </Avatar>
            <Button variant="outline">Cambiar Foto</Button>
          </div>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div className="space-y-2">
              <Label htmlFor="name">Nombre</Label>
              <Input id="name" defaultValue={user?.displayName || "Usuario"} />
            </div>
            <div className="space-y-2">
              <Label htmlFor="email">Email</Label>
              <Input id="email" type="email" defaultValue={user?.email || ""} disabled />
            </div>
          </div>
          <div className="space-y-2">
            <Label htmlFor="age">Edad</Label>
            <Input id="age" type="number" defaultValue="34" />
          </div>
          <div className="flex justify-end">
            <Button>Guardar Cambios</Button>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
