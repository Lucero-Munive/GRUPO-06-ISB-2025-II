import Link from 'next/link';
import { Button } from '@/components/ui/button';
import { CardioCalmLogo } from '@/components/icons';

export default function LandingPage() {
  return (
    <div className="flex flex-col min-h-screen bg-background">
      <header className="container mx-auto px-4 sm:px-6 lg:px-8 py-4 flex items-center justify-between">
        <Link href="/" className="flex items-center gap-3">
          <CardioCalmLogo className="h-8 w-8 text-primary" />
          <h1 className="text-2xl font-bold text-primary font-headline">
            CardioCalm AI
          </h1>
        </Link>
        <Button asChild variant="ghost">
          <Link href="/login">Entrar a la App</Link>
        </Button>
      </header>
      <main className="flex-grow flex items-center">
        <div className="container mx-auto px-4 sm:px-6 lg:px-8">
          <div className="text-center">
            <h2 className="text-4xl md:text-6xl font-extrabold font-headline text-foreground mb-4 tracking-tighter">
              Entiende tu Corazón.
              <br />
              <span className="text-primary">Calma tu Mente.</span>
            </h2>
            <p className="text-lg md:text-xl text-muted-foreground max-w-3xl mx-auto mb-8">
              CardioCalm AI analiza tus señales fisiológicas para darte
              información sobre tus niveles de ansiedad, ayudándote a alcanzar un estado de
              calma y bienestar.
            </p>
            <div className="flex gap-4 justify-center">
              <Button asChild size="lg" className="font-semibold">
                <Link href="/login">Comenzar</Link>
              </Button>
              <Button
                asChild
                size="lg"
                variant="outline"
                className="font-semibold"
              >
                <Link href="#">Saber Más</Link>
              </Button>
            </div>
          </div>
        </div>
      </main>
      <footer className="container mx-auto px-4 sm:px-6 lg:px-8 py-6 text-center text-muted-foreground text-sm">
        <p>
          &copy; {new Date().getFullYear()} CardioCalm AI. Todos los derechos reservados.
        </p>
      </footer>
    </div>
  );
}
