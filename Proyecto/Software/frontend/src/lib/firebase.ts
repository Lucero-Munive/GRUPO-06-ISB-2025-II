import { initializeApp, getApps, getApp } from "firebase/app";
import { getFirestore } from "firebase/firestore";
import { getAuth } from "firebase/auth";

// Tu configuración real del proyecto 'studio-...' que vimos en la consola
const firebaseConfig = {
  apiKey: "AIzaSyDChlcc7jLU0E4j-b3d7jCbFl08tAD4ncI",
  authDomain: "studio-6590148871-6778d.firebaseapp.com",
  projectId: "studio-6590148871-6778d",
  storageBucket: "studio-6590148871-6778d.firebasestorage.app",
  messagingSenderId: "65187920779",
  appId: "1:65187920779:web:01f78994224d80f802cd8e"
};

// Inicialización Singleton (Evita que Next.js se queje si recargas la página)
const app = !getApps().length ? initializeApp(firebaseConfig) : getApp();

export const db = getFirestore(app);
export const auth = getAuth(app);
export default app;