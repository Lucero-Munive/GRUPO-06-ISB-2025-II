"use client";

import { createContext, useContext, useEffect, useState, ReactNode } from "react";
import { 
  onAuthStateChanged, 
  User, 
  signInWithEmailAndPassword, 
  createUserWithEmailAndPassword, 
  signOut 
} from "firebase/auth";
import { auth } from "../lib/firebase";

interface AuthContextType {
  user: User | null;
  loading: boolean;
  login: (email: string, pass: string) => Promise<void>;
  signup: (email: string, pass: string) => Promise<void>;
  logout: () => Promise<void>;
}

const AuthContext = createContext<AuthContextType>({} as AuthContextType);

export function AuthProvider({ children }: { children: ReactNode }) {
  const [user, setUser] = useState<User | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const unsubscribe = onAuthStateChanged(auth, (currentUser) => {
      setUser(currentUser);
      setLoading(false);
      console.log("Estado Auth:", currentUser ? "Usuario Logueado" : "Sin Usuario");
    });
    return () => unsubscribe();
  }, []);

  // --- CORRECCIÓN AQUÍ ---
  // Usamos async/await y no retornamos nada para cumplir con Promise<void>
  
  const login = async (email: string, pass: string) => {
    await signInWithEmailAndPassword(auth, email, pass);
  };

  const signup = async (email: string, pass: string) => {
    await createUserWithEmailAndPassword(auth, email, pass);
  };

  const logout = async () => {
    await signOut(auth);
  };

  return (
    <AuthContext.Provider value={{ user, loading, login, signup, logout }}>
      {!loading && children} 
    </AuthContext.Provider>
  );
}

export const useAuth = () => useContext(AuthContext);