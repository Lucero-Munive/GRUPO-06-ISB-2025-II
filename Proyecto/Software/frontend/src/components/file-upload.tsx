'use client';

import { useState, type DragEvent, type ChangeEvent } from 'react';
import { UploadCloud, FileText } from 'lucide-react';
import { cn } from '@/lib/utils';
import { Button } from './ui/button';

interface FileUploadProps {
  onFileAccepted: (content: string, fileName: string) => void;
}

export function FileUpload({ onFileAccepted }: FileUploadProps) {
  const [dragging, setDragging] = useState(false);
  const [file, setFile] = useState<File | null>(null);

  const handleDragEnter = (e: DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    e.stopPropagation();
    setDragging(true);
  };

  const handleDragLeave = (e: DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    e.stopPropagation();
    setDragging(false);
  };

  const handleDragOver = (e: DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    e.stopPropagation();
  };

  const handleDrop = (e: DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    e.stopPropagation();
    setDragging(false);
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      handleFile(e.dataTransfer.files[0]);
      e.dataTransfer.clearData();
    }
  };

  const handleFileChange = (e: ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files.length > 0) {
      handleFile(e.target.files[0]);
    }
  };

  const handleFile = (selectedFile: File) => {
    if (selectedFile && (selectedFile.type === 'text/csv' || selectedFile.type === 'text/plain')) {
      const reader = new FileReader();
      reader.onload = (loadEvent) => {
        const content = loadEvent.target?.result as string;
        setFile(selectedFile);
        onFileAccepted(content, selectedFile.name);
      };
      reader.readAsText(selectedFile);
    } else {
      alert('Por favor, sube un archivo .csv o .txt.');
    }
  };

  return (
    <div
      onDragEnter={handleDragEnter}
      onDragLeave={handleDragLeave}
      onDragOver={handleDragOver}
      onDrop={handleDrop}
      className={cn(
        'border-2 border-dashed rounded-2xl p-8 text-center transition-colors',
        dragging ? 'border-primary bg-accent/20' : 'border-border hover:border-primary/50'
      )}
    >
      <input
        type="file"
        id="file-upload"
        className="hidden"
        accept=".csv,.txt"
        onChange={handleFileChange}
      />
      {file ? (
        <div className="flex flex-col items-center gap-4">
          <FileText className="w-16 h-16 text-primary" />
          <p className="font-semibold text-lg">{file.name}</p>
          <p className="text-muted-foreground text-sm">
            Listo para analizar.
          </p>
          <Button onClick={() => document.getElementById('file-upload')?.click()}>
            Elegir otro archivo
          </Button>
        </div>
      ) : (
        <div className="flex flex-col items-center gap-4">
          <UploadCloud className="w-16 h-16 text-muted-foreground" />
          <p className="font-semibold text-lg">Arrastra y suelta tu archivo de datos aquí</p>
          <p className="text-muted-foreground text-sm">Se aceptan archivos .csv o .txt</p>
          <Button asChild variant="outline">
            <label htmlFor="file-upload" className="cursor-pointer">
              O busca en tus archivos
            </label>
          </Button>
        </div>
      )}
    </div>
  );
}
