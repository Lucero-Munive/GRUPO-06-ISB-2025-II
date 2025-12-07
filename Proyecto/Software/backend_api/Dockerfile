# Usamos una imagen base de Python ligera
FROM python:3.10-slim

# Instalar dependencias del sistema (CORREGIDO: Reemplazamos libgl1-mesa-glx por libgl1)
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Configurar directorio de trabajo
WORKDIR /app

# Copiar dependencias e instalarlas
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copiar el resto de archivos (código, modelos, llaves)
COPY . .

# Exponer el puerto que usa Cloud Run
ENV PORT=8080

# Comando de arranque
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]