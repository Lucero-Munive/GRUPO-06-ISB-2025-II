# Script de despliegue automático para CardioCalm FrontEnd

Write-Host "Configurando entorno de despliegue..."

# Asegurar que se usen los experimentos de frameworks web (necesario en versiones antiguas de CLI)
$env:FIREBASE_CLI_EXPERIMENTS = "webframeworks"

# Establecer el proyecto activo
Write-Host "Seleccionando proyecto cardiocalm-api-65187920779..."
firebase use studio-6590148871-6778d

# Ejecutar despliegue
Write-Host "Iniciando 'firebase deploy'..."
firebase deploy --non-interactive

Write-Host "Despliegue finalizado."
