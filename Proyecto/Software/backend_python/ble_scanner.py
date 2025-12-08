import asyncio
from bleak import BleakScanner

async def scan():
    print("📡 Escaneando aire en busca de 'CardioCalm'...")
    
    # discover devulve una lista de dispositivos encontrados
    devices = await BleakScanner.discover()
    
    target_found = False
    
    for d in devices:
        # Imprimimos lo básico para no causar errores de atributos
        name = d.name or "Desconocido"
        address = d.address
        
        print(f"   -> Visto: {name} [{address}]")
        
        # Buscamos tu nombre exacto definido en el ESP32
        if "CardioCalm" in name:
            target_found = True
            print("\n" + "="*40)
            print(f"✅ ¡ÉXITO! ESP32 ENCONTRADO")
            print(f"   Nombre: {name}")
            print(f"   Dirección: {address}")
            print("="*40 + "\n")

    if not target_found:
        print("\n❌ No encontré 'CardioCalm-Wearable'.")
        print("   Tips: 1. ¿Está prendido el OLED?")
        print("   Tips: 2. ¿Tu PC tiene Bluetooth activado?")

if __name__ == "__main__":
    try:
        asyncio.run(scan())
    except Exception as e:
        print(f"Error fatal: {e}")