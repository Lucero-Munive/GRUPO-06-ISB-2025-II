// frontend/src/@types/web-bluetooth.d.ts

// Definiciones de tipos para Web Bluetooth API
// Esto resuelve los errores: Cannot find name 'Bluetooth', 'BluetoothDevice', etc.

interface Navigator {
    readonly bluetooth: Bluetooth;
}

interface Bluetooth {
    requestDevice(options?: RequestDeviceOptions): Promise<BluetoothDevice>;
    // Puedes añadir otros métodos aquí si es necesario
}

interface BluetoothDevice {
    readonly id: string;
    readonly name?: string;
    readonly gatt?: BluetoothRemoteGATTServer;
    removeEventListener(type: string, listener: EventListenerOrEventListenerObject): void;
    // Para simplificar, declaramos las más usadas:
    new (id: string, name?: string): BluetoothDevice;
}

interface BluetoothRemoteGATTServer {
    readonly connected: boolean;
    connect(): Promise<BluetoothRemoteGATTServer>;
    disconnect(): void;
    getPrimaryService(service: BluetoothServiceUUID): Promise<BluetoothRemoteGATTService>;
}

interface BluetoothRemoteGATTService {
    getCharacteristic(characteristic: BluetoothCharacteristicUUID): Promise<BluetoothRemoteGATTCharacteristic>;
}

// Extender EventTarget para que TypeScript sepa que tiene addEventListener/removeEventListener
interface BluetoothRemoteGATTCharacteristic extends EventTarget {
    readonly service: BluetoothRemoteGATTService;
    readonly value?: DataView;
    startNotifications(): Promise<BluetoothRemoteGATTCharacteristic>;
    addEventListener(type: 'characteristicvaluechanged', listener: (event: Event) => void): void;
}

// Tipos utilitarios
type BluetoothServiceUUID = number | string;
type BluetoothCharacteristicUUID = number | string;
type RequestDeviceOptions = {
    filters: BluetoothLEScanFilter[];
    optionalServices?: BluetoothServiceUUID[];
};
type BluetoothLEScanFilter = {
    services?: BluetoothServiceUUID[];
};