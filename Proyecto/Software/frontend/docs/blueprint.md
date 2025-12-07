# **App Name**: CardioCalm AI

## Core Features:

- Anxiety Status Dashboard: Display a circular gauge showing the user's current anxiety level (Low/Moderate/High) based on analyzed data.
- Physiological Signal Analysis: Analyze ECG and GSR data, processed using cloud functions, to detect anxiety levels.
- Data Upload (Analyze Mode): Allow users to upload .csv or .txt files containing physiological data for analysis and display raw data immediately after upload on a line chart.
- Live Sensor Connection (Analyze Mode): Enable connection to an ESP32 device via Web Bluetooth API to capture live physiological data.
- Emotion Validation (Analyze Mode): Incorporate facial emotion recognition via the camera as a tool for assessing patient state in conjunction with physiological measurements, and storing the 'mood_label' with session data.
- Session History: Record past anxiety sessions with mood labels, metrics, anxiety scores, and a link to the raw sensor data; securely store the session histories and raw data file links.
- User Profile Management: Enable users to create and manage their profiles, storing personal details securely. User info is stored to keep record of the user.

## Style Guidelines:

- Primary color: Petroleum Blue (#1A4B84) to evoke a clinical yet calming atmosphere.
- Background color: Very light blue (#E8F0F7), nearly white, to provide a clean, neutral backdrop that is easy on the eyes.
- Accent color: Mint Green (#6ED0B2) for interactive elements and to represent health and well-being.
- Body and headline font: 'Inter', a sans-serif font providing a modern, neutral and clean reading experience for both headlines and body text.
- Use linear medical icons to maintain a professional and clean aesthetic.
- Employ large, rounded cards (20px border radius) and generous whitespace to create a comfortable, zen-like user interface, like Apple Health and Calm.
- Use subtle Framer Motion animations for smooth transitions and interactions, enhancing the user experience without causing distraction.