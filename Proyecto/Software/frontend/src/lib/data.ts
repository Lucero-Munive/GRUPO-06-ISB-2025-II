export type Session = {
  sessionId: string;
  timestamp: string;
  mood_label: string;
  anxiety_score: number;
  calculated_hrv: {
    mean_rr: number;
    sdrr: number;
    rmssd: number;
    pnn50: number;
  };
  raw_data_url: string;
};

export const mockSessionHistory: Session[] = [
  {
    sessionId: 'sess_1a2b3c',
    timestamp: '2024-07-28T10:30:00Z',
    mood_label: 'Ansioso',
    anxiety_score: 78,
    calculated_hrv: {
      mean_rr: 750,
      sdrr: 60,
      rmssd: 35,
      pnn50: 12,
    },
    raw_data_url: '/data/session_1a2b3c.csv',
  },
  {
    sessionId: 'sess_4d5e6f',
    timestamp: '2024-07-27T15:00:00Z',
    mood_label: 'Neutral',
    anxiety_score: 45,
    calculated_hrv: {
      mean_rr: 850,
      sdrr: 50,
      rmssd: 45,
      pnn50: 18,
    },
    raw_data_url: '/data/session_4d5e6f.csv',
  },
  {
    sessionId: 'sess_7g8h9i',
    timestamp: '2024-07-26T09:15:00Z',
    mood_label: 'Calmado',
    anxiety_score: 22,
    calculated_hrv: {
      mean_rr: 950,
      sdrr: 40,
      rmssd: 55,
      pnn50: 25,
    },
    raw_data_url: '/data/session_7g8h9i.csv',
  },
  {
    sessionId: 'sess_j1k2l3',
    timestamp: '2024-07-25T18:45:00Z',
    mood_label: 'Estresado',
    anxiety_score: 85,
    calculated_hrv: {
      mean_rr: 720,
      sdrr: 65,
      rmssd: 30,
      pnn50: 8,
    },
    raw_data_url: '/data/session_j1k2l3.csv',
  },
];


// --- New Data for History Page ---

export type RecentSession = {
  id: string;
  level: 'Bajo' | 'Moderado' | 'Alto';
  timestamp: string;
  hrv: number;
};

export type HistoryData = {
  label: string;
  anxietyTrend: {
    level: 'Bajo' | 'Moderado' | 'Alto';
    change: number;
  };
  hrvTrend: {
    average: number;
    change: number;
  };
  anxietyData: { day: string; score: number }[];
  hrvData: { day: string; value: number }[];
  recentSessions: RecentSession[];
};

export const mockWeeklyData: HistoryData = {
  label: '7 días',
  anxietyTrend: {
    level: 'Moderado',
    change: 5,
  },
  hrvTrend: {
    average: 65,
    change: -3,
  },
  anxietyData: [
    { day: 'L', score: 60 },
    { day: 'M', score: 55 },
    { day: 'X', score: 75 },
    { day: 'J', score: 40 },
    { day: 'V', score: 35 },
    { day: 'S', score: 68 },
    { day: 'D', score: 72 },
  ],
  hrvData: [
    { day: 'L', value: 70 },
    { day: 'M', value: 75 },
    { day: 'X', value: 60 },
    { day: 'J', value: 80 },
    { day: 'V', value: 85 },
    { day: 'S', value: 62 },
    { day: 'D', value: 58 },
  ],
  recentSessions: [
    { id: '1', level: 'Alto', timestamp: '2024-10-24T10:15:00Z', hrv: 48 },
    { id: '2', level: 'Moderado', timestamp: '2024-10-23T20:30:00Z', hrv: 52 },
    { id: '3', level: 'Bajo', timestamp: '2024-10-23T09:05:00Z', hrv: 71 },
    { id: '4', level: 'Bajo', timestamp: '2024-10-22T18:45:00Z', hrv: 68 },
  ],
};

export const mockMonthlyData: HistoryData = {
  label: '30 días',
  anxietyTrend: {
    level: 'Bajo',
    change: -12,
  },
  hrvTrend: {
    average: 72,
    change: 8,
  },
  anxietyData: Array.from({ length: 30 }, (_, i) => ({ day: `${i + 1}`, score: 40 + Math.random() * 30 - 15 })),
  hrvData: Array.from({ length: 30 }, (_, i) => ({ day: `${i + 1}`, value: 65 + Math.random() * 20 - 10 })),
  recentSessions: [
     ...mockWeeklyData.recentSessions,
     { id: '5', level: 'Moderado', timestamp: '2024-10-15T10:15:00Z', hrv: 55 },
  ]
};

export const mockYearlyData: HistoryData = {
  label: 'Año',
  anxietyTrend: {
    level: 'Bajo',
    change: -25,
  },
  hrvTrend: {
    average: 75,
    change: 15,
  },
  anxietyData: [
    { day: 'Ene', score: 70 }, { day: 'Feb', score: 65 }, { day: 'Mar', score: 60 },
    { day: 'Abr', score: 50 }, { day: 'May', score: 45 }, { day: 'Jun', score: 40 },
    { day: 'Jul', score: 35 }, { day: 'Ago', score: 42 }, { day: 'Sep', score: 55 },
    { day: 'Oct', score: 60 }, { day: 'Nov', score: 50 }, { day: 'Dic', score: 48 },
  ],
   hrvData: [
    { day: 'Ene', value: 60 }, { day: 'Feb', value: 62 }, { day: 'Mar', value: 68 },
    { day: 'Abr', value: 75 }, { day: 'May', value: 78 }, { day: 'Jun', value: 82 },
    { day: 'Jul', value: 85 }, { day: 'Ago', value: 80 }, { day: 'Sep', value: 72 },
    { day: 'Oct', value: 68 }, { day: 'Nov', value: 71 }, { day: 'Dic', value: 74 },
  ],
  recentSessions: [
    ...mockMonthlyData.recentSessions,
    { id: '6', level: 'Alto', timestamp: '2024-09-10T10:15:00Z', hrv: 45 },
  ]
};