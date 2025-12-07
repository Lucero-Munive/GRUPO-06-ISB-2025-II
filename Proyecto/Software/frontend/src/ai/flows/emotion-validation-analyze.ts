'use server';

/**
 * @fileOverview Implementa el reconocimiento de emociones faciales para proporcionar una etiqueta de estado de ánimo para el análisis de la sesión de ansiedad.
 *
 * - analyzeEmotion - Una función que procesa los datos de emoción facial para etiquetar el estado de ánimo.
 * - AnalyzeEmotionInput - El tipo de entrada para la función analyzeEmotion.
 * - AnalyzeEmotionOutput - El tipo de retorno para la función analyzeEmotion.
 */

import {ai} from '@/ai/genkit';
import {z} from 'genkit';

const AnalyzeEmotionInputSchema = z.object({
  faceDataUri: z
    .string()
    .describe(
      "Un data URI que contiene la imagen facial para el análisis de emociones, incluyendo el tipo MIME y la codificación Base64 (ej., data:<mimetype>;base64,<encoded_data>)."
    ),
});

export type AnalyzeEmotionInput = z.infer<typeof AnalyzeEmotionInputSchema>;

const AnalyzeEmotionOutputSchema = z.object({
  moodLabel: z
    .string()
    .describe('La etiqueta de ánimo detectada basada en el reconocimiento de emociones faciales (ej., Feliz, Triste, Ansioso).'),
});

export type AnalyzeEmotionOutput = z.infer<typeof AnalyzeEmotionOutputSchema>;

export async function analyzeEmotion(
  input: AnalyzeEmotionInput
): Promise<AnalyzeEmotionOutput> {
  return analyzeEmotionFlow(input);
}

const analyzeEmotionPrompt = ai.definePrompt({
  name: 'analyzeEmotionPrompt',
  input: {schema: AnalyzeEmotionInputSchema},
  output: {schema: AnalyzeEmotionOutputSchema},
  prompt: `Analiza la expresión facial en la imagen proporcionada y determina el estado de ánimo del usuario. Responde en español.

Imagen Facial: {{media url=faceDataUri}}

Basado en la expresión facial, clasifica el estado de ánimo del usuario en una de las siguientes categorías: Feliz, Triste, Enojado, Sorprendido, Neutral, Ansioso.

Etiqueta de Ánimo: `,
});

const analyzeEmotionFlow = ai.defineFlow(
  {
    name: 'analyzeEmotionFlow',
    inputSchema: AnalyzeEmotionInputSchema,
    outputSchema: AnalyzeEmotionOutputSchema,
  },
  async input => {
    const {output} = await analyzeEmotionPrompt(input);
    return output!;
  }
);
