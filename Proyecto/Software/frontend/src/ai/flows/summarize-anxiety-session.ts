'use server';

/**
 * @fileOverview Summarizes pre-calculated anxiety session metrics using AI.
 * This flow's responsibility is not to analyze raw data, but to create a human-readable summary
 * from the output of a deterministic analysis backend (e.g., Python Cloud Function).
 *
 * - summarizeAnxietySession - A function that handles the summarization of anxiety session data.
 * - SummarizeAnxietySessionInput - The input type for the summarizeAnxietySession function.
 * - SummarizeAnxietySessionOutput - The return type for the summarizeAnxietySession function.
 */

import {ai} from '@/ai/genkit';
import {z} from 'genkit';

const SummarizeAnxietySessionInputSchema = z.object({
  sessionMetrics: z
    .string()
    .describe(
      'Una cadena de texto con métricas pre-calculadas de una sesión de ansiedad, como nivel de ansiedad, RMSSD, pNN50, etc.'
    ),
});
export type SummarizeAnxietySessionInput = z.infer<typeof SummarizeAnxietySessionInputSchema>;

const SummarizeAnxietySessionOutputSchema = z.object({
  summary: z.string().describe('Un resumen conciso y empático de las métricas de la sesión de ansiedad.'),
});
export type SummarizeAnxietySessionOutput = z.infer<typeof SummarizeAnxietySessionOutputSchema>;

export async function summarizeAnxietySession(
  input: SummarizeAnxietySessionInput
): Promise<SummarizeAnxietySessionOutput> {
  return summarizeAnxietySessionFlow(input);
}

const summarizeAnxietySessionPrompt = ai.definePrompt({
  name: 'summarizeAnxietySessionPrompt',
  input: {schema: SummarizeAnxietySessionInputSchema},
  output: {schema: SummarizeAnxietySessionOutputSchema},
  prompt: `Eres un asistente de IA que escribe resúmenes empáticos de datos fisiológicos para un usuario.
  El trabajo pesado de procesamiento de señales ya ha sido hecho. Tu trabajo es traducir los números fríos y duros en una narrativa cálida y comprensible. Responde en español.
  
  Enfócate en lo que las métricas implican en términos simples. Por ejemplo, si el RMSSD es bajo y la ansiedad es alta, podrías decir "Parece que tu cuerpo estaba en un estado de alta alerta durante esta sesión, y tu ritmo cardíaco mostró menos de la variación natural que vemos en momentos de calma."

  NO inventes nuevas métricas ni contradigas los datos proporcionados.

  Métricas de Sesión Pre-calculadas: {{{sessionMetrics}}}
  
  Proporciona un resumen conciso:`,
});

const summarizeAnxietySessionFlow = ai.defineFlow(
  {
    name: 'summarizeAnxietySessionFlow',
    inputSchema: SummarizeAnxietySessionInputSchema,
    outputSchema: SummarizeAnxietySessionOutputSchema,
  },
  async input => {
    // This flow now expects metrics, not raw data.
    const {output} = await summarizeAnxietySessionPrompt({ sessionMetrics: input.sessionMetrics });
    return output!;
  }
);
