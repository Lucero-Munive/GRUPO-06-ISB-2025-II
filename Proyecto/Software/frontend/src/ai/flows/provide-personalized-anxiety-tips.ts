'use server';
/**
 * @fileOverview A flow to provide personalized anxiety management tips based on pre-analyzed session data.
 * Gemini's role here is purely qualitative and supportive, based on metrics calculated by a deterministic backend.
 *
 * - providePersonalizedAnxietyTips - A function that returns personalized anxiety management tips.
 * - ProvidePersonalizedAnxietyTipsInput - The input type for the providePersonalizedAnxietyTips function.
 * - ProvidePersonalizedAnxietyTipsOutput - The return type for the providePersonalizedAnxietyTips function.
 */

import {ai} from '@/ai/genkit';
import {z} from 'genkit';

const ProvidePersonalizedAnxietyTipsInputSchema = z.object({
  anxietyLevel: z
    .string()
    .describe("El nivel de ansiedad pre-calculado del usuario (ej., Bajo, Moderado, Alto)."),
  sessionDataSummary: z
    .string()
    .describe('Un resumen de los datos de la sesión del usuario, incluyendo métricas clave como RMSSD calculadas por el backend.'),
  userPreferences: z
    .string()
    .optional()
    .describe('Preferencias de usuario opcionales o metas relacionadas con el manejo de la ansiedad.'),
});

export type ProvidePersonalizedAnxietyTipsInput = z.infer<
  typeof ProvidePersonalizedAnxietyTipsInputSchema
>;

const ProvidePersonalizedAnxietyTipsOutputSchema = z.object({
  tips: z
    .array(z.string())
    .describe('Una lista de consejos personalizados y accionables para manejar la ansiedad.'),
});

export type ProvidePersonalizedAnxietyTipsOutput = z.infer<
  typeof ProvidePersonalizedAnxietyTipsOutputSchema
>;

export async function providePersonalizedAnxietyTips(
  input: ProvidePersonalizedAnxietyTipsInput
): Promise<ProvidePersonalizedAnxietyTipsOutput> {
  return providePersonalizedAnxietyTipsFlow(input);
}

const prompt = ai.definePrompt({
  name: 'providePersonalizedAnxietyTipsPrompt',
  input: {schema: ProvidePersonalizedAnxietyTipsInputSchema},
  output: {schema: ProvidePersonalizedAnxietyTipsOutputSchema},
  prompt: `Eres una IA de bienestar empática. Basado en el resumen pre-analizado de la sesión de ansiedad de un usuario, proporciona algunos consejos breves, accionables y de apoyo. Responde en español.

Resumen del Análisis: El nivel de ansiedad del usuario fue determinado como '{{{anxietyLevel}}}'. {{{sessionDataSummary}}}
Preferencias del Usuario: {{{userPreferences}}}

Genera 2-3 recomendaciones breves y prácticas. Enfócate en ejercicios de respiración, mindfulness o estrategias de afrontamiento a corto plazo. No des consejos médicos ni cuestiones el análisis.`,
});

const providePersonalizedAnxietyTipsFlow = ai.defineFlow(
  {
    name: 'providePersonalizedAnxietyTipsFlow',
    inputSchema: ProvidePersonalizedAnxietyTipsInputSchema,
    outputSchema: ProvidePersonalizedAnxietyTipsOutputSchema,
  },
  async input => {
    const {output} = await prompt(input);
    return output!;
  }
);
