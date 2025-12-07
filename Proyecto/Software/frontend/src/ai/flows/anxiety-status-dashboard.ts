'use server';

/**
 * @fileOverview Este flujo ahora está OBSOLETO para el cálculo directo de la ansiedad.
 * Su nuevo rol es proporcionar un resumen cualitativo basado en métricas precalculadas por un backend determinista.
 *
 * - getAnxietyStatus - Una función que devuelve una interpretación cualitativa del estado de ansiedad del usuario.
 * - GetAnxietyStatusInput - El tipo de entrada para la función getAnxietyStatus.
 * - GetAnxietyStatusOutput - El tipo de retorno para la función getAnxietyStatus.
 */

import {ai} from '@/ai/genkit';
import {z} from 'genkit';

const GetAnxietyStatusInputSchema = z.object({
  // El backend determinista (ej. una Cloud Function de Python) ahora es responsable de calcular esto.
  // El rol de Gemini es interpretarlos.
  anxietyLevel: z
    .enum(['Bajo', 'Moderado', 'Alto'])
    .describe("El nivel de ansiedad del usuario, pre-calculado por un modelo determinista."),
  heartRate: z.number().describe("La frecuencia cardíaca promedio del usuario en PPM."),
  hrv: z.number().describe("La variabilidad de la frecuencia cardíaca (RMSSD) pre-calculada del usuario."),
  moodLabel: z
    .string()
    .optional()
    .describe('Etiqueta de ánimo opcional del reconocimiento facial de emociones.'),
});

export type GetAnxietyStatusInput = z.infer<typeof GetAnxietyStatusInputSchema>;

const GetAnxietyStatusOutputSchema = z.object({
  anxietyLevel: z
    .enum(['Bajo', 'Moderado', 'Alto'])
    .describe("El nivel de ansiedad actual del usuario."),
  explanation: z
    .string()
    .describe('Una explicación cualitativa y empática del nivel de ansiedad pre-calculado.'),
});

export type GetAnxietyStatusOutput = z.infer<typeof GetAnxietyStatusOutputSchema>;

export async function getAnxietyStatus(input: GetAnxietyStatusInput): Promise<GetAnxietyStatusOutput> {
  return getAnxietyStatusFlow(input);
}

const prompt = ai.definePrompt({
  name: 'getAnxietyStatusPrompt',
  input: {schema: GetAnxietyStatusInputSchema},
  output: {schema: GetAnxietyStatusOutputSchema},
  prompt: `Eres un coach de bienestar empático. Los datos fisiológicos de un usuario han sido analizados por un algoritmo de grado médico, resultando en las siguientes métricas. Tu tarea es proporcionar una explicación breve, de apoyo y cualitativa de su estado. NO recalcules ni cuestiones el nivel de ansiedad proporcionado. Responde en español.

Nivel de Ansiedad Pre-calculado: {{{anxietyLevel}}}
Frecuencia Cardíaca Promedio: {{{heartRate}}} PPM
Variabilidad de la Frecuencia Cardíaca (RMSSD): {{{hrv}}}
Estado de Ánimo Reportado: {{{moodLabel}}}

Basado en esto, proporciona una explicación simple de una o dos frases para el panel del usuario. Ejemplo para 'Alto': "Las señales de tu cuerpo sugieren un alto estado de alerta en este momento. Es un buen momento para pausar y centrarte en tu respiración."

Explicación: `,
});

const getAnxietyStatusFlow = ai.defineFlow(
  {
    name: 'getAnxietyStatusFlow',
    inputSchema: GetAnxietyStatusInputSchema,
    outputSchema: GetAnxietyStatusOutputSchema,
  },
  async input => {
    const {output} = await prompt(input);
    // Pasa el nivel de ansiedad original, pero usa la explicación generada por Gemini.
    return {
      anxietyLevel: input.anxietyLevel,
      explanation: output!.explanation,
    };
  }
);
