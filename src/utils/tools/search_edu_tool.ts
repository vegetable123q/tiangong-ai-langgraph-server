import { DynamicStructuredTool } from '@langchain/core/tools';
import { z } from 'zod';
import cleanObject from './_shared/clean_object';

class SearchEduTool extends DynamicStructuredTool {
  private email: string;
  private password: string;

  constructor({ email, password }: { email: string; password: string }) {
    super({
      name: 'Search_Edu_Tool',
      description: 'Search the environmental educational materials database for information.',
      schema: z.object({
        query: z.string().min(1).describe('Requirements or questions from the user.'),
        topK: z.number().default(5).describe('Number of top chunk results to return.'),
        extK: z
          .number()
          .default(0)
          .describe('Number of additional chunks to include before and after each topK result.'),
        filter: z
          .object({
            rec_id: z.array(z.string()).optional().describe('Filter by record ID.'),
            course: z.array(z.string()).optional().describe('Filter by course.'),
          })
          .optional()
          .describe(
            'DO NOT USE IT IF NOT EXPLICIT REQUESTED IN THE QUERY. Optional filter conditions for specific fields, as an object with optional arrays of values.',
          ),
      }),
      func: async (args) => {
        return this.searchEdu(args);
      },
    });

    this.email = email;
    this.password = password;
  }

  private async searchEdu({
    query,
    topK,
    extK,
    filter,
  }: {
    query: string;
    topK: number;
    extK: number;
    filter?: {
      rec_id?: string[];
      course?: string[];
    };
  }): Promise<string> {
    const url = `${process.env.BASE_URL}/edu_search`;
    try {
      const response = await fetch(url, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          Authorization: `Bearer ${process.env.SUPABASE_ANON_KEY ?? ''}`,
          email: this.email,
          password: this.password,
          'x-region': process.env.X_REGION ?? '',
        },
        body: JSON.stringify(
          cleanObject({
            query,
            topK,
            extK,
            filter,
          }),
        ),
      });
      if (!response.ok) {
        throw new Error(`HTTP error: ${response.status} ${response.statusText}`);
      }
      const data = await response.json();
      return JSON.stringify(data);
    } catch (error) {
      console.error('Error making the request:', error);
      throw error;
    }
  }
}

export default SearchEduTool;
