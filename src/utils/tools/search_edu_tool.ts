import { DynamicStructuredTool } from '@langchain/core/tools';
import { z } from 'zod';

type FilterType = { course: string[] } | Record<string | number | symbol, never>;

class SearchEduTool extends DynamicStructuredTool {
  private email: string;
  private password: string;

  constructor({ email, password }: { email: string; password: string }) {
    super({
      name: 'Search_Edu_Tool',
      description:
        'Use this tool to search the environmental educational materials database for information.',
      schema: z.object({
        query: z.string().min(1).describe('Requirements or questions from the user.'),
        course: z.array(z.string()).default([]).describe('Course name to filter the search.'),
        topK: z.number().default(5).describe('Number of top chunk results to return.'),
      }),
      func: async ({ query, course, topK }: { query: string; course: string[]; topK: number }) => {
        const filter: FilterType = course.length > 0 ? { course: course } : {};
        const isFilterEmpty = Object.keys(filter).length === 0;
        const requestBody = JSON.stringify(
          isFilterEmpty ? { query, topK } : { query, topK, filter },
        );

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
            body: requestBody,
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
      },
    });
    this.email = email;
    this.password = password;
  }
}

export default SearchEduTool;
