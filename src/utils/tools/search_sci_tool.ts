import { DynamicStructuredTool } from '@langchain/core/tools';
import { z } from 'zod';
import cleanObject from './_shared/clean_object';

class SearchSciTool extends DynamicStructuredTool {
  private email: string;
  private password: string;

  constructor({ email, password }: { email: string; password: string }) {
    super({
      name: 'Search_Sci_Tool',
      description: 'Perform search on academic database for precise and specialized information.',
      schema: z.object({
        query: z.string().min(1).describe('Requirements or questions from the user.'),
        topK: z.number().default(5).describe('Number of top chunk results to return.'),
        extK: z
          .number()
          .default(0)
          .describe('Number of additional chunks to include before and after each topK result.'),
        filter: z
          .object({
            journal: z.array(z.string()).optional().describe('Filter by journal.'),
          })
          .optional()
          .describe(
            'DO NOT USE IT IF NOT EXPLICIT REQUESTED IN THE QUERY. Optional filter conditions for specific fields, as an object with optional arrays of values.',
          ),
        dateFilter: z
          .object({
            date: z
              .object({
                gte: z.number().optional(),
                lte: z.number().optional(),
              })
              .optional(),
          })
          .optional()
          .describe(
            'DO NOT USE IT IF NOT EXPLICIT REQUESTED IN THE QUERY. Optional filter conditions for date ranges in UNIX timestamps.',
          ),
      }),
      func: async (args) => {
        return this.searchSci(args);
      },
    });

    this.email = email;
    this.password = password;
  }

  private async searchSci({
    query,
    topK,
    extK,
    filter,
    dateFilter,
  }: {
    query: string;
    topK: number;
    extK: number;
    filter?: {
      journal?: string[];
    };
    dateFilter?: {
      date?: {
        gte?: number;
        lte?: number;
      };
    };
  }): Promise<string> {
    const url = `${process.env.BASE_URL}/sci_search`;
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
            dateFilter,
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

export default SearchSciTool;
