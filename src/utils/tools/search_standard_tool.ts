import { DynamicStructuredTool } from '@langchain/core/tools';
import { z } from 'zod';
import cleanObject from './_shared/clean_object';

class SearchStandardTool extends DynamicStructuredTool {
  private email: string;
  private password: string;

  constructor({ email, password }: { email: string; password: string }) {
    super({
      name: 'Search_Standard_Tool',
      description: 'Perform search on standard database for precise and specialized information.',
      schema: z.object({
        query: z.string().min(1).describe('Requirements or questions from the user.'),
        topK: z.number().default(1).describe('Number of top chunk results to return.'),
        extK: z
          .number()
          .default(2)
          .describe('Number of additional chunks to include before and after each topK result.'),
        filter: z
          .object({
            title: z.array(z.string()).optional().describe('Filter by the standard title.'),
            issuing_organization: z
              .array(z.string())
              .optional()
              .describe('Filter by the organization that issued this standard.'),
          })
          .optional()
          .describe(
            'DO NOT USE IT IF NOT EXPLICIT REQUESTED IN THE QUERY. Optional filter conditions for specific fields, as an object with optional arrays of values.',
          ),
      }),
      func: async (args) => {
        return this.searchStandard(args);
      },
    });

    this.email = email;
    this.password = password;
  }

  private async searchStandard({
    query,
    topK,
    extK,
    filter,
  }: {
    query: string;
    topK: number;
    extK: number;
    filter?: {
      title?: string[];
      issuing_organization?: string[];
    };
  }): Promise<string> {
    const url = `${process.env.BASE_URL}/standard_search`;
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

export default SearchStandardTool;
