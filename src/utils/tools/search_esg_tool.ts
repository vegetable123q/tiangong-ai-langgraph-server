import { DynamicStructuredTool } from '@langchain/core/tools';
import { z } from 'zod';
import cleanObject from './_shared/clean_object';

class SearchEsgTool extends DynamicStructuredTool {
  private email: string;
  private password: string;

  constructor({ email, password }: { email: string; password: string }) {
    super({
      name: 'Search_ESG_Tool',
      description: 'Perform search on ESG database.',
      schema: z.object({
        query: z.string().min(1).describe('Requirements or questions from the user.'),
        topK: z.number().default(5).describe('Number of top chunk results to return.'),
        extK: z
          .number()
          .default(0)
          .describe('Number of additional chunks to include before and after each topK result.'),
        metaContains: z
          .string()
          .optional()
          .describe(
            'An optional keyword string used for fuzzy searching within document metadata, such as report titles, company names, or other metadata fields. DO NOT USE IT BY DEFAULT.',
          ),
        filter: z
          .object({
            rec_id: z.array(z.string()).optional().describe('Filter by record ID.'),
            country: z.array(z.string()).optional().describe('Filter by country.'),
          })
          .optional()
          .describe(
            'DO NOT USE IT IF NOT EXPLICIT REQUESTED IN THE QUERY. Optional filter conditions for specific fields, as an object with optional arrays of values.',
          ),
        dateFilter: z
          .object({
            publication_date: z
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
        return this.searchEsg(args);
      },
    });

    this.email = email;
    this.password = password;
  }

  private async searchEsg({
    query,
    topK,
    extK,
    metaContains,
    filter,
    dateFilter,
  }: {
    query: string;
    topK: number;
    extK: number;
    metaContains?: string;
    filter?: {
      rec_id?: string[];
      country?: string[];
    };
    dateFilter?: {
      publication_date?: {
        gte?: number;
        lte?: number;
      };
    };
  }): Promise<string> {
    const url = `${process.env.BASE_URL}/esg_search`;
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
            metaContains,
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

export default SearchEsgTool;
