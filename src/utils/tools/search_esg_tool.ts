import { DynamicStructuredTool } from '@langchain/core/tools';
import { z } from 'zod';

class SearchEsgTool extends DynamicStructuredTool {
  private email: string;
  private password: string;

  constructor({ email, password }: { email: string; password: string }) {
    super({
      name: 'Search_ESG_Tool',
      description:
        'Use this tool to perform hybrid search (semantic and key word search) on the ESG database for precise and specialized information.',
      schema: z.object({
        query: z.string().min(1).describe('Requirements or questions from the user.'),
        topK: z.number().default(5).describe('Number of top chunk results to return.'),
        extK: z
          .number()
          .default(0)
          .describe('Number of additional chunks to include before and after each topK result.'),
        // meta_contains: z
        //   .string()
        //   .optional()
        //   .describe('A keyword string used for fuzzy searching within document metadata, such as report titles, company names, or other metadata fields. This is used to narrow down the search scope to documents containing the specified keyword in their metadata.'),
        filter: z
          .record(z.string(), z.array(z.string()))
          .optional()
          .describe(
            'Filter conditions for precise matching of terms. Use a record where field names are keys and their corresponding values are arrays of strings. For example, to filter by specific document IDs (rec_id), use: "filter": {"rec_id": ["value1","value2"]}, or mixed filter: "filter": { "field1": ["value1", "value2"], "field2": ["value3"] }',
          ),
        datefilter: z
          .record(
            z.string(),
            z.object({
              gte: z.number().optional(),
              lte: z.number().optional(),
            }),
          )
          .optional()
          .describe(
            'Filter conditions for date ranges, as a record of field names to objects with optional "gte" and "lte" number values (UNIX timestamps). Example: { "dateField": { "gte": 1609459200, "lte": 1640995200 } }',
          ),
      }),
      func: async ({
        query,
        topK,
        extK,
        // meta_contains,
        filter,
        datefilter,
      }: {
        query: string;
        topK: number;
        extK: number;
        // meta_contains?: string;
        filter?: Record<string, string[]>;
        datefilter?: Record<string, { gte?: number; lte?: number }>;
      }) => {
        const requestBody = {
          query,
          topK,
          extK,
          // ...(meta_contains ? { meta_contains } : {}),
          ...(filter ? { filter } : {}),
          ...(datefilter ? { datefilter } : {}),
        };

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
            body: JSON.stringify(requestBody),
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

export default SearchEsgTool;
