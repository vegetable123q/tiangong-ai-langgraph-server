import { DynamicStructuredTool } from '@langchain/core/tools';
import { z } from 'zod';

const filterSchema = z
  .object({
    tag: z.array(z.string()).optional(),
  })
  .optional()
  .describe('Task sources to filter the search results.');

type Filter = z.infer<typeof filterSchema>;

class SearchInternalTool extends DynamicStructuredTool {
  private email: string;
  private password: string;

  constructor({ email, password }: { email: string; password: string }) {
    super({
      name: 'Search_Internal_Tool',
      description: 'Call this tool to search internal reports for company information.',
      schema: z.object({
        query: z.string().min(1).describe('Requirements or questions from the user.'),
        filter: filterSchema,
        topK: z.number().default(5).describe('Number of results to return.'),
      }),
      func: async ({
        query,
        filter,
        topK,
      }: {
        query: string;
        filter: Filter;
        topK: number;
        email: string;
        password: string;
      }) => {
        const requestBody = JSON.stringify({ query, filter, topK });
        // console.log('Request body:', requestBody);

        const url = `${process.env.BASE_URL}/internal_search`;

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

export default SearchInternalTool;
