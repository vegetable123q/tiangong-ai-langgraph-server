import { ChatOpenAI } from '@langchain/openai';
import dotenv from 'dotenv';
import fs from 'fs';
import PQueue from 'p-queue';
import path from 'path';
import { z } from 'zod';

dotenv.config();

const openai_api_key = process.env.OPENAI_API_KEY ?? '';
const openai_model = process.env.OPENAI_CHAT_MODEL ?? '';

type questionElement = {
  index: number;
  question: string;
  answer: string;
  difficulty: string;
  type: string;
  field: string;
  model: string;
  response: string;
};

async function evaluateResponse(input: questionElement) {
  const result = z.object({
    accuracy: z.object({
      comments: z
        .string()
        .describe('Chinese explanations and comments on the accuracy score given.'),
      score: z
        .number()
        .describe(
          'This score ranging from 0 to 100 indicates how well the response meets the accuracy criteria.',
        ),
    }),
  });

  const model = new ChatOpenAI({
    apiKey: openai_api_key,
    modelName: openai_model,
    streaming: false,
  });

  const structuredLlm = model.withStructuredOutput(result);

  const criteria = `Overall Evaluation: Assesses alignment with reference answers, focusing on factual correctness, logical reasoning, and clarity/completeness of calculations.
Scoring Standards:
- 90-100 Points:
  - Factual Correctness: Completely accurate, covering all key points.
  - Logical Reasoning: Clear and rigorous, fully aligned with reference answer.
  - Calculation Process: Correct results with organized steps following core logic. Different methods acceptable if final result correct. Consider significant digits if results differ. Additional content fully covers standard answer.
- 70-89 Points:
  - Factual Correctness: Generally accurate with minor omissions or small errors.
  - Logical Reasoning: Overall reasonable with occasional minor flaws.
  - Calculation Process: Mostly correct with slight errors or unclear steps. Different methods acceptable if final result correct. Differences in significant digits considered. Additional content covers standard answer without omissions.
- 50-69 Points:
  - Factual Correctness: Noticeable errors or insufficient understanding of key concepts.
  - Logical Reasoning: Significant gaps, partially correct conclusions.
  - Calculation Process: Substantial errors, disorganized or missing key steps. Different methods may lead to incorrect results. Significant digits not properly considered. Additional content partially covers standard answer.
- 30-49 Points:
  - Factual Correctness: Severe errors or missing critical content.
  - Logical Reasoning: Illogical reasoning, conclusions deviate significantly.
  - Calculation Process: Disorganized, largely incorrect results. Different methods lead to incorrect outcomes. Significant digits not considered. Additional content does not adequately cover standard answer.
- 10-29 Points:
  - Factual Correctness: Majority incorrect or irrelevant.
  - Logical Reasoning: Lacks reasonable reasoning, unsupported conclusions.
  - Calculation Process: Entirely incorrect or missing. Different methods do not produce correct results. Significant digits ignored. Additional content fails to cover standard answer.`;

  const evaluationResults = await structuredLlm.invoke([
    {
      role: 'human',
      content: `Please evaluate the responses to the following question, focusing on logical coherence and whether it provides clear and reasonable results, and provide suggestion (in Chinese) on logic thinking to improve the quality of the response. The evaluation criteria are:
        Criteria: ${criteria}
        Question:${input.question}
        Standard Answer: ${input.answer}
        Response: ${input.response}
          `,
    },
  ]);

  console.log(input.question, evaluationResults.accuracy.score);

  return {
    index: input.index,
    question: input.question,
    answer: input.answer,
    difficulty: input.difficulty,
    type: input.type,
    field: input.field,
    response: input.response,
    model: input.model,
    evaluation: evaluationResults.accuracy,
  };
}

/**
 * 遍历目录并合并 JSON 文件
 * @param {string} dirPath - 目录路径
 * @returns {Promise<questionElement[]>} - 合并后的 JSON 数组
 */
async function mergeJsonFiles(dirPath: string) {
  const files = fs.readdirSync(dirPath);
  const result: questionElement[] = [];
  // console.log(files)
  files.forEach((file) => {
    const filePath = path.join(dirPath, file);
    if (path.extname(filePath) === '.json') {
      const data = fs.readFileSync(filePath, 'utf8');
      const json = JSON.parse(data);
      json.forEach((element: questionElement) => {
        result.push(element);
      });
    }
  });

  return result;
}

async function main() {
  // merge all JSON files in the directory
  const data: questionElement[] = await mergeJsonFiles('./src/data/responses');
  const file: string = 'file_name'; //  outputfile name or responses file name without extension

  // get specefic JSON file
  // const data: questionElement[] = JSON.parse(
  //   fs.readFileSync('./src/data/responses/' + file + '.json', 'utf-8'),
  // );

  const queue = new PQueue({ concurrency: 10 });
  const results: any[] = [];

  const tasks = data.map((item) =>
    queue.add(async () => {
      // console.log(item.model, item.question);
      const result = await evaluateResponse(item);
      results.push(result);
    }),
  );

  console.log(results);

  await Promise.all(tasks);

  fs.writeFileSync(
    './src/data/evaluation/' + file + '_accuracy.json',
    JSON.stringify(results, null, 2),
    'utf-8',
  );
}

main().catch(console.error);
