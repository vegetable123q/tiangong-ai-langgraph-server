import { TavilySearchResults } from '@langchain/community/tools/tavily_search';
import { AIMessage, BaseMessage } from '@langchain/core/messages';
import { ChatOpenAI } from '@langchain/openai';

import { Annotation, StateGraph } from '@langchain/langgraph';
import { ToolNode } from '@langchain/langgraph/prebuilt';

import SearchEduTool from 'utils/tools/search_edu_tool';
import SearchEsgTool from 'utils/tools/search_esg_tool';
import SearchSciTool from 'utils/tools/search_sci_tool';
import { z } from 'zod';
import SearchStandardTool from './utils/tools/search_standard_tool';

const email = process.env.EMAIL ?? '';
const password = process.env.PASSWORD ?? '';

const openai_api_key = process.env.OPENAI_API_KEY ?? '';
// const openai_chat_model = process.env.OPENAI_CHAT_MODEL ?? '';
// const openai_chat_model = 'o1-preview-2024-09-12';

const tools = [
  new SearchEduTool({ email, password }),
  // new TavilySearchResults({ maxResults: 5 }),
  new SearchEsgTool({ email, password }),
  new SearchSciTool({ email, password }),
  new SearchStandardTool({ email, password }),
];

const StateAnnotation = Annotation.Root({
  messages: Annotation<BaseMessage[]>({
    reducer: (x, y) => x.concat(y),
  }),
  answers: Annotation<string[]>({
    reducer: (x, y) => (y ? x.concat(y) : x),
  }),
  suggestion: Annotation<string>(),
  score: Annotation<number>(),
});

async function callModel(state: typeof StateAnnotation.State) {
  console.log('---- callModel ----');
  const model = new ChatOpenAI({
    apiKey: openai_api_key,
    modelName: 'gpt-4o-2024-11-20',
    streamUsage: false,
    streaming: false,
  }).bindTools(tools);

  const response = await model.invoke([
    {
      role: 'human',
      content: `You are an expert in the field of environmental science. Your task is to solve the given problem and provide a detailed, well-structured answer.
      Please adhere to the following guidelines:
      - Analyze and Decompose the Problem: Carefully read and comprehend the problem statement. Break down the problem into its core components, identifying the key aspects that require information retrieval.
      - Retrieve Supportive Information: Utilize available search tools to gather the latest and most authoritative information relevant to the problem. Ensure the relevance and reliability of the information, disregarding any data that is irrelevant or comes from dubious sources.
      - Evaluate and Integrate Information: Assess the validity and credibility of the retrieved information. Integrate the pertinent information to form a comprehensive understanding of the problem.
      - Construct the Solution: Based on the integrated information, develop a logically coherent and evidence-based solution. Ensure that your answer aligns with the provided evidence and addresses all key aspects of the problem. 
      - Review and Refine: After drafting your response, review it for logical consistency, clarity, and completeness. Make necessary revisions to ensure the answer is comprehensive and easy to understand.
        ${state.suggestion !== '' ? '- Suggestions: ' + state.suggestion : ''}
        `,
    },
    ...state.messages,
  ]);

  return {
    messages: response,
    answers: [response.lc_kwargs.content],
  };
}

// Define the function that determines whether to continue or not
function routeModelOutput(state: typeof StateAnnotation.State) {
  console.log('------ routeModelOutput ------');
  const messages = state.messages;
  const lastMessage: AIMessage = messages[messages.length - 1];
  // console.log(lastMessage);
  if (lastMessage.tool_calls?.length) {
    return 'tools';
  }
  return '__end__';
}

const subgraphBuilder = new StateGraph(StateAnnotation)
  .addNode('callModel', callModel)
  .addNode('tools', new ToolNode(tools))
  .addEdge('__start__', 'callModel')
  .addConditionalEdges('callModel', routeModelOutput, ['tools', '__end__'])
  .addEdge('tools', 'callModel');

const subgraph = subgraphBuilder.compile();

async function getAnswer(state: typeof StateAnnotation.State) {
  console.log('---- getAnswer ----');
  const { messages, answers } = await subgraph.invoke({
    messages: [state.messages[0]],
    suggestion: state.suggestion ? state.suggestion : '',
  });

  return {
    messages: messages,
    answers: answers,
  };
}

async function evaluateAnswer(state: typeof StateAnnotation.State) {
  console.log('--- evaluateOutput ---');
  const firstMessage: AIMessage = state.messages[0];
  const answer: string = state.answers[state.answers.length - 1];
  console.log('---- answer ----');
  console.log(answer);

  const scoreResult = z.object({
    score: z
      .number()
      .describe(
        'This score ranging from 0 to 100 indicates how well the response meets the evaluate criteria, with 100 being the perfect answer while 0 being the worst.',
      ),
    suggestion: z
      .string()
      .optional()
      .describe('Suggestion for improving the approachs to problem solving'),
  });

  const evaluationCriteria = `
    1. Accuracy (totally 50 points):
      - Full Marks (50): The answer directly addresses all parts of the question, providing all relevant details and demonstrating a thorough understanding of the topic.
      - Partial Marks (30-49): The answer addresses most parts of the question but may have minor inaccuracies or omissions in details.
      - Minimal Marks (10-29): The answer addresses the question but with significant gaps in understanding or important inaccuracies.
      - No Marks (0-9): The answer does not address the question, or the response is fundamentally incorrect.
    2. Clarity (totally 20 points):
      - Full Marks (20): The answer is well-organized, logically structured, and easy to follow. Ideas are communicated clearly, and there is no ambiguity.
      - Partial Marks (10-19): The answer is generally clear, but some parts may be awkwardly worded, leading to minor confusion.
      - Minimal Marks (1-9): The response is difficult to follow due to poor organization or unclear explanations.
      - No Marks (0): The response is so unclear that it is almost impossible to understand the main point.
    3. Depth (totally 20 points):
      - Full Marks (20): The answer demonstrates critical thinking, offering a well-rounded perspective with insightful analysis and examples.
      - Partial Marks (10-19): The response provides some depth, but lacks comprehensive analysis or leaves some parts underexplored.
      - Minimal Marks (1-9): The response is shallow, merely repeating basic facts without any meaningful analysis or insights.
      - No Marks (0): The answer lacks depth and does not provide any substantial reasoning or exploration of the topic.
    4. Grammar and Style (totally 10 points):
      - Full Marks (10): The response is grammatically correct, free of spelling errors, and uses professional, formal language. The tone is appropriate for the context.
      - Partial Marks (5-9): The response has a few grammatical errors or awkward phrasing, but it doesn’t detract significantly from the overall understanding.
      - Minimal Marks (1-4): Frequent grammatical errors or poor style significantly hinder understanding, though the core meaning is still apparent.
      - No Marks (0): The response contains major grammatical or stylistic issues that make it difficult to understand or unprofessional.
  `;

  const model = new ChatOpenAI({
    apiKey: openai_api_key,
    modelName: 'o1-2024-12-17',
    streamUsage: false,
    streaming: false,
  });

  const structuredLlm = model.withStructuredOutput(scoreResult);

  const response = await structuredLlm.invoke([
    {
      role: 'human',
      content: `Evaluate the following response based on the criteria provided and provide suggestion on logic thinking to improve the quality of the response:\n\n
      Criteria: ${evaluationCriteria}.
      Question: ${firstMessage.content}.
      Response: ${answer}.`,
    },
  ]);

  return {
    answers: state.answers,
    messages: response,
    suggestion: response?.suggestion ?? '',
    score: response.score,
  };
}

function routeModelThink(state: typeof StateAnnotation.State) {
  console.log('--- routeModelThink ---');
  if (state.score > 90) {
    console.log('---- finalResult ----');
    console.log(state.answers[state.answers.length - 1]);
    return '__end__';
  }
  console.log('---- getAnswer ----');
  return 'getAnswer';
}

const workflow = new StateGraph(StateAnnotation)
  .addNode('getAnswer', getAnswer)
  .addNode('evaluateAnswer', evaluateAnswer)
  // .addNode('outputModel', outputModel)
  .addEdge('__start__', 'getAnswer')
  .addEdge('getAnswer', 'evaluateAnswer')
  .addConditionalEdges('evaluateAnswer', routeModelThink, ['getAnswer', '__end__']);
// .addEdge('outputModel', '__end__');

export const graph = workflow.compile({
  // if you want to update the state before calling the tools
  // interruptBefore: [],
});
