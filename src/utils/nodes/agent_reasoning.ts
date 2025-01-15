import { AIMessage, BaseMessage } from '@langchain/core/messages';
import { ChatOpenAI } from '@langchain/openai';

import { TavilySearchResults } from '@langchain/community/tools/tavily_search';
import { Annotation, StateGraph } from '@langchain/langgraph';
import { ToolNode } from '@langchain/langgraph/prebuilt';
import SearchEduTool from 'utils/tools/search_edu_tool';
import SearchEsgTool from 'utils/tools/search_esg_tool';
import SearchSciTool from 'utils/tools/search_sci_tool';
import SearchStandardTool from 'utils/tools/search_standard_tool';

const email = process.env.EMAIL ?? '';
const password = process.env.PASSWORD ?? '';

const openai_api_key = process.env.OPENAI_API_KEY ?? '';
const openai_chat_model = process.env.OPENAI_CHAT_MODEL ?? '';
// const openai_chat_model = 'o1-preview-2024-09-12';

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

const baseTools = [
  new SearchEduTool({ email, password }),
  new TavilySearchResults({ maxResults: 5 }),
  new SearchEsgTool({ email, password }),
  new SearchSciTool({ email, password }),
  new SearchStandardTool({ email, password }),
];
const toolNode = new ToolNode(baseTools);

async function callModel(state: typeof StateAnnotation.State) {
  console.log('---- Reasoning: callModel ----');

  const model = new ChatOpenAI({
    apiKey: openai_api_key,
    modelName: openai_chat_model,
    streaming: false,
  }).bindTools(baseTools);

  const response = await model.invoke([
    {
      role: 'human',
      content: `You are an environmental science expert tasked with answering questions. Please follow these guidelines to answer the problem.
1. Read the Problem Carefully: Understand the question and determine whether it requires logical reasoning or factual knowledge.
2. Retrieve Necessary Information: Use the available search tools to gather the most authoritative information. Ensure all information is relevant, credible, and trustworthy.
3. Answer Appropriately:
- For Reasoning: Provide a step-by-step logical explanation, deducing the answer from the given information.
- For Knowledge: Provide a direct, fact-based answer or explanation relevant to the subject matter.
4. Be Clear and Concise: Ensure the answer is well-structured and precise.
5. Language: use the same language as the question.
${state.suggestion !== '' ? `*** Suggestions ***  ${state.suggestion}` : ''}`,
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
  console.log('------ Reasoning: routeModelOutput ------');
  const messages = state.messages;
  const lastMessage: AIMessage = messages[messages.length - 1];
  // console.log(lastMessage);
  if (lastMessage.tool_calls?.length) {
    return 'tools';
  }
  return '__end__';
}

const reasoningGraph = new StateGraph(StateAnnotation)
  .addNode('callModel', callModel)
  .addNode('tools', toolNode)
  .addEdge('__start__', 'callModel')
  .addConditionalEdges('callModel', routeModelOutput, ['tools', '__end__'])
  .addEdge('tools', 'callModel');

export const Reasoning = reasoningGraph.compile({
  // if you want to update the state before calling the tools
  // interruptBefore: [],
});
