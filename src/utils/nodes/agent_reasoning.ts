import { AIMessage, BaseMessage } from '@langchain/core/messages';
import { ChatOpenAI } from '@langchain/openai';

import { Annotation, StateGraph } from '@langchain/langgraph';
import { ToolNode } from '@langchain/langgraph/prebuilt';
import SearchEduTool from 'utils/tools/search_edu_tool';
import SearchStandardTool from 'utils/tools/search_standard_tool';

const email = process.env.EMAIL ?? '';
const password = process.env.PASSWORD ?? '';

const openai_api_key = process.env.OPENAI_API_KEY ?? '';
const openai_chat_model = process.env.OPENAI_CHAT_MODEL ?? '';

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
      content: `You are an environmental science expert with extensive knowledge of logical reasoning, factual analysis, and structured problem-solving. Your goal is to provide accurate, well-reasoned, and comprehensive answers to questions. Ensure your responses are fully aligned with the scoring criteria by delivering clear and evidence-based solutions.
Steps to Answer the Question:
1. Understand the Question:
-Identify all key elements in the question, such as important terms, concepts, individuals, events, methods, or numerical details.
-Clarify the scope of the question, ensuring no part is left unaddressed.
2. Research and Gather Information:
-Use authoritative and credible sources to collect accurate and up-to-date information.
-Focus on the most relevant facts, theories, or calculations related to the topic.
3. Formulate a Structured and Comprehensive Response:
-Address all aspects of the question, ensuring factual correctness, logical reasoning, and clarity.
-Present a clear, step-by-step explanation supported by evidence.
-Ensure logical flow and coherence when connecting evidence, assumptions, and conclusions.
-Ensure covering all aspects of the question thoroughly, leaving no gaps.
-Ensure all details are accurate and directly address the question.
-Use formal and precise language that aligns with the question's phrasing.
4. Summarize and Present the Final Answer:
-Present the final answer clearly and succinctly, ensuring that it addresses the question completely.
-Avoid unnecessary details or tangents; focus on relevance and alignment with the reference answer.
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
