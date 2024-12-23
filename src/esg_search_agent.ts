import { TavilySearchResults } from '@langchain/community/tools/tavily_search';
import type { AIMessage } from '@langchain/core/messages';
import { AIMessageChunk } from '@langchain/core/messages';
import { ChatOpenAI } from '@langchain/openai';

import { MessagesAnnotation, StateGraph, Annotation } from '@langchain/langgraph';
import { ToolNode } from '@langchain/langgraph/prebuilt';
import { z } from 'zod';
import SearchInternetTool from 'utils/tools/search_internet_tool';


const InternalStateAnnotation = MessagesAnnotation;
const OutputStateAnnotation = Annotation.Root({
  last_content: Annotation<string[]>(),
});

const tools = [new TavilySearchResults({ maxResults: 5 })];

// const email = process.env.EMAIL ?? '';
// const password = process.env.PASSWORD ?? '';
// const tools = [
//   new SearchInternetTool({ email, password }),
// ];

// Define the function that calls the model
async function callModel(state: typeof InternalStateAnnotation.State) {
  const model = new ChatOpenAI({
    model: 'gpt-4o-mini',
  }).bindTools(tools);

  // console.log(state.messages);

  const response = await model.invoke([
    {
      role: 'system',
      content: `You are an expert assistant specialized in extracting official ESG (Environmental, Social, and Governance) report URLs from internet search results. Begin by searching for PDF versions from authoritative sources such as company websites, investor relations pages, or trusted repositories. Use the company's country and language to perform translated searches for PDFs from the outset. If PDFs are unavailable in any language, seek direct HTML reports. Exclude irrelevant links, summaries, or partial content. Provide only valid, direct links to full ESG reports. If no valid URLs are found, adjust search conditions and employ various methods to refine and regenerate search queries to improve accuracy and retrieve the desired ESG report links.`,
    },
    ...state.messages,
  ]);

  // MessagesAnnotation supports returning a single message or array of messages
  return { messages: response };
}

// Define the function that determines whether to continue or not
function routeModelOutput(state: typeof InternalStateAnnotation.State) {
  const messages = state.messages;
  const lastMessage: AIMessage = messages[messages.length - 1];
  // If the LLM is invoking tools, route there.
  if ((lastMessage?.tool_calls?.length ?? 0) > 0) {
    return 'tools';
  }
  // Otherwise to the outputModel.
  return 'outputModel';
}

async function outputModel(state: typeof InternalStateAnnotation.State) {
  const model = new ChatOpenAI({
    model: 'gpt-4o-mini',
  });

  const ResponseFormatter = z.object({
    urls: z.array(z.string()).describe('Urls of the ESG report'),
  });

  const modelWithStructuredOutput = model.withStructuredOutput(ResponseFormatter);

  const lastRelevantMessage = state.messages.slice(-1);
  // console.log(lastRelevantMessage);

  const response = await modelWithStructuredOutput.invoke([
    {
      role: 'system',
      content: `Convert the extracted URLs into a structured format.`,
    },
    ...lastRelevantMessage,
  ]);

  return { last_content: response.urls };

  // const messages = new AIMessageChunk(JSON.stringify(response));
  // // MessagesAnnotation supports returning a single message or array of messages
  // return { messages: messages };
}


const workflow = new StateGraph({
  input: InternalStateAnnotation,
  output: OutputStateAnnotation,
  stateSchema: InternalStateAnnotation,
})
  .addNode('callModel', callModel)
  .addNode('tools', new ToolNode(tools))
  .addNode('outputModel', outputModel)
  .addEdge('__start__', 'callModel')
  .addConditionalEdges(
    'callModel',
    routeModelOutput,
    ['tools', 'outputModel'],
  )
  .addEdge('tools', 'callModel')
  .addEdge('outputModel', '__end__');

export const graph = workflow.compile({});
