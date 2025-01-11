import { TavilySearchResults } from '@langchain/community/tools/tavily_search';
import { AIMessage, BaseMessage } from '@langchain/core/messages';
import { ChatOpenAI } from '@langchain/openai';

import { Annotation, StateGraph } from '@langchain/langgraph';
import { ToolNode } from '@langchain/langgraph/prebuilt';

import SearchEduTool from 'utils/tools/search_edu_tool';
import SearchEsgTool from 'utils/tools/search_esg_tool';
import SearchSciTool from 'utils/tools/search_sci_tool';
import SearchStandardTool from './utils/tools/search_standard_tool';

import { PythonInterpreterTool } from '@langchain/community/experimental/tools/pyinterpreter';
import pyodideModule from 'pyodide';

import { z } from 'zod';

const email = process.env.EMAIL ?? '';
const password = process.env.PASSWORD ?? '';

const openai_api_key = process.env.OPENAI_API_KEY ?? '';
// const openai_chat_model = process.env.OPENAI_CHAT_MODEL ?? '';
// const openai_chat_model = 'o1-preview-2024-09-12';

async function createPythonTool() {
  const pyodide = await pyodideModule.loadPyodide();
  if (!pyodide) {
    console.error('Failed to load Pyodide');
  } else {
    console.log('Pyodide loaded successfully');
  }
  await pyodide.loadPackage(['numpy', 'pandas', 'scipy', 'sympy']);
  const pythonTool = new PythonInterpreterTool({ instance: pyodide });
  pythonTool.description =
    'Executes Python code in a sandboxed environment and print the execution results. The environment resets after each execution. The tool captures both standard output (stdout) and error output (stderr) and returns them, ensuring any generated output or errors are available for further analysis.';
  pyodide.globals.set('console', {
    log: (msg: string) => {
      console.log('Python Output:', msg);
    },
    error: (msg: string) => {
      console.error('Python Error:', msg);
    },
  });
  return pythonTool;
}

async function loadAllTools() {
  const baseTools = [
    new SearchEduTool({ email, password }),
    new TavilySearchResults({ maxResults: 5 }),
    new SearchEsgTool({ email, password }),
    new SearchSciTool({ email, password }),
    new SearchStandardTool({ email, password }),
  ];
  const pythonTool = await createPythonTool();
  return [...baseTools, pythonTool];
}

async function buildToolNode() {
  const toolsPromise = await loadAllTools();
  const toolNode = new ToolNode(toolsPromise);
  return toolNode;
}

const toolsPromise = loadAllTools();
// const tools = [getWeather];

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

  const loadedTools = await toolsPromise;
  const model = new ChatOpenAI({
    apiKey: openai_api_key,
    modelName: 'o1-2024-12-17',
    streamUsage: false,
    streaming: false,
  }).bindTools(loadedTools);

  const response = await model.invoke([
    {
      role: 'human',
      content: `You are an expert in environmental science, tasked with solving the given problem. Please Analyze the problem and provide a detailed solution based on the information provided. Ensure your response is accurate, logical, and well-structured. Here are some guidelines:

      1. **Analyze and Decompose the Problem**: Read the problem statement carefully. Identify the key components that need further investigation or data retrieval.
      2. **Retrieve Relevant Information**: Use the available search tools to gather the latest and most authoritative information. Ensure all information is relevant, credible, and trustworthy.
      3. **Evaluate and Synthesize the Information**: Assess the validity of the retrieved information, and integrate the most pertinent data to form a well-rounded understanding of the problem.
      4. **Formulate the Solution**: Based on the synthesized information, develop a clear, logical, and evidence-based solution. Ensure your answer directly addresses the problem's core aspects.
      5. **Review and Refine**: After drafting the solution, review it for accuracy, coherence, and clarity. Make necessary revisions to improve the logical flow and ensure completeness.
      6. **Especially for calculation questions**: Invoke the appropriate tools (e.g., Python) to run the necessary code and ensure any Python code executed print the outputs, if the problem requires computation or specific code execution. Be sure to analyze the Python execution result (e.g., exact values) with supportive materials and integrate it into your final answer. Do not make any assumptions and do not infer parameter values.  
      ***Important:*** DO NOT TRUMP UP! 

      ${state.suggestion !== '' ? `- Suggestions: ${state.suggestion}` : ''}`,
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
  .addNode('tools', buildToolNode)
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
  // console.log(state.messages);
  const firstMessage: AIMessage = state.messages[0];
  const answer: string = state.answers[state.answers.length - 1];
  console.log('---- answer ----');
  console.log(answer);

  const scoreResult = z.object({
    score: z
      .number()
      .describe(
        'This score ranging from 0 to 30 indicates how well the response meets the evaluate criteria, with 30 being the perfect answer while 0 being the worst.',
      ),
    suggestion: z
      .string()
      .optional()
      .describe('Suggestion for improving the approachs to problem solving'),
  });

  const evaluationCriteria = `
        *** Logical Coherence (totally 20 points) *** 
        - Structure: Is the analysis logically structured, with a clear and complete flow?
          10–9 points: Highly logical and well-structured
          8–7 points: Mostly clear, with minor flaws
          6–5 points: Noticeable gaps or missing steps
          4–3 points: Poor structure or incomplete reasoning
          2–1 points: Chaotic or entirely illogical
        - Use of Conditions: Are all provided conditions effectively used?
          10–9 points: Fully and correctly utilized
          8–7 points: Mostly utilized, with minor omissions
          6–5 points: Some conditions not effectively used
          2–1 points: Misused or ignored

        *** Clarity of Results (10 points) *** 
        - If this is a computation qustion: Are the numerical results accurate, clear, and logically consistent?
          10–9 points: Clear, accurate, and rigorous
          8–7 points: Mostly correct, with minor inaccuracies
          6–5 points: Significant errors or unclear presentation
          2–1 points: Incorrect or irrelevant results
        - Else, for other questions (responses require qualitative Insights): Are the explanations and reasoning clear and insightful?
          10–9 points: Clear, insightful, and well-supported
          8–7 points: Generally sound but somewhat superficial
          6–5 points: Lacks depth or has inconsistencies
          2–1 points: Weak or missing analysis
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
      content: `Please evaluate the responses to the following question, focusing on logical coherence and whether it provides clear and reasonable results, and provide suggestion on logic thinking to improve the quality of the response. The evaluation criteria are:\n\n
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
  if (state.score > 25) {
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
  .addEdge('__start__', 'getAnswer')
  .addEdge('getAnswer', 'evaluateAnswer')
  .addConditionalEdges('evaluateAnswer', routeModelThink, ['getAnswer', '__end__']);

export const graph = workflow.compile({
  // if you want to update the state before calling the tools
  // interruptBefore: [],
});
