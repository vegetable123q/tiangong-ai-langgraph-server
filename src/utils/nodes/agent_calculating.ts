import { AIMessage, BaseMessage } from '@langchain/core/messages';
import { ChatOpenAI } from '@langchain/openai';

import { Annotation, StateGraph } from '@langchain/langgraph';
import { ToolNode } from '@langchain/langgraph/prebuilt';

import SearchEduTool from 'utils/tools/search_edu_tool';
import SearchStandardTool from 'utils/tools/search_standard_tool';

import { PythonInterpreterTool } from '@langchain/community/experimental/tools/pyinterpreter';
import pyodideModule from 'pyodide';

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
    'Executes Python code in a sandboxed environment and must print and return the execution results. The environment resets after each execution. The tool captures both standard output (stdout) and error output (stderr) and must print out them, ensuring any generated output or errors are available for further analysis.';
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

async function loadCalculationTools() {
  const baseTools = [
    new SearchEduTool({ email, password }),
    new SearchStandardTool({ email, password }),
  ];
  const pythonTool = await createPythonTool();
  return [...baseTools, pythonTool];
}

async function calculationTool() {
  const toolsPromise = await loadCalculationTools();
  const toolNode = new ToolNode(toolsPromise);
  return toolNode;
}

const toolsPromise = loadCalculationTools();

async function callModel(state: typeof StateAnnotation.State) {
  console.log('---- callModel ----');

  const loadedTools = await toolsPromise;

  const model = new ChatOpenAI({
    apiKey: openai_api_key,
    modelName: openai_chat_model,
    streaming: false,
  }).bindTools(loadedTools);

  const response = await model.invoke([
    {
      role: 'human',
      content: `You are an environmental science expert tasked with solving a calculation-based problem. Please follow the steps below to give a comprehensive and precise solution:
1. Understand and Decompose the Problem: Carefully read the problem statement to identify all essential components, parameters, and any missing data that may need clarification or retrieval.
2. Retrieve Accurate and Relevant Information:
- Utilize the provided search tools to find the latest, most credible, and authoritative sources.
- Ensure the data aligns with the problem's context and objectives.
3. Solve Using Logical Steps:
- Develop a clear, step-by-step solution based on the synthesized information, and then write the python code.
- Under no circumstances should values or parameters be inferred or fabricated. Focus solely on the given data and any verifiable external sources.
- Ensure all results are clearly visible with print() functions.
4. For Python-based calculations:
- MUST invoke the PythonInterpreterTool to execute the required computations, avoid manual calculations.
- Executed results should be returned with print() functions.
5. Output Results:
- Provide the final answer(s) with the correct units and significant figures.
6. Interpret Results in the context of the problem:
- Be sure to analyze the Python execution result (exact values) with supportive materials and integrate it into your final answer.
- Use the language same as the input question.
- Explain the logic behind each calculation to demonstrate understanding.
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
  console.log('------ routeModelOutput ------');
  const messages = state.messages;
  const lastMessage: AIMessage = messages[messages.length - 1];
  // console.log(lastMessage);
  if (lastMessage.tool_calls?.length) {
    return 'tools';
  }
  return '__end__';
}

const calculationGraph = new StateGraph(StateAnnotation)
  .addNode('callModel', callModel)
  .addNode('tools', calculationTool)
  .addEdge('__start__', 'callModel')
  .addConditionalEdges('callModel', routeModelOutput, ['tools', '__end__'])
  .addEdge('tools', 'callModel');

// const Calculator = calculationGraph.compile();

export const Calculator = calculationGraph.compile({
  // if you want to update the state before calling the tools
  // interruptBefore: [],
});
