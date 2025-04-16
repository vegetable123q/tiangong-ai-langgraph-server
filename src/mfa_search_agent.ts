import { AIMessage, BaseMessage } from '@langchain/core/messages';
import { ChatPromptTemplate } from '@langchain/core/prompts';
import { Annotation, StateGraph } from '@langchain/langgraph';
import { ToolNode } from '@langchain/langgraph/prebuilt';
import { ChatOpenAI } from '@langchain/openai';
import SearchSciTool from './utils/tools/search_sci_tool';

// 环境变量配置
const email = process.env.EMAIL ?? '';
const password = process.env.PASSWORD ?? '';
const openai_api_key = process.env.OPENAI_API_KEY ?? '';
const openai_chat_model = process.env.OPENAI_CHAT_MODEL ?? '';


const StateAnnotation = Annotation.Root({
  messages: Annotation<BaseMessage[]>({
    reducer: (x, y) => x.concat(y),
  }),
  contents: Annotation<string[]>({
    reducer: (_x, y) => [...y],
  }),
});


const tools = [new SearchSciTool({ email, password })];

const toolNode = new ToolNode(tools);

async function callModel(state: typeof StateAnnotation.State) {
  console.log('---    PROCESS: SEARCH    ---');

  const messages = state.messages;
  const lastMessage = [messages[messages.length - 1] as AIMessage];

  const model = new ChatOpenAI({
    apiKey: openai_api_key,
    modelName: openai_chat_model,
    temperature: 0,
    streaming: true,
  }).bindTools(tools);

  const prompt = ChatPromptTemplate.fromTemplate(
    `You are given a task\n
    The retrieved results are 30 to include more related information. \n
    The extK should be 5 to include more related information. \n
    You should search by the doi in the filter\n
    Task description: {description}\n`,
  );
  const chain = prompt.pipe(model);
  const response = (await chain.invoke({ description: lastMessage })) as AIMessage;

  return {
    messages: [response],
  };
}


function shouldContinue(state: typeof StateAnnotation.State) {
  const messages = state.messages;
  const lastMessage = messages[messages.length - 1] as AIMessage;

  if (lastMessage.tool_calls?.length) {
    console.log('---     DECISION: CALL TOOLS      ---');
    return 'tools';
  }
  console.log('---          DECISION: END        ---');
  return '__end__';
}


async function prepareData(
  state: typeof StateAnnotation.State,
): Promise<Partial<typeof StateAnnotation.State>> {
  console.log('---    PROCESS: PREPARE DATA    ---');

  const messages = state.messages;
  const processedContents: string[] = [];

  messages.forEach((item: BaseMessage) => {
    const data_j = JSON.stringify(item);
    const data = JSON.parse(data_j);
    if (data['type'] === 'constructor') {
      const id = data['id'] as string[];
      if (id[id.length - 1] === 'ToolMessage') {
        // 获取原始内容
        const originalContent = data['kwargs']['content'] as string;
        processedContents.push(originalContent);
      }
    }
  });
    
  return {
    contents: processedContents,
  };
}

// 修改工作流，支持批处理
const workflow = new StateGraph(StateAnnotation)
  .addNode('callModel', callModel)
  .addNode('tools', toolNode)
  .addNode('prepareData', prepareData)
  .addEdge('__start__', 'callModel')
  .addConditionalEdges('callModel', shouldContinue, ['__end__', 'tools'])
  .addEdge('tools', 'prepareData')
  .addEdge('prepareData', '__end__');

export const graph = workflow.compile({});