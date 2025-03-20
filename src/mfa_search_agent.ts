import { AIMessage, BaseMessage } from '@langchain/core/messages';
import { ChatPromptTemplate } from '@langchain/core/prompts';
import { Annotation, StateGraph } from '@langchain/langgraph';
import { ToolNode } from '@langchain/langgraph/prebuilt';
import { ChatOpenAI } from '@langchain/openai';
import { z } from 'zod';
import SearchSciTool from './utils/tools/search_sci_tool';

// 环境变量配置
const email = process.env.EMAIL ?? '';
const password = process.env.PASSWORD ?? '';
const openai_api_key = process.env.OPENAI_API_KEY ?? '';
const openai_chat_model = process.env.OPENAI_CHAT_MODEL ?? '';

// 定义类型接口，避免使用any
interface BoundaryItem {
  source: string;
  SpatialScope: string;
  timeRange: string;
  policyRecommendations: string[];
}


const StateAnnotation = Annotation.Root({
  messages: Annotation<BaseMessage[]>({
    reducer: (x, y) => x.concat(y),
  }),
  contents: Annotation<string[]>({
    reducer: (_x, y) => [...y],
  }),
  contentBatches: Annotation<string[]>({
    default: () => [],
    reducer: (_x, y) => y,
  }),
});


const tools = [new SearchSciTool({ email, password })];

const toolNode = new ToolNode(tools);

// 修改 callModel 函数，提取并存储原始查询
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
    The retrieved results are 150 to include more related information. \n
    Please only use the SearchSciTool to search for the task description.\n
    You can increase the extK parameter to get more information.extK varies from 9 to 10.\n
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


// 修改日志消息，匹配函数名称
async function prepareData(
  state: typeof StateAnnotation.State,
): Promise<Partial<typeof StateAnnotation.State>> {
  console.log('---    PROCESS: PREPARE DATA    ---');

  const messages = state.messages;
  const retrievedContents: string[] = [];

  // 提取工具返回的内容
  messages.forEach((item: BaseMessage) => {
    const data_j = JSON.stringify(item);
    const data = JSON.parse(data_j);
    if (data['type'] === 'constructor') {
      const id = data['id'] as string[];
      if (id[id.length - 1] === 'ToolMessage') {
        retrievedContents.push(data['kwargs']['content'] as string);
      }
    }
  });

  // 处理每个内容项，尝试识别content和source
  const contentSourceGroups: string[] = [];
  
  retrievedContents.forEach((content, index) => {
    try {
      // 尝试将内容解析为JSON
      const parsedContent = JSON.parse(content);
      
      // 处理各种可能的数据结构
      if (Array.isArray(parsedContent)) {
        // 如果是数组，每个元素作为一个组
        parsedContent.forEach((item, itemIndex) => {
          const source = item.source || item.title || `Source ${index}.${itemIndex}`;
          const itemContent = typeof item.content === 'string' ? item.content : JSON.stringify(item);
          contentSourceGroups.push(`SOURCE: ${source}\n\nCONTENT: ${itemContent}`);
        });
      } else if (typeof parsedContent === 'object') {
        // 单个对象，作为一个组
        const source = parsedContent.source || parsedContent.title || `Source ${index}`;
        const itemContent = parsedContent.content || JSON.stringify(parsedContent);
        contentSourceGroups.push(`SOURCE: ${source}\n\nCONTENT: ${itemContent}`);
      } else {
        // 如果不是结构化数据，整体作为一个组
        contentSourceGroups.push(`SOURCE: Source ${index}\n\nCONTENT: ${content}`);
      }
    } catch (e) {
      // 解析失败，整体作为一个组
      contentSourceGroups.push(`SOURCE: Source ${index}\n\nCONTENT: ${content}`);
    }
  });

  console.log(`Processed ${contentSourceGroups.length} content-source groups`);

  return {
    contentBatches: contentSourceGroups,
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