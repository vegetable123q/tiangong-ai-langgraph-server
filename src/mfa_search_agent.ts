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
  originalQuery: Annotation<string>({  // 新增：存储原始查询
    default: () => "",
    reducer: (_x, y) => y,
  }),
  contents: Annotation<string[]>({
    reducer: (_x, y) => [...y],
  }),
  // 批处理相关状态
  contentBatches: Annotation<string[][]>({
    default: () => [],
    reducer: (_x, y) => y,
  }),
  currentBatchIndex: Annotation<number>({
    default: () => -1, // -1 表示尚未开始批处理
    reducer: (_x, y) => y,
  }),
  allBoundaryResults: Annotation<BoundaryItem[]>({
    default: () => [],
    reducer: (x, y) => [...x, ...y],
  }),
  processingComplete: Annotation<boolean>({
    default: () => false,
    reducer: (_x, y) => y,
  }),
});


// 定义批处理大小
const BATCH_SIZE = 5;

const tools = [new SearchSciTool({ email, password })];

const toolNode = new ToolNode(tools);

// 修改 callModel 函数，提取并存储原始查询
async function callModel(state: typeof StateAnnotation.State) {
  console.log('---    PROCESS: SEARCH    ---');

  const messages = state.messages;
  const lastMessage = [messages[messages.length - 1] as AIMessage];
  
  // 提取并存储原始查询（如果尚未存储）
  let originalQuery = state.originalQuery;
  if (!originalQuery && messages.length > 0) {
    // 只获取第一条消息的内容，这通常是用户的原始查询
    originalQuery = typeof messages[0].content === 'string' ? 
      messages[0].content.substring(0, 1000) : // 限制长度为安全值
      "Query not available";
    console.log(`Storing original query: "${originalQuery.substring(0, 100)}..."`);
  }

  const model = new ChatOpenAI({
    apiKey: openai_api_key,
    modelName: openai_chat_model,
    temperature: 0,
    streaming: true,
  }).bindTools(tools);

  const prompt = ChatPromptTemplate.fromTemplate(
    `You are given a task\n
    The retrieved results are 80 to include more related information. \n
    Please only use the SearchSciTool to search for the task description.\n
    You can increase the extK parameter to get more information.extK varies from 5 to 8.\n
    Task description: {description}\n`,
  );
  const chain = prompt.pipe(model);
  const response = (await chain.invoke({ description: lastMessage })) as AIMessage;

  return {
    messages: [response],
    originalQuery: originalQuery,
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

// 从工具返回结果中提取内容并按content和source分组后分批
async function prepareBatches(
  state: typeof StateAnnotation.State,
): Promise<Partial<typeof StateAnnotation.State>> {
  console.log('---    PROCESS: PREPARE BATCHES    ---');

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

  // 将内容分批，每批BATCH_SIZE个
  const batches: string[][] = [];
  for (let i = 0; i < contentSourceGroups.length; i += BATCH_SIZE) {
    const batch = contentSourceGroups.slice(i, i + BATCH_SIZE);
    batches.push(batch);
  }

  console.log(`Prepared ${batches.length} batches from ${contentSourceGroups.length} content-source groups`);

  return {
    contentBatches: batches,
    currentBatchIndex: batches.length > 0 ? 0 : -1, // 如果有批次，设置为第一批
  };
}


// 批处理决策函数
function decideBatchProcessing(state: typeof StateAnnotation.State) {
  const { currentBatchIndex, contentBatches, processingComplete } = state;

  // 如果处理已完成，进入合并阶段
  if (processingComplete) {
    console.log('---    DECISION: ALL BATCHES PROCESSED, MERGING    ---');
    return 'merge';
  }

  // 如果没有批次或已处理完所有批次，完成处理
  if (contentBatches.length === 0 || currentBatchIndex >= contentBatches.length) {
    console.log('---    DECISION: NO MORE BATCHES, COMPLETING    ---');
    return 'complete';
  }

  // 否则处理当前批次
  console.log(`---    DECISION: PROCESSING BATCH ${currentBatchIndex + 1}/${contentBatches.length}    ---`);
  return 'process_batch';
}


// 处理当前批次
async function processBatch(
  state: typeof StateAnnotation.State,
): Promise<Partial<typeof StateAnnotation.State>> {
  const { currentBatchIndex, contentBatches, originalQuery, allBoundaryResults } = state;
  const query = originalQuery || "Query not available";

  console.log(`---    PROCESS: BATCH ${currentBatchIndex + 1}/${contentBatches.length}    ---`);

  // 获取当前批次内容
  const currentBatch = contentBatches[currentBatchIndex];
  
  
  const boundaryResults: BoundaryItem[] = [];
  const boundarySchema = z.object({
        source: z.string().describe('Source of the extracted information'),
        SpatialScope: z.string().describe('Detailed Study Spatial scope'),
        timeRange: z.string().describe('Research time range'),
        policyRecommendations: z.array(z.string()).describe('Policy recommendations'),
      });

  const boundaryPrompt = ChatPromptTemplate.fromTemplate(
        `Please extract the study boundary information based on the user query and Retrieved content given.\n
        Please make sure to extract from every source provided. Even the information may not be so complete.\n
        For each source, extract key research spatial scopes, time ranges and policy recommendations.\n
        The source should just copy the source information, including the title, author, doi, and other information.\n
        
        User query: {query}\n
        
        Retrieved content:\n
        {retrievedContents}\n
        
        
        For policy recommendations, extract relevant information from the content.\n
        And for policy recommendations it should include at least one sentence above and below the key information`,
      );

  const boundaryModel = new ChatOpenAI({
        apiKey: openai_api_key,
        modelName: openai_chat_model,
        temperature: 0, 
      }).bindTools([
        {
          name: 'extract_boundary',
          description: '提取研究边界信息',
          schema: boundarySchema,
        },
      ]);

  try {
    const boundaryResponse = await boundaryPrompt.pipe(boundaryModel).invoke({
          query: query,
          retrievedContents: currentBatch.join('\n\n---NEXT DOCUMENT---\n\n'),
        }) as AIMessage;

        // 提取边界结果
    if (boundaryResponse.tool_calls) {
          boundaryResponse.tool_calls.forEach(call => {
            if (call.name === 'extract_boundary') {
              boundaryResults.push(call.args as unknown as BoundaryItem);
            }
          });
        }

        console.log(`Extracted ${boundaryResults.length} boundary items from batch ${currentBatchIndex + 1}`);
      } catch (error) {
        console.error(`Error in boundary extraction: ${error}`);
        // 发生错误也继续执行下一批次
    
    }

    // 更新到下一批次
    const nextBatchIndex = currentBatchIndex + 1;
    const isComplete = nextBatchIndex >= contentBatches.length;

  return {
      currentBatchIndex: nextBatchIndex,
      allBoundaryResults: boundaryResults,
      processingComplete: isComplete,
    };

  }


// 完成处理，设置完成标志
async function completeProcessing(
  _state: typeof StateAnnotation.State, // 添加下划线前缀表示有意不使用该参数
): Promise<Partial<typeof StateAnnotation.State>> {
  console.log('---    PROCESS: COMPLETING BATCH PROCESSING    ---');
  return {
    processingComplete: true,
  };
}

// 合并所有批次的结果
async function mergeResults(
  state: typeof StateAnnotation.State,
): Promise<Partial<typeof StateAnnotation.State>> {
  console.log('---    PROCESS: MERGING ALL RESULTS    ---');
  
  const { allBoundaryResults } = state;
  console.log(`Merging ${allBoundaryResults.length} total boundary items`);

  // 如果没有结果，返回空消息
  if (allBoundaryResults.length === 0) {
    return {
      messages: [
        {
          role: 'assistant',
          content: 'No relevant boundary information was found from the search results.',
        } as unknown as BaseMessage,
      ],
    };
  }

  // 创建包含所有结果的消息，使用JSON.stringify保持原始数据格式
  return {
    messages: [
      {
        role: 'assistant',
        content: JSON.stringify(allBoundaryResults, null, 2),
      } as unknown as BaseMessage,
    ],
  };
}
// 修改工作流，支持批处理
const workflow = new StateGraph(StateAnnotation)
  .addNode('callModel', callModel)
  .addNode('tools', toolNode)
  .addNode('prepareBatches', prepareBatches)
  .addNode('processBatch', processBatch)
  .addNode('completeProcessing', completeProcessing)
  .addNode('mergeResults', mergeResults)
  .addEdge('__start__', 'callModel')
  .addConditionalEdges('callModel', shouldContinue, ['__end__', 'tools'])
  .addEdge('tools', 'prepareBatches')
  .addConditionalEdges('prepareBatches', decideBatchProcessing, {
    'process_batch': 'processBatch',
    'complete': 'completeProcessing',
    'merge': 'mergeResults',
  })
  .addConditionalEdges('processBatch', decideBatchProcessing, {
    'process_batch': 'processBatch',
    'complete': 'completeProcessing',
    'merge': 'mergeResults',
  })
  .addEdge('completeProcessing', 'mergeResults')
  .addEdge('mergeResults', '__end__');

export const graph = workflow.compile({});