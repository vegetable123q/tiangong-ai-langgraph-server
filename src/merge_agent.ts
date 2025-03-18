import { ChatOpenAI } from '@langchain/openai';
import { ChatPromptTemplate } from '@langchain/core/prompts';
import { Annotation, StateGraph } from '@langchain/langgraph';
import { z } from 'zod';
import { AIMessage, BaseMessage } from '@langchain/core/messages';
import dotenv from 'dotenv';

dotenv.config();

// 环境变量配置
const openai_api_key = process.env.OPENAI_API_KEY ?? '';
const openai_chat_model = process.env.OPENAI_CHAT_MODEL ?? '';

// 修改数据结构：移除keyFindings，添加spatialTag（不含not specified选项）
const BoundaryItem = z.object({
  source: z.string().describe('Source of the extracted information'),
  SpatialScope: z.string().describe('Detailed Study Spatial scope'),
  spatialTag: z.string().optional().describe('Spatial scope classification tag (city/province/national/focus)'),
  timeRange: z.string().describe('Research time range'),
  policyRecommendations: z.array(z.string()).describe('Policy recommendations'),
});

type BoundaryItemType = z.infer<typeof BoundaryItem>;

// 定义状态注解
const StateAnnotation = Annotation.Root({
  messages: Annotation<BaseMessage[]>({
    reducer: (x, y) => x.concat(y),
  }),
  boundaryItems: Annotation<BoundaryItemType[]>({
    default: () => [],
    reducer: (_x, y) => y,
  }),
  filteredItems: Annotation<BoundaryItemType[]>({
    default: () => [],
    reducer: (_x, y) => y,
  }),
  mergedResults: Annotation<BoundaryItemType[]>({
    default: () => [],
    reducer: (_x, y) => y,
  }),
  taggedResults: Annotation<BoundaryItemType[]>({
    default: () => [],
    reducer: (_x, y) => y,
  }),
  validTaggedResults: Annotation<BoundaryItemType[]>({
    default: () => [],
    reducer: (_x, y) => y,
  })
});

// 处理输入数据的函数
async function processInput(state: typeof StateAnnotation.State): Promise<Partial<typeof StateAnnotation.State>> {
  console.log('---    PROCESS: PARSING INPUT DATA    ---');
  
  const messages = state.messages;
  const inputData: BoundaryItemType[] = [];
  
  // 从消息中提取边界数据
  for (const message of messages) {
    if (typeof message.content === 'string') {
      // 尝试解析JSON字符串
      const parsedContent = JSON.parse(message.content);
      if (Array.isArray(parsedContent)) {
        for (const item of parsedContent) {
          // 移除keyFindings字段
          const { keyFindings, ...rest } = item;
          const validatedItem = BoundaryItem.parse(rest);
          inputData.push(validatedItem);
        }
      } else if (typeof parsedContent === 'object') {
        // 移除keyFindings字段
        const { keyFindings, ...rest } = parsedContent;
        const validatedItem = BoundaryItem.parse(rest);
        inputData.push(validatedItem);
      }
    } else if (typeof message.content === 'object' && message.content !== null) {
      // 直接处理对象
      const content = message.content as any;
      // 移除keyFindings字段
      const { keyFindings, ...rest } = content;
      const validatedItem = BoundaryItem.parse(rest);
      inputData.push(validatedItem);
    }
    
    // 处理可能的工具调用结果
    if ('tool_calls' in message && Array.isArray(message.tool_calls)) {
      for (const toolCall of message.tool_calls) {
        if (toolCall.name === 'extract_boundary' && toolCall.args) {
          // 移除keyFindings字段
          const args = toolCall.args as any;
          const { keyFindings, ...rest } = args;
          const validatedItem = BoundaryItem.parse(rest);
          inputData.push(validatedItem);
        }
      }
    }
  }

  console.log(`Processed ${inputData.length} boundary items from input`);
  
  return {
    boundaryItems: inputData
  };
}

// 过滤非中国地理范围的研究
async function filterNonChineseLocations(state: typeof StateAnnotation.State): Promise<Partial<typeof StateAnnotation.State>> {
  console.log('---    PROCESS: FILTERING NON-CHINESE LOCATIONS    ---');
  
  const boundaryItems = state.boundaryItems;
  const filteredItems: BoundaryItemType[] = [];
  
  const model = new ChatOpenAI({
    apiKey: openai_api_key,
    modelName: openai_chat_model,
    temperature: 0.1,
  });
  
  // 创建判断工具
  const checkLocationSchema = z.object({
    isInChina: z.boolean().describe('Whether the spatial scope is located in China'),
    confidence: z.number().min(0).max(1).describe('Confidence level of the judgment (0-1)')
  });
  
  const checkLocationTool = {
    name: 'check_location_in_china',
    description: '判断研究的地理范围是否位于中国境内',
    schema: checkLocationSchema,
  };
  
  const prompt = ChatPromptTemplate.fromTemplate(`
    Determine if the following spatial scope is located within China (including mainland China, Hong Kong, Macau, and Taiwan):
    
    Spatial scope: "{spatialScope}"
    
    Respond with:
    1. isInChina: true if the location is in China, false otherwise
    2. confidence: your confidence in this judgment (0-1)
    
    If the location is not explicitly mentioned or is ambiguous, but research context suggests it's about China, consider it as in China.
    If it mentions multiple countries including China, judge based on the main focus.
    If it's a global or international study without specific focus on China, consider it not in China.
    If the spatial scope is too vague or "not specified", mark as not in China with low confidence.
  `);
  
  const chain = prompt.pipe(model.bindTools([checkLocationTool], {
    tool_choice: checkLocationTool.name,
  }));
  
  // 处理每个项目，判断是否在中国
  for (const item of boundaryItems) {
    // 执行地理位置判断
    const result = await chain.invoke({
      spatialScope: item.SpatialScope
    }) as AIMessage;
    
    // 从工具调用中提取结果
    if (result.tool_calls && result.tool_calls.length > 0) {
      const locationResult = result.tool_calls[0].args as any;
      
      // 如果在中国且高置信度，保留该项目
      if (locationResult.isInChina && locationResult.confidence > 0.6) {
        filteredItems.push(item);
      } else {
        console.log(`Filtered out non-Chinese location: "${item.SpatialScope}" (confidence: ${locationResult.confidence})`);
      }
    }
  }
  
  console.log(`Filtered ${boundaryItems.length - filteredItems.length} non-Chinese locations, keeping ${filteredItems.length} items`);
  
  return {
    filteredItems
  };
}

// 合并函数，按source字段合并数据
async function mergeBySource(state: typeof StateAnnotation.State): Promise<Partial<typeof StateAnnotation.State>> {
  console.log('---    PROCESS: MERGING BY SOURCE    ---');
  
  const boundaryItems = state.filteredItems;
  
  // 按source分组
  const groupedBySource: Record<string, BoundaryItemType[]> = {};
  
  for (const item of boundaryItems) {
    if (!groupedBySource[item.source]) {
      groupedBySource[item.source] = [];
    }
    groupedBySource[item.source].push(item);
  }
  
  const mergedResults: BoundaryItemType[] = [];
  
  // 处理每个源的多个项目
  for (const source in groupedBySource) {
    const items = groupedBySource[source];
    
    // 如果只有一个项目，直接添加
    if (items.length === 1) {
      mergedResults.push(items[0]);
      continue;
    }
    
    // 多个项目需要合并
    const model = new ChatOpenAI({
      apiKey: openai_api_key,
      modelName: openai_chat_model,
      temperature: 0.3,
    });
    
    // 创建合并工具的模式 - 不包含keyFindings字段
    const mergeSchema = z.object({
      source: z.string().describe('Source of the extracted information'),
      SpatialScope: z.string().describe('Merged spatial scope information'),
      timeRange: z.string().describe('Merged time range information'),
      policyRecommendations: z.array(z.string()).describe('Merged list of policy recommendations without duplicates'),
    });
    
    const mergeTool = {
      name: 'merge_boundary_items',
      description: '合并来自同一源的多个边界项目',
      schema: mergeSchema,
    };
    
    const prompt = ChatPromptTemplate.fromTemplate(`
      I have multiple boundary items extracted from the same source that need to be merged.
      The items are from source: "{source}"
      
      Items to merge:
      {items}
      
      Please merge these items following these rules:
      1. Keep the same source
      2. Combine spatial scope information, removing any duplicates or very similar points
      3. Combine time range information, taking the most comprehensive range
      4. Merge policy recommendations, removing any duplicates or very similar points
      
      Return a single merged item.
    `);
    
    const chain = prompt.pipe(model.bindTools([mergeTool], {
      tool_choice: mergeTool.name,
    }));
    
    // 执行合并
    const result = await chain.invoke({
      source: source,
      items: JSON.stringify(items, null, 2)
    }) as AIMessage;
    
    // 从工具调用中提取结果
    if (result.tool_calls && result.tool_calls.length > 0) {
      const mergedItem = result.tool_calls[0].args as unknown as BoundaryItemType;
      mergedResults.push(mergedItem);
    }
  }
  
  console.log(`Merged ${boundaryItems.length} items into ${mergedResults.length} unique sources`);
  
  return {
    mergedResults
  };
}

// 添加空间标签函数 - 修改为只使用有效标签
async function addSpatialTags(state: typeof StateAnnotation.State): Promise<Partial<typeof StateAnnotation.State>> {
  console.log('---    PROCESS: ADDING SPATIAL TAGS    ---');
  
  const mergedItems = state.mergedResults;
  const taggedResults: BoundaryItemType[] = [];
  
  const model = new ChatOpenAI({
    apiKey: openai_api_key,
    modelName: openai_chat_model,
    temperature: 0.1,
  });
  
  // 创建标签工具 - 移除"not specified"选项
  const tagSchema = z.object({
    spatialTag: z.enum(['city', 'province', 'national', 'focus'])
      .describe('Classification tag for the spatial scope')
  });
  
  const tagTool = {
    name: 'tag_spatial_scope',
    description: '给地理研究范围分类并添加标签',
    schema: tagSchema,
  };
  
  const prompt = ChatPromptTemplate.fromTemplate(`
    Classify the following spatial scope into one of these categories:
    - "city": City-level spatial scope (e.g., Beijing, Shanghai)
    - "province": Province/state/region level spatial scope (e.g., Guangdong, Hebei)
    - "national": Country level or above (e.g., China, Global)
    - "focus": Below city level (e.g., specific area, district, site, building)
    
    If the spatial scope is unclear, ambiguous, or not specified, do NOT return a tag.
    
    Spatial scope to classify: "{spatialScope}"
    
    Return the appropriate classification tag only if you can determine it with high confidence.
  `);
  
  const chain = prompt.pipe(model.bindTools([tagTool], {
    tool_choice: tagTool.name,
  }));
  
  // 处理每个项目添加标签
  for (const item of mergedItems) {
    try {
      // 执行标签分类
      const result = await chain.invoke({
        spatialScope: item.SpatialScope
      }) as AIMessage;
      
      // 从工具调用中提取结果
      if (result.tool_calls && result.tool_calls.length > 0) {
        const tagResult = result.tool_calls[0].args as any;
        // 创建带标签的新项目
        const taggedItem = {
          ...item,
          spatialTag: tagResult.spatialTag
        };
        taggedResults.push(taggedItem);
      }
      // 如果无法分类，则不添加该项目（不再使用默认的"not specified"标签）
    } catch (error) {
      // 分类失败，跳过此项目
      console.log(`Failed to classify spatial scope for: "${item.SpatialScope}"`);
    }
  }
  
  console.log(`Added spatial tags to ${taggedResults.length} items out of ${mergedItems.length} total items`);
  
  return {
    taggedResults
  };
}

// 格式化输出结果
async function formatOutput(state: typeof StateAnnotation.State): Promise<Partial<typeof StateAnnotation.State>> {
  console.log('---    PROCESS: FORMATTING OUTPUT    ---');
  
  const results = state.taggedResults;
  
  console.log(`Returning ${results.length} tagged results`);
  
  // 按spatialTag分组输出
  const outputMessage = {
    role: "assistant",
    content: JSON.stringify(results, null, 2)
  };
  
  return {
    messages: [outputMessage as unknown as BaseMessage]
  };
}

// 修改工作流，增加地理过滤功能
const workflow = new StateGraph(StateAnnotation)
  .addNode('processInput', processInput)
  .addNode('filterNonChineseLocations', filterNonChineseLocations)
  .addNode('mergeBySource', mergeBySource)
  .addNode('addSpatialTags', addSpatialTags)
  .addNode('formatOutput', formatOutput)
  .addEdge('__start__', 'processInput')
  .addEdge('processInput', 'filterNonChineseLocations')
  .addEdge('filterNonChineseLocations', 'mergeBySource')
  .addEdge('mergeBySource', 'addSpatialTags')
  .addEdge('addSpatialTags', 'formatOutput')
  .addEdge('formatOutput', '__end__');

export const graph = workflow.compile({});