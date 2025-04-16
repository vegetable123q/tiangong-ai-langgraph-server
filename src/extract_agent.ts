import { AIMessage, BaseMessage } from '@langchain/core/messages';
import { ChatPromptTemplate, PromptTemplate } from '@langchain/core/prompts';
import { Annotation, StateGraph } from '@langchain/langgraph';
import { ChatOpenAI } from '@langchain/openai';
import dotenv from 'dotenv';
import { z } from 'zod';

dotenv.config();

// 环境变量配置
const openai_api_key = process.env.OPENAI_API_KEY ?? '';
const openai_chat_model = process.env.OPENAI_CHAT_MODEL ?? '';


// 定义单个小节的结构
const SingleSubsection = z.object({
  subsectionTitle: z.string()
    .describe('generate a title of the subsection'),
  mainContent: z.string()
    .describe('Extract the core arguments in this subsection'),
  policyRecommendations: z.array(z.string())
    .describe('The specific policy recommendations extracted from this subsection, or a statement that no clear recommendations were found'),
});

// 定义最终的PolicyRecommendation结构
const PolicyItem = z.object({
  subSections: z.array(SingleSubsection)
    .describe('A list of subsections with their title, content summary, and policy recommendations'),
});



const BoundaryItem = z.object({
  spatialScope: 
    z.string()
    .describe('Detailed geographic or system boundary of the study'),
  timeRange: 
    z.string()
    .describe('Research time range covered in the study'),
});


// 定义状态类型
const StateAnnotation = Annotation.Root({
  messages: Annotation<BaseMessage[]>({
    reducer: (x, y) => x.concat(y),
  }),
  abstract: Annotation<string>({
    default: () => "",
    reducer: (_, y) => y,
  }),
  searchResults1: Annotation<any[]>({
    default: () => [],
    reducer: (_, y) => y,
  }),
  searchResults2: Annotation<any[]>({
    default: () => [],
    reducer: (_, y) => y,
  }),
  boundaryItem: Annotation<any>({
    default: () => null,
    reducer: (_, y) => y,
  }),
  policyItem: Annotation<any[]>({
    default: () => [],
    reducer: (_, y) => y,
  }),
  output: Annotation<any[]>({
    default: () => [],
    reducer: (_, y) => y,
  }),
});
/**
 * 提取边界信息
 */
async function extractBoundary(
  state: typeof StateAnnotation.State,
): Promise<Partial<typeof StateAnnotation.State>> {
  console.log(`---    EXTRACTING BOUNDARY   ---`);

  const { abstract, searchResults1 } = state;
  
  // 创建LLM实例
  const model = new ChatOpenAI({
    apiKey: openai_api_key,
    modelName: openai_chat_model,
    temperature: 0, // 使用温度0以获得更一致的输出
  });

  // 创建边界和政策提取工具
  const extractTool = {
    name: 'extract_boundary',
    description: 'Extract geographic boundaries and time ranges from the article',
    schema: BoundaryItem,
  };
  
  // 构建完整内容，只使用摘要和搜索结果
  const articleContent = `
    Abstract:
    ${abstract}
    
    First half of the article:
    ${searchResults1}
  `;

  // 提示模板
  const prompt = ChatPromptTemplate.fromTemplate(`
    You are a research assistant analyzing scientific papers on material flow analysis and circular economy.\n
  
    Your task is to EXTRACT the spatial scope and time range from the provided research article information, which consists of an abstract and the first half of the article.\n
  
    Research Article Information:\n
    {article_content}\n\n
  
    Extract the following information:\n
  
    1. Spatial Scope: What specific geographic area or system boundary does the study cover?\n
      - Be as precise as possible (country, region, city, or specific system boundaries)\n
      - Focus on the main area being studied, not areas mentioned in passing\n
      - Look for phrases like "case study in...", "we analyzed...", "this study focuses on..."\n
      - Common locations in abstract or methodology sections\n
      - If multiple areas are studied, list all of them\n
  
    2. Time Range: What specific time period does the study analyze?\n
      - Include exact years if available (e.g., "2010-2015", "from 1990 to 2020")\n
       - Look for phrases like "data from...", "during the period...", "between years..."\n
      - This information is often in methodology sections or when discussing data sources\n
      - If historic and future periods are mentioned, include both\n
  
    IMPORTANT GUIDELINES:\n
    - Only extract information explicitly stated in the text\n
    - Use "Not specified" if the information cannot be found in the provided text\n
    - Be concise but complete in your extraction\n
    - Do not make assumptions about information not explicitly stated\n
    - Focus on the actual study period, not publication dates or references to other studies\n
  `);

  // 绑定工具
  const chain = prompt.pipe(model.bindTools([extractTool], {
    tool_choice: extractTool.name,
  }));

  try {
    // 执行边界提取
    const response = await chain.invoke({
      article_content: articleContent,
    }) as AIMessage;

    // 解析工具调用结果
    if (response.tool_calls && response.tool_calls.length > 0) {
      const boundaryData = response.tool_calls[0].args;
      
      console.log(`Extracted boundary information`);
      
      // 验证数据符合模式
      const validatedItem = BoundaryItem.parse(boundaryData);
      
      return {
        boundaryItem: validatedItem,
        messages: [
          new AIMessage({ content: JSON.stringify(validatedItem, null, 2) }),
        ],
      };
    }
    
    // 没有提取到边界数据
    return {
      boundaryItem: null,
      messages: [
        new AIMessage({ content: 'Failed to extract boundary information and policies' }),
      ],
    };
  } catch (error) {
    console.error('Error extracting boundary information and policies:', error);
    return {
      boundaryItem: null,
      messages: [
        new AIMessage({ content: `Error extracting information: ${error}` }),
      ],
    };
  }
}
async function extractPolicy(
  state: typeof StateAnnotation.State,
): Promise<Partial<typeof StateAnnotation.State>> {
  console.log(`---    EXTRACTING POLICY RECOMMENDATIONS   ---`);

  const { abstract, searchResults2 } = state;
  
  // Create LLM instance
  const model = new ChatOpenAI({
    apiKey: openai_api_key,
    modelName: openai_chat_model,
    temperature: 0, // Use temperature 0 for more consistent outputs
  });

  // Create policy extraction tool
  const extractTool = {
    name: 'extract_policy',
    description: 'Extract policy recommendations from the article',
    schema: PolicyItem,
  };
  

  const prompt = ChatPromptTemplate.fromTemplate(`
    Your task is to read the following text, which represents the last 50% of an academic paper. Carefully analyze the "Discussion" or similar sections (e.g., "Discussion & Conclusion," "Policy Implications," etc.).

    Here is the last 50% of the article content:\n
    {searchResults2}\n\n

    Your goal is to identify and summarize the main content of each subsection within the "Discussion" or similar sections and, importantly, extract any explicitly stated or implied policy recommendations.

    Please output your analysis in the following format:

    **Subsection Title:**
    * Main Content: [Extract the core arguments in this subsection]
    * Policy Recommendations: [Extract the specific policy recommendations from this subsection. If there are no clear policy recommendations, please state so.]


    ...

    Please pay attention to the following points:

    * **Focus on Policy Recommendations:** Policy recommendations are your primary objective. Even if a subsection does not explicitly use words like "recommendation" or "suggestion," determine if its content implies advice for policymakers.
    * **Look for clear recommendation language directed at governments: Focus on sentences that use terms like "recommend,", "suggest","should", "it is necessary to," "we advise," etc. SPECIFICALLY when they suggest actions that governments, policymakers should take.
    * **Use DIRECT QUOTES:** Copy the exact sentences containing policy recommendations rather than paraphrasing them.
    * **Distinguish Main Content from Details:** When summarizing the main content, focus on the core arguments and avoid overly detailed descriptions.
    * **Maintain Objectivity:** Your task is to extract information, not to evaluate or modify the viewpoints of the paper.
    * **If a subsection does not contain clear policy recommendations, explicitly state it.** For example: "This subsection mainly discusses..., and does not explicitly propose any policy recommendations."

  ---
    `)

  // Bind the tool
  const chain = prompt.pipe(model.bindTools([extractTool], {
    tool_choice: extractTool.name,
  }));

  try {
    // Execute policy extraction
    const response = await chain.invoke({
      // 将文章内容传递给链
      searchResults2: searchResults2,
    }) as AIMessage;

    // Parse tool call results
    if (response.tool_calls && response.tool_calls.length > 0) {
      const policyData = response.tool_calls[0].args;
      
      console.log(`Extracted policy recommendations`);
      
      // Validate data matches schema
      const validatedItems = PolicyItem.parse(policyData);
      
      return {
        policyItem: validatedItems.subSections,
        messages: [
          new AIMessage({ content: JSON.stringify(validatedItems, null, 2) }),
        ],
      };
    }
    
    // No policy data extracted
    return {
      policyItem: [],
      messages: [
        new AIMessage({ content: 'Failed to extract policy recommendations' }),
      ],
    };
  } catch (error) {
    console.error('Error extracting policy recommendations:', error);
    return {
      policyItem: [],
      messages: [
        new AIMessage({ content: `Error extracting policy recommendations: ${error}` }),
      ],
    };
  }
}

async function merge(
  state: typeof StateAnnotation.State,
): Promise<Partial<typeof StateAnnotation.State>> {
  console.log(`---    MERGING BOUNDARY AND POLICY INFORMATION   ---`);

  const { boundaryItem, policyItem } = state;
  
  // Create merged result object
  const mergedResult = {
    spatialScope: boundaryItem?.spatialScope || "Not specified",
    timeRange: boundaryItem?.timeRange || "Not specified",
    subSections: policyItem || []
  };
  
  console.log(`Merged boundary information with ${policyItem?.length || 0} policy recommendations`);
  
  return {
    messages: [
      new AIMessage({ content: JSON.stringify(mergedResult, null, 2) }),
    ],
    output: [mergedResult]
  };
}

// 修改工作流图，添加输出节点
const workflow = new StateGraph(StateAnnotation)
  .addNode('extractBoundary', extractBoundary)
  .addNode('extractPolicy', extractPolicy)
  .addNode('merge', merge)
  .addEdge('__start__', 'extractBoundary')
  .addEdge('extractBoundary', 'extractPolicy')
  .addEdge('extractPolicy', 'merge')
  .addEdge('merge','__end__' ); 

// 编译工作流
export const graph = workflow.compile({});

