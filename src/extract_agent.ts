import { AIMessage, BaseMessage, HumanMessage } from '@langchain/core/messages';
import { ChatPromptTemplate } from '@langchain/core/prompts';
import { Annotation, StateGraph } from '@langchain/langgraph';
import { ChatOpenAI } from '@langchain/openai';
import dotenv from 'dotenv';
import { z } from 'zod';

dotenv.config();

// 环境变量配置
const openai_api_key = process.env.OPENAI_API_KEY ?? '';
const openai_chat_model = process.env.OPENAI_CHAT_MODEL ?? '';

// 边界项数据模型
const BoundaryItem = z.object({
  source: z.string().describe('Source of the extracted information'),
  SpatialScope: z.string().describe('Detailed Study Spatial scope'),
  timeRange: z.string().describe('Research time range'),
  policyRecommendations: z.array(z.string()).describe('Policy recommendations'),
});

// 相关性结果模型
const RelevanceResult = z.object({
  isRelevant: z.boolean().describe('Whether the content is relevant to the query'),
  confidence: z.number().min(0).max(1).describe('Confidence level for relevance judgment (0-1)'),
  explanation: z.string().describe('Brief explanation for the relevance judgment'),
});

// 政策建议评估结果模型
const PolicyEvaluationResult = z.object({
  isComplete: z.boolean().describe('Whether the policy recommendations are sufficiently complete'),
  score: z.number().min(0).max(1).describe('Completeness score (0-1)'),
  missingAspects: z.array(z.string()).describe('Aspects that are missing or incomplete in the recommendations'),
  improvementSuggestions: z.array(z.string()).describe('Specific suggestions for improving the recommendations'),
});

// 定义状态类型
const StateAnnotation = Annotation.Root({
  messages: Annotation<BaseMessage[]>({
    reducer: (x, y) => x.concat(y),
  }),
  query: Annotation<string>({
    default: () => '',
    reducer: (_, y) => y,
  }),
  content: Annotation<string>({
    default: () => '',
    reducer: (_, y) => y,
  }),
  isRelevant: Annotation<boolean>({
    default: () => false,
    reducer: (_, y) => y,
  }),
  boundaryItem: Annotation<any>({
    default: () => null,
    reducer: (_, y) => y,
  }),
  cycleCount: Annotation<number>({
    default: () => 0,
    reducer: (x, _) => x + 1,
  }),
  evaluationResult: Annotation<any>({
    default: () => null,
    reducer: (_, y) => y,
  }),
  feedbackSuggestions: Annotation<string[]>({
    default: () => [],
    reducer: (_, y) => y,
  }),
  rawContent: Annotation<string>({
    default: () => '',
    reducer: (_, y) => y,
  }),
  sourceInfo: Annotation<string>({
    default: () => 'Unknown Source',
    reducer: (_, y) => y,
  }),
});

/**
 * 检查内容是否与查询相关
 */
async function checkRelevance(
  state: typeof StateAnnotation.State,
): Promise<Partial<typeof StateAnnotation.State>> {
  console.log('---    CHECKING RELEVANCE    ---');

  const { query, content } = state;
  
  // 创建LLM实例
  const model = new ChatOpenAI({
    apiKey: openai_api_key,
    modelName: openai_chat_model,
    temperature: 0,
  });

  // 创建相关性判断工具
  const relevanceTool = {
    name: 'judge_relevance',
    description: 'Judge the relevance of research content to the user query',
    schema: RelevanceResult,
  };

  // 创建提示模板
  const relevancePrompt = ChatPromptTemplate.fromTemplate(`
    Your task is to determine if the provided research content is relevant to the user's query.\n
    
    Query: {query}\n \n
    
    Content: \n
    {content} \n \n
    
    Make your judgment based on:\n
    Topical alignment - Does the content discuss the same subject matter as the query?\n
    
    Respond with:\n
    - isRelevant: true if the content is relevant, or some parts of the content is relevant, false if the content is basically inrelevant\n
    - confidence: your confidence level from 0.0 to 1.0\n
    - explanation: a brief explanation for your judgment\n
  `);

  // 绑定工具
  const chain = relevancePrompt.pipe(model.bindTools([relevanceTool], {
    tool_choice: relevanceTool.name,
  }));

  try {
    // 解析源信息
    let sourceInfo = "Unknown Source";
    const sourceMatch = content.match(/SOURCE: (.*?)\n\n/);
    if (sourceMatch) {
      sourceInfo = sourceMatch[1];
    }
    
    // 提取实际内容部分
    let actualContent = content;
    const contentMatch = content.match(/CONTENT: ([\s\S]*)/);
    if (contentMatch) {
      actualContent = contentMatch[1];
    }
    
    // 执行相关性判断
    const response = await chain.invoke({
      query,
      content,
    }) as AIMessage;

    let isRelevant = false;
    let explanation = "Couldn't determine relevance";

    // 解析工具调用结果
    if (response.tool_calls && response.tool_calls.length > 0) {
      const result = response.tool_calls[0].args;
      isRelevant = result.isRelevant;
      explanation = result.explanation;
      
      console.log(`Relevance: ${isRelevant ? 'YES' : 'NO'} (Confidence: ${result.confidence.toFixed(2)})`);
      console.log(`Explanation: ${explanation}`);
    }

    return {
      isRelevant,
      sourceInfo,
      rawContent: actualContent,
      messages: [
        new HumanMessage(`Relevance check result: ${isRelevant ? 'Relevant' : 'Not relevant'}\nExplanation: ${explanation}`),
      ],
    };
  } catch (error) {
    console.error('Error checking relevance:', error);
    return {
      isRelevant: false,
      messages: [
        new HumanMessage('Error checking relevance. Treating as not relevant.'),
      ],
    };
  }
}

/**
 * 提取边界信息
 */
async function boundaryExtract(
  state: typeof StateAnnotation.State,
): Promise<Partial<typeof StateAnnotation.State>> {
  const cycleCount = state.cycleCount;
  console.log(`---    EXTRACTING BOUNDARY INFORMATION (CYCLE ${cycleCount + 1})    ---`);

  const { query, rawContent, sourceInfo, feedbackSuggestions } = state;
  
  // 创建LLM实例
  const model = new ChatOpenAI({
    apiKey: openai_api_key,
    modelName: openai_chat_model,
    temperature: 0,
  });

  // 创建边界提取工具
  const extractTool = {
    name: 'extract_boundary',
    description: 'Extract key boundary information and policy recommendations from scientific research content',
    schema: BoundaryItem,
  };

  // 基本提示模板
  let promptTemplate = `
    You are a research assistant analyzing scientific papers about material flow analysis and circular economy.\n
    Extract precise boundary information from the provided content, focusing on aspects relevant to: {query}\n

    Content:\n
    {content}\n

    Extract the following key boundary information:\n
    
    1. Source: Extract the complete citation or reference information\n
    2. Spatial Scope: What specific geographic area or system boundary does the study cover? Be as detailed as possible.\n
    3. Time Range: What specific time period does the study analyze? Include start and end years.\n
    4. Policy Recommendations: Extract all policy recommendations from the study. Provide them as complete sentences with context.\n
       - Include the rationale and context for each recommendation\n
       - Preserve the specific language used in the original text\n
    
    Provide only factual information directly stated in the text.\n
    For policy recommendations, extract all recommendations as separate items, ensuring each one is complete and contextual.\n
  `;
  
  // 如果有改进建议，则添加到提示中
  if (feedbackSuggestions && feedbackSuggestions.length > 0) {
    promptTemplate += `\n\nIMPORTANT FEEDBACK - Please address these issues with your previous extraction:
    ${feedbackSuggestions.map((suggestion, i) => `${i+1}. ${suggestion}`).join('\n    ')}
    
    Make sure to thoroughly review the content again and extract more comprehensive and contextual policy recommendations.`;
  }
  
  // 创建提示模板
  const extractPrompt = ChatPromptTemplate.fromTemplate(promptTemplate);

  // 绑定工具
  const chain = extractPrompt.pipe(model.bindTools([extractTool], {
    tool_choice: extractTool.name,
  }));

  try {
    // 执行边界提取
    const response = await chain.invoke({
      query,
      content: rawContent,
    }) as AIMessage;

    // 解析工具调用结果
    if (response.tool_calls && response.tool_calls.length > 0) {
      const boundaryData = response.tool_calls[0].args;
      
      // 确保source使用原始source信息
      boundaryData.source = sourceInfo;
      
      console.log(`Extracted boundary data from: ${sourceInfo.substring(0, 50)}...`);
      console.log(`Found ${boundaryData.policyRecommendations.length} policy recommendations`);
      
      // 验证数据符合模式
      const validatedItem = BoundaryItem.parse(boundaryData);
      
      return {
        boundaryItem: validatedItem,
        cycleCount: state.cycleCount,
        messages: [
          new AIMessage({ content: JSON.stringify(validatedItem, null, 2) }),
        ],
      };
    }
    
    // 没有提取到边界数据
    return {
      boundaryItem: null,
      cycleCount: state.cycleCount,
      messages: [
        new AIMessage({ content: 'Failed to extract boundary information' }),
      ],
    };
  } catch (error) {
    console.error('Error extracting boundary information:', error);
    return {
      boundaryItem: null,
      cycleCount: state.cycleCount,
      messages: [
        new AIMessage({ content: 'Error extracting boundary information' }),
      ],
    };
  }
}

/**
 * 评估政策建议的完整性
 */
async function evaluatePolicyRecommendations(
  state: typeof StateAnnotation.State,
): Promise<Partial<typeof StateAnnotation.State>> {
  console.log('---    EVALUATING POLICY RECOMMENDATIONS    ---');

  const { boundaryItem, rawContent } = state;
  
  if (!boundaryItem || !boundaryItem.policyRecommendations || boundaryItem.policyRecommendations.length === 0) {
    console.log('No policy recommendations to evaluate');
    return {
      evaluationResult: {
        isComplete: false,
        score: 0,
        missingAspects: ['No policy recommendations found'],
        improvementSuggestions: ['Extract policy recommendations from the text']
      }
    };
  }
  
  // 创建LLM实例
  const model = new ChatOpenAI({
    apiKey: openai_api_key,
    modelName: openai_chat_model,
    temperature: 0,
  });

  // 创建评估工具
  const evaluationTool = {
    name: 'evaluate_policy_recommendations',
    description: 'Evaluate the completeness and quality of extracted policy recommendations',
    schema: PolicyEvaluationResult,
  };

  // 创建提示模板
  const evaluationPrompt = ChatPromptTemplate.fromTemplate(`
    Evaluate the completeness and quality of the following extracted policy recommendations from a scientific paper.\n
    
    Original content:\n
    {original_content}\n \n
    
    Extracted policy recommendations:\n
    {policy_recommendations}\n \n
    
    Evaluate based on these criteria:\n
    1. Comprehensiveness - Do the recommendations cover all policy suggestions mentioned in the original text?\n
    2. Contextual detail - Do they include the supporting context and rationale? \n
    3. Accuracy - Do they accurately reflect the author's intentions without distortion?\n
    
    Respond with:\n
    - isComplete: true if the recommendations are sufficiently complete, false otherwise\n
    - score: a score from 0.0 to 1.0 indicating overall completeness\n
    - missingAspects: list specific aspects that are missing or incomplete\n
    - improvementSuggestions: specific suggestions for how to improve the recommendations\n
  `);

  // 绑定工具
  const chain = evaluationPrompt.pipe(model.bindTools([evaluationTool], {
    tool_choice: evaluationTool.name,
  }));

  try {
    // 执行评估
    const response = await chain.invoke({
      original_content: rawContent,
      policy_recommendations: boundaryItem.policyRecommendations.join('\n\n'),
    }) as AIMessage;

    // 解析工具调用结果
    if (response.tool_calls && response.tool_calls.length > 0) {
      const evaluationResult = response.tool_calls[0].args;
      
      console.log(`Evaluation result: ${evaluationResult.isComplete ? 'COMPLETE' : 'INCOMPLETE'} (Score: ${evaluationResult.score.toFixed(2)})`);
      
      if (!evaluationResult.isComplete) {
        console.log('Missing aspects:', evaluationResult.missingAspects);
        console.log('Improvement suggestions:', evaluationResult.improvementSuggestions);
      }
      
      return {
        evaluationResult,
        feedbackSuggestions: evaluationResult.improvementSuggestions,
      };
    }
    
    // 没有得到评估结果
    return {
      evaluationResult: {
        isComplete: true,  // 默认为完整，避免无限循环
        score: 0.5,
        missingAspects: ['Evaluation failed'],
        improvementSuggestions: []
      }
    };
  } catch (error) {
    console.error('Error evaluating policy recommendations:', error);
    return {
      evaluationResult: {
        isComplete: true,  // 默认为完整，避免无限循环
        score: 0.5,
        missingAspects: ['Evaluation error'],
        improvementSuggestions: []
      }
    };
  }
}


function decideRelevance(state: typeof StateAnnotation.State): string {
  // 如果内容相关，进行边界提取
  if (state.isRelevant) {
    console.log('---    DECISION: CONTENT IS RELEVANT, PROCEED TO BOUNDARY EXTRACTION    ---');
    return 'extract';
  }
  
  // 否则结束处理
  console.log('---    DECISION: CONTENT IS NOT RELEVANT, END PROCESSING    ---');
  return 'end';
}


function decideRegenerate(state: typeof StateAnnotation.State): string {
  const { evaluationResult, cycleCount } = state;
  
  // 如果评估结果表明建议完整，或已达到最大循环次数，则结束
  if (evaluationResult.isComplete || cycleCount >= 2) {
    if (evaluationResult.isComplete) {
      console.log('---    DECISION: POLICY RECOMMENDATIONS ARE COMPLETE, END PROCESSING    ---');
    } else {
      console.log(`---    DECISION: REACHED MAX CYCLES (${cycleCount}), END PROCESSING    ---`);
    }
    return 'end';
  }
  
  // 否则返回重新生成
  console.log('---    DECISION: POLICY RECOMMENDATIONS ARE INCOMPLETE, REGENERATING    ---');
  return 'regenerate';
}

// 创建工作流图
const workflow = new StateGraph(StateAnnotation)
  .addNode('checkRelevance', checkRelevance)
  .addNode('boundaryExtract', boundaryExtract)
  .addNode('evaluatePolicyRecommendations', evaluatePolicyRecommendations)
  .addEdge('__start__', 'checkRelevance')
  .addConditionalEdges(
    'checkRelevance',
    decideRelevance,
    {
      'extract': 'boundaryExtract',
      'end': '__end__',
    }
  )
  .addEdge('boundaryExtract', 'evaluatePolicyRecommendations')
  .addConditionalEdges(
    'evaluatePolicyRecommendations',
    decideRegenerate,
    {
      'regenerate': 'boundaryExtract',
      'end': '__end__',
    }
  );

// 编译工作流
export const graph = workflow.compile({});

