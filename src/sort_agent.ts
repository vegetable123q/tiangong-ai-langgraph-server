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

// 智能体的状态类型
const StateAnnotation = Annotation.Root({
  // 消息历史
  messages: Annotation<BaseMessage[]>({
    reducer: (x, y) => x.concat(y),
  }),
  // 当前处理的文章标题
  title: Annotation<string>({
    default: () => '',
    reducer: (_, y) => y,
  }),
  // 当前处理的文章摘要
  abstract: Annotation<string>({
    default: () => '',
    reducer: (_, y) => y,
  }),
  // 相关性判断结果
  isRelevant: Annotation<boolean>({
    default: () => false,
    reducer: (_, y) => y,
  }),

});


/**
 * 判断文章是否与中国建筑物质流分析相关
 */
async function judgeRelevance(
    state: typeof StateAnnotation.State,
  ): Promise<Partial<typeof StateAnnotation.State>> {
    console.log('---    JUDGING RELEVANCE    ---');
  
    const { title, abstract } = state;
    
    // 定义输出模式
  const RelevanceSchema = z.object({
  judgment: z.string().describe('Answer "Yes" if the article relates to material flows in built environment (buildings, infrastructure) OR analyzes carbon emissions through material flow in China; otherwise answer "No"'),
  reason: z.string().describe('Brief explanation for your judgment, highlighting connections to built environment, construction materials, or material flow analysis in China'),
});

  // 创建工具
  const relevanceTool = {
  name: 'judge_article_relevance',
  description: 'Evaluate if an article is relevant to China\'s material flows in built environment, or material flow-based carbon accounting',
  schema: RelevanceSchema,
};

// 创建LLM实例
  const model = new ChatOpenAI({
  apiKey: openai_api_key,
  modelName: openai_chat_model,
  temperature: 0,
});

// 创建提示模板
  const relevancePrompt = ChatPromptTemplate.fromTemplate(`  
  You are given an article's title and abstract. Please exaime if the study is related to building material flow research in China.

  Please evaluate the article title and abstract to determine if it the study relates to material flows in the built environment (buildings, infrastructure) or key construction materials or the study analyzes carbon emissions through material flow approaches \n
  
  Article Title: {title}\n\n
  
  Abstract: {abstract}\n\n
  
  Important considerations:\n
  - The study should focus on built environment broadly (buildings, residential areas, urban infrastructure, etc.)\n
  - The study area should be within China\n  
  - The study that related to key construction materials should also be related to built environment\n
  
  Please provide a clear "Yes" or "No" judgment.
`);
  
      // 绑定工具
      const chain = relevancePrompt.pipe(model.bindTools([relevanceTool], {
        tool_choice: relevanceTool.name,
      }));
      
      // 调用模型进行判断
      const response = await chain.invoke({
        title: title || "No title",
        abstract: abstract || "No abstract"
      }) as AIMessage;
      
      // 尝试从工具调用中提取结果
      let judgment = "No"; // 默认为不相关
      
      if (response.tool_calls && response.tool_calls.length > 0) {
        const toolCall = response.tool_calls[0];
        judgment = toolCall.args.judgment;
      } else {
        // 如果没有工具调用，则尝试从文本内容中提取
        const content = response.content.toString().trim().toLowerCase();
        if (content.includes('yes')) {
          judgment = "Yes";
        }
      }
      
      // 记录判断结果
      console.log(`Title: "${title.substring(0, 50)}..."`);
      console.log(`Judgment: ${judgment}`);
      
      // 设置相关性标志
      const isRelevant = judgment.toLowerCase() === "yes";
      
      // 返回判断结果
      return {
        isRelevant,
        messages: [
          new AIMessage(`${judgment}`),
        ],
      };
    } 


// 工作流图构建
const workflow = new StateGraph(StateAnnotation)
  .addNode('judgeRelevance', judgeRelevance)
  .addEdge('__start__', 'judgeRelevance')
  .addEdge('judgeRelevance', '__end__');

// 编译工作流
export const graph = workflow.compile({});