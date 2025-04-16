import { Client } from '@langchain/langgraph-sdk';
import { RemoteGraph } from '@langchain/langgraph/remote';
import dotenv from 'dotenv';
import fs from 'fs';
import path from 'path';
import { v4 as uuidv4 } from 'uuid';

dotenv.config();

// 环境配置
const url = process.env.DEPLOYMENT_URL ?? '';
const apiKey = process.env.LANGSMITH_API_KEY ?? '';
const sortAgentGraphName = 'sort_agent';

// 文章接口定义 - 匹配saved_articles.json格式
interface Article {
  "Article Title"?: string;
  "Abstract"?: string;
  "DOI"?: string;
  "Authors"?: string;
  "Source Title"?: string;
  "Publication Date"?: string;
  "Publication Year"?: number;
  "isRelevant"?: boolean;  // 添加相关性标志
  [key: string]: any;  // 允许其他字段
}

// 初始化客户端和远程图
const client = new Client({ apiUrl: url, apiKey: apiKey });
const sortAgentGraph = new RemoteGraph({ 
  graphId: sortAgentGraphName, 
  url: url, 
  apiKey: apiKey 
});

/**
 * 检查单篇文章是否与中国建筑物质流分析相关
 */
async function checkArticleRelevance(article: Article): Promise<boolean> {
  try {
    const title = article["Article Title"] || '';
    
    if (title && title.length > 0) {
      console.log(`Checking relevance for article: "${title.substring(0, 50)}..."`);
    } else {
      console.log(`Checking relevance for article with missing title`);
    }
    
    // 创建新线程
    const thread = await client.threads.create();
    const config = { 
      configurable: { thread_id: thread.thread_id },
      recursionLimit: 10
    };
    
    // 准备输入数据 - 只提取标题和摘要
    const abstract = article.Abstract || '';
    
    // 调用相关性判断代理
    const result = await sortAgentGraph.invoke({ 
      title: title,
      abstract: abstract
    }, config);
    
    // 检查结果
    if (result.isRelevant !== undefined) {
      console.log(`Article relevance result: ${result.isRelevant ? 'RELEVANT' : 'NOT RELEVANT'}`);
      return result.isRelevant;
    } else {
      // 如果没有明确的相关性标志，尝试从消息中提取
      if (result.messages && result.messages.length > 0) {
        const lastMessage = result.messages[result.messages.length - 1];
        const content = typeof lastMessage.content === 'string' 
          ? lastMessage.content.toLowerCase() 
          : '';
        const isRelevant = content.includes('Yes');
        console.log(`Article relevance (extracted from message): ${isRelevant ? 'RELEVANT' : 'NOT RELEVANT'}`);
        return isRelevant;
      }
      
      console.log('Could not determine relevance, treating as not relevant');
      return false;
    }
  } catch (error) {
    console.error('Error checking article relevance:', error);
    return false;  // 错误时默认为不相关
  }
}

/**
 * 执行完整的文章筛选工作流
 */
export async function runSortWorkflow(inputFilePath: string, outputBasePath?: string): Promise<any> {
  try {
    // 创建会话标识
    const sessionId = uuidv4().substring(0, 8);
    
    // 确定输出文件路径
    let relevantOutputPath: string;
    let irrelevantOutputPath: string;
    
    if (!outputBasePath) {
      const outputDir = path.join(__dirname, '../../data');
      if (!fs.existsSync(outputDir)) {
        fs.mkdirSync(outputDir, { recursive: true });
      }
      const basename = path.basename(inputFilePath, path.extname(inputFilePath));
      relevantOutputPath = path.join(outputDir, `${basename}_relevant_${sessionId}${path.extname(inputFilePath)}`);
      irrelevantOutputPath = path.join(outputDir, `${basename}_irrelevant_${sessionId}${path.extname(inputFilePath)}`);
    } else {
      // 如果提供了基础路径，基于它创建两个输出文件
      const outputDir = path.dirname(outputBasePath);
      const basename = path.basename(outputBasePath, path.extname(outputBasePath));
      const extension = path.extname(outputBasePath);
      relevantOutputPath = path.join(outputDir, `${basename}_relevant${extension}`);
      irrelevantOutputPath = path.join(outputDir, `${basename}_irrelevant${extension}`);
    }
    
    console.log(`==========================================`);
    console.log(`Starting article sorting workflow`);
    console.log(`Session ID: ${sessionId}`);
    console.log(`Input file: ${inputFilePath}`);
    console.log(`Relevant articles output: ${relevantOutputPath}`);
    console.log(`Irrelevant articles output: ${irrelevantOutputPath}`);
    console.log(`==========================================`);
    
    // 步骤1: 读取文章数据
    console.log(`\n====== STEP 1: READING ARTICLES ======`);
    
    // 确保输入文件存在
    if (!fs.existsSync(inputFilePath)) {
      throw new Error(`Input file not found: ${inputFilePath}`);
    }
    
    // 读取文件内容
    const fileData = fs.readFileSync(inputFilePath, 'utf-8');
    const articles: Article[] = JSON.parse(fileData);
    
    console.log(`Loaded ${articles.length} articles from input file`);
    
    // 步骤2: 检查文章相关性
    console.log(`\n====== STEP 2: CHECKING RELEVANCE ======`);
    
    // 并行处理文章，限制并发数
    const relevancePromises: Promise<{ article: Article; isRelevant: boolean }>[] = [];
    const maxConcurrent = 5;
    let activePromises = 0;
    let index = 0;
    
    while (index < articles.length || activePromises > 0) {
      if (index < articles.length && activePromises < maxConcurrent) {
        // 启动新的相关性检查任务
        console.log(`Starting relevance check for article ${index + 1}/${articles.length}`);
        const article = articles[index];
        
        const promise = checkArticleRelevance(article)
          .then(isRelevant => {
            activePromises--;
            // 复制原始文章对象并添加相关性标志
            const annotatedArticle = { ...article, isRelevant };
            return { article: annotatedArticle, isRelevant };
          })
          .catch(error => {
            console.error(`Error in relevance promise ${index + 1}:`, error);
            activePromises--;
            // 发生错误时，将文章标记为不相关
            const annotatedArticle = { ...article, isRelevant: false };
            return { article: annotatedArticle, isRelevant: false };
          });
        
        relevancePromises.push(promise);
        activePromises++;
        index++;
      } else {
        // 等待一个任务完成
        await new Promise(resolve => setTimeout(resolve, 100));
      }
    }
    
    // 等待所有相关性检查任务完成
    const allRelevanceResults = await Promise.all(relevancePromises);
    
    // 分组文章
    const relevantArticles = allRelevanceResults
      .filter(result => result.isRelevant)
      .map(result => result.article);
      
    const irrelevantArticles = allRelevanceResults
      .filter(result => !result.isRelevant)
      .map(result => result.article);
    
    console.log(`\nSorted ${allRelevanceResults.length} articles in total`);
    console.log(`${relevantArticles.length} articles are relevant (keep)`);
    console.log(`${irrelevantArticles.length} articles are irrelevant (discard)`);
    
    // 步骤3: 保存分组结果
    console.log(`\n====== STEP 3: SAVING RESULTS ======`);
    
    // 保存相关文章
    fs.writeFileSync(relevantOutputPath, JSON.stringify(relevantArticles, null, 2));
    console.log(`Relevant articles saved to: ${relevantOutputPath}`);
    
    // 保存不相关文章
    fs.writeFileSync(irrelevantOutputPath, JSON.stringify(irrelevantArticles, null, 2));
    console.log(`Irrelevant articles saved to: ${irrelevantOutputPath}`);
    
    // 返回结果摘要
    return {
      success: true,
      totalArticles: articles.length,
      relevantArticles: relevantArticles.length,
      irrelevantArticles: irrelevantArticles.length,
      relevantOutputPath: relevantOutputPath,
      irrelevantOutputPath: irrelevantOutputPath
    };
  } catch (error) {
    console.error('Error in article sorting workflow:', error);
    return {
      success: false,
      error: error instanceof Error ? error.message : 'Unknown error in workflow'
    };
  }
}

// 如果直接运行此脚本，处理命令行参数
if (require.main === module) {
  // 获取输入文件路径
  const inputFilePath = process.argv[2] || path.join(__dirname, '../../data/English_articles_1.json');
  
  // 获取可选的输出文件路径
  const outputBasePath = process.argv[3];
  
  console.log("开始执行文章分组工作流...");
  runSortWorkflow(inputFilePath, outputBasePath)
    .then(results => {
      if (!results.success) {
        console.log(`\n处理失败: ${results.error}`);
        return;
      }
      
      console.log(`\n===== 文章分组摘要 =====`);
      console.log(`总文章数: ${results.totalArticles}`);
      console.log(`相关文章数: ${results.relevantArticles}`);
      console.log(`不相关文章数: ${results.irrelevantArticles}`);
      console.log(`相关性比率: ${((results.relevantArticles / results.totalArticles) * 100).toFixed(2)}%`);
      console.log(`相关文章已保存至: ${results.relevantOutputPath}`);
      console.log(`不相关文章已保存至: ${results.irrelevantOutputPath}`);
    })
    .catch(error => {
      console.error("处理过程中出现错误:", error);
    });
}