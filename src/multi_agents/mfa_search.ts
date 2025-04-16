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
const mfaSearchGraphName = 'mfa_search_agent';

// 文章接口定义
interface Article {
  "Article Title"?: string;
  "DOI"?: string;
  "Abstract"?: string;
  "Authors"?: string;
  "Source Title"?: string;
  "Publication Year"?: number;
  "searchResults"?: any[];  // 用于存储搜索结果
  [key: string]: any;  // 允许其他字段
}

// 初始化客户端和远程图
const client = new Client({ apiUrl: url, apiKey: apiKey });
const mfaSearchGraph = new RemoteGraph({ 
  graphId: mfaSearchGraphName, 
  url: url, 
  apiKey: apiKey 
});


async function searchArticleContent(article: Article): Promise<any[]> {
  if (!article.DOI) {
    console.log(`无法搜索文章: "${article["Article Title"] || '未知标题'}" - DOI不存在`);
    return [];
  }
  
  const doi = article.DOI;
  console.log(`正在检索文章: "${article["Article Title"]?.substring(0, 50)}..." DOI: ${doi}`);
  
  // 创建新线程
  const thread = await client.threads.create();
  
  // 设置配置
  const config = { 
    configurable: { thread_id: thread.thread_id },
    recursionLimit: 10,
  };
  
  // 构建查询字符串
  const query = `建筑物质流研究研究 doi:${doi}`;
  console.log(`发送查询: "${query}"`);
  
  // 调用搜索代理
  const result = await mfaSearchGraph.invoke({ messages: [{ role: "user", content: query }] }, config);
  
  // 检查结果
  if (!result) {
    console.log(`文章检索返回空结果`);
    return [];
  }
  
  // 处理不同类型的结果
  if (result.contents) {
    if (Array.isArray(result.contents)) {
      console.log(`获取到内容数组，长度: ${result.contents.length}`);
      return result.contents;
    } else {
      return [result.contents];
    }
  } else if (result.messages && Array.isArray(result.messages)) {
    const extractedContents = [];
    
    for (const msg of result.messages) {
      if (msg && typeof msg === 'object') {
        if (msg.content) {
          extractedContents.push(msg.content);
        } else if (msg.kwargs && msg.kwargs.content) {
          extractedContents.push(msg.kwargs.content);
        }
      }
    }
    
    return extractedContents.length > 0 ? extractedContents : [];
  } else {
    return [result];
  }
}

/**
 * 执行批量文章内容检索
 */
export async function enrichArticlesWithSearchResults(
  inputFilePath: string, 
  outputFilePath?: string, 
  maxConcurrent: number = 4
): Promise<any> {
  try {
    // 创建会话标识
    const sessionId = uuidv4().substring(0, 8);
    
    // 确定输出文件路径
    if (!outputFilePath) {
      const outputDir = path.join(__dirname, '../../data/process');
      if (!fs.existsSync(outputDir)) {
        fs.mkdirSync(outputDir, { recursive: true });
      }
      const basename = path.basename(inputFilePath, path.extname(inputFilePath));
      outputFilePath = path.join(outputDir, `${basename}_enriched_${sessionId}${path.extname(inputFilePath)}`);
    }
    
    console.log(`==========================================`);
    console.log(`开始文章内容检索工作流`);
    console.log(`会话ID: ${sessionId}`);
    console.log(`输入文件: ${inputFilePath}`);
    console.log(`输出文件: ${outputFilePath}`);
    console.log(`最大并发数: ${maxConcurrent}`);
    console.log(`==========================================`);
    
    // 步骤1: 读取文章数据
    console.log(`\n====== 第1步: 读取文章数据 ======`);
    
    // 确保输入文件存在
    if (!fs.existsSync(inputFilePath)) {
      throw new Error(`输入文件不存在: ${inputFilePath}`);
    }
    
    // 读取文件内容
    const fileData = fs.readFileSync(inputFilePath, 'utf-8');
    const articles: Article[] = JSON.parse(fileData);
    
    console.log(`从输入文件加载了 ${articles.length} 篇文章`);
    
    // 步骤2: 进行内容检索
    console.log(`\n====== 第2步: 进行内容检索 ======`);
    
    // 并行处理文章，限制并发数
    const searchPromises: Promise<{ article: Article; enriched: boolean }>[] = [];
    let activePromises = 0;
    let index = 0;
    
    while (index < articles.length || activePromises > 0) {
      if (index < articles.length && activePromises < maxConcurrent) {
        // 启动新的检索任务
        console.log(`开始检索文章 ${index + 1}/${articles.length}`);
        const article = articles[index];
        
        const promise = searchArticleContent(article)
          .then(searchResults => {
            activePromises--;
            // 为文章添加检索结果
            article.searchResults = searchResults;
            return { article, enriched: searchResults.length > 0 };
          })
          .catch(error => {
            console.error(`文章检索任务 ${index + 1} 出错:`, error);
            activePromises--;
            return { article, enriched: false };
          });
        
        searchPromises.push(promise);
        activePromises++;
        index++;
      } else {
        // 等待一个任务完成
        await new Promise(resolve => setTimeout(resolve, 100));
      }
    }
    
    // 等待所有检索任务完成
    const allResults = await Promise.all(searchPromises);
    
    // 统计检索结果
    const enrichedCount = allResults.filter(result => result.enriched).length;
    
    console.log(`\n共处理 ${allResults.length} 篇文章`);
    console.log(`成功添加检索结果的文章: ${enrichedCount} 篇`);
    console.log(`未能添加检索结果的文章: ${allResults.length - enrichedCount} 篇`);
    
    // 步骤3: 保存增强后的文章数据
    console.log(`\n====== 第3步: 保存增强后的文章数据 ======`);
    
    // 保存增强后的文章
    fs.writeFileSync(outputFilePath, JSON.stringify(articles, null, 2));
    console.log(`增强后的文章已保存至: ${outputFilePath}`);
    
    // 返回结果摘要
    return {
      success: true,
      totalArticles: articles.length,
      enrichedArticles: enrichedCount,
      outputFilePath: outputFilePath
    };
  } catch (error) {
    console.error('文章内容检索工作流出错:', error);
    return {
      success: false,
      error: error instanceof Error ? error.message : '工作流中出现未知错误'
    };
  }
}

// 如果直接运行此脚本，处理命令行参数
if (require.main === module) {
  // 获取输入文件路径
  const inputFilePath = process.argv[2] || path.join(__dirname, '../../data/source/English_articles.json');
  
  // 获取可选的输出文件路径
  const outputFilePath = process.argv[3];
  
  // 获取可选的最大并发数
  const maxConcurrent = parseInt(process.argv[4] || '3', 10);
  
  console.log("开始执行文章内容检索工作流...");
  enrichArticlesWithSearchResults(inputFilePath, outputFilePath, maxConcurrent)
    .then(results => {
      if (!results.success) {
        console.log(`\n处理失败: ${results.error}`);
        return;
      }
      
      console.log(`\n===== 文章检索摘要 =====`);
      console.log(`总文章数: ${results.totalArticles}`);
      console.log(`成功增强文章数: ${results.enrichedArticles}`);
      console.log(`成功率: ${((results.enrichedArticles / results.totalArticles) * 100).toFixed(2)}%`);
      console.log(`增强后的文章已保存至: ${results.outputFilePath}`);
    })
    .catch(error => {
      console.error("处理过程中出现错误:", error);
    });
}
