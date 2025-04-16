import { Client } from '@langchain/langgraph-sdk';
import { RemoteGraph } from '@langchain/langgraph/remote';
import dotenv from 'dotenv';
import fs from 'fs';
import path from 'path';

// 加载环境变量
dotenv.config();

// 处理配置
const PROCESSING_BATCH_SIZE = 5; // 每个批次处理的文章数
const MAX_RETRIES = 3; // 最大重试次数

// 环境配置
const url = process.env.DEPLOYMENT_URL ?? '';
const apiKey = process.env.LANGSMITH_API_KEY ?? '';
const extractGraphName = 'extract_agent';

// 初始化客户端和远程图
const client = new Client({ apiUrl: url, apiKey: apiKey });
const extractGraph = new RemoteGraph({ 
  graphId: extractGraphName, 
  url: url, 
  apiKey: apiKey 
});

// 文章接口定义
interface Article {
    "Article Title"?: string;
    "DOI"?: string;
    "Abstract"?: string;
    "Authors"?: string;
    "Source Title"?: string;
    "Publication Year"?: number;
    "searchResults"?: any[];  // 存储搜索结果
    "spatialScope": string ;
    "timeRange": string;
    "subSections": Array<{
      subsectionTitle: string;
      mainContent: string;
      policyRecommendations: string[];
      }>;
    [key: string]: any;  // 允许其他字段
  }


// 提取单篇文章信息
async function extractArticleInfo(article: Article): Promise<Article> {
    try {
      const thread = await client.threads.create();
      // 设置配置
      const config = { 
        configurable: { thread_id: thread.thread_id },
        recursionLimit: 10,
      };
  
      // 获取文章摘要和搜索结果
    const abstract = article.Abstract || '';
    const searchResults = article.searchResults || [];

    // 将搜索结果转换为单一字符串
    let fullText = '';
    if (Array.isArray(searchResults)) {
    // 如果是数组，将所有元素连接成一个字符串
    fullText = searchResults.map(item => {
        if (typeof item === 'string') return item;
        return JSON.stringify(item);
    }).join('\n\n');
    } else if (typeof searchResults === 'string') {
    // 如果已经是字符串，直接使用
    fullText = searchResults;
    } else {
    // 其他情况尝试转换为字符串
    fullText = JSON.stringify(searchResults);
    }

    // 基于字符长度分割文本
    const midpoint = Math.ceil(fullText.length / 2);
    const text1 = fullText.substring(0, midpoint);  // 前半部分
    const text2 = fullText.substring(midpoint);     // 后半部分

    console.log(`Article "${article["Article Title"] || 'Untitled'}" - Split searchResults text: ${text1.length} chars and ${text2.length} chars`);

    // 调用远程图处理，使用处理后的文本
    const result = await extractGraph.invoke({ 
    abstract: abstract,
    searchResults1: [{role: "user", content: text1}],
    searchResults2: [{role: "user", content: text2}],
    }, config);
      // 检查结果
      if (!result) {
        console.log(`Empty result returned for article "${article["Article Title"] || 'Untitled'}"`);
        return {
          "Article Title": article["Article Title"],
          "DOI": article["DOI"],
          "Authors": article["Authors"],
          "Source Title": article["Source Title"],
          "Publication Year": article["Publication Year"],
          "spatialScope": 'No result returned',
          "timeRange": 'No result returned',
          "subSections": []
        };
      }
  
      // 处理有效的输出
    // 在 extractArticleInfo 函数中处理 API 返回结果
if (result.output && result.output.length > 0) {
    // 从result.output获取提取的信息
    const extractedData = result.output[0];
    
    // 确保 extractedData 的各个字段都是预期的类型
    let spatialScope = "Not specified";
    let timeRange = "Not specified";
    let subSections = [];
    
    // 处理 spatialScope
    if (typeof extractedData.spatialScope === 'string') {
      spatialScope = extractedData.spatialScope;
    } else if (extractedData.spatialScope && typeof extractedData.spatialScope === 'object') {
      // 尝试从对象中提取字符串值
      spatialScope = extractedData.spatialScope.toString();
    }
    
    // 处理 timeRange
    if (typeof extractedData.timeRange === 'string') {
      timeRange = extractedData.timeRange;
    } else if (extractedData.timeRange && typeof extractedData.timeRange === 'object') {
      timeRange = extractedData.timeRange.toString();
    }
    
    // 处理 subSections
    if (Array.isArray(extractedData.subSections)) {
      subSections = extractedData.subSections;
    }
    
    // 如果 subSections 为空，添加一个默认内容
    if (subSections.length === 0) {
      subSections = [{
        subsectionTitle: "No Policy Sections Found",
        mainContent: "The analysis could not identify specific policy sections in this document.",
        policyRecommendations: ["No specific policy recommendations could be extracted from this document."]
      }];
    }
    console.log(`Successfully extracted info for article "${article["Article Title"] || 'Untitled'}" with ${subSections.length} policy sections`);
    
    // 返回处理后的文章
    return {
      "Article Title": article["Article Title"],
      "DOI": article["DOI"],
      "Authors": article["Authors"],
      "Source Title": article["Source Title"],
      "Publication Year": article["Publication Year"],
      "spatialScope": spatialScope,
      "timeRange": timeRange,
      "subSections": subSections
    };
  }
    else {
        console.log(`No valid data in output for article "${article["Article Title"] || 'Untitled'}"`);
        
        // 返回错误状态
    return {
        "Article Title": article["Article Title"],
        "DOI": article["DOI"],
        "Authors": article["Authors"],
        "Source Title": article["Source Title"],
        "Publication Year": article["Publication Year"],
        "spatialScope": 'Error during processing',
        "timeRange": 'Error during processing',
        "subSections": [] 

    };
      }
    } catch (error) {
      console.error(`Error processing article "${article["Article Title"] || 'Untitled'}":`, error);
      
      return {
        "Article Title": article["Article Title"],
        "DOI": article["DOI"],
        "Authors": article["Authors"],
        "Source Title": article["Source Title"],
        "Publication Year": article["Publication Year"],
        "spatialScope": 'Error during processing',
        "timeRange": 'Error during processing',
        "subSections": []
      };
    }
  }

// 批量处理文章信息
async function batchExtractArticles(articles: Article[]): Promise<Article[]> {
    const results: Article[] = [];
    
    // 分批处理文章
    for (let i = 0; i < articles.length; i += PROCESSING_BATCH_SIZE) {
        const batch = articles.slice(i, i + PROCESSING_BATCH_SIZE);
        console.log(`Processing batch ${Math.floor(i / PROCESSING_BATCH_SIZE) + 1} (articles ${i + 1} to ${Math.min(i + PROCESSING_BATCH_SIZE, articles.length)})`);
        
        // 并行处理当前批次
        const batchPromises = batch.map(async (article) => {
            // 检查searchResults是否包含错误信息
            const searchResults = article.searchResults || [];
            const hasInternalServerError = searchResults.some(result => 
                typeof result === 'string' && 
                result.includes("nal Server Error\n Please fix your mistakes.")
            );
            
            // 如果包含内部服务器错误，则跳过处理
            if (hasInternalServerError) {
                console.log(`Skipping article with server error: "${article["Article Title"] || 'Untitled'}"`);
                return {
                    ...article,
                    spatialScope: 'Skipped due to server error',
                    timeRange: 'Skipped due to server error',
                    subSections: [] 
                };
            }
            
            let retries = 0;
            let success = false;
            let result: Article | null = null;
            
            // 添加重试逻辑
            while (!success && retries < MAX_RETRIES) {
                try {
                    result = await extractArticleInfo(article);
                    success = true;
                } catch (error) {
                    retries++;
                    console.error(`Error processing article "${article["Article Title"] || 'Untitled'}", attempt ${retries}/${MAX_RETRIES}: ${error}`);
                    
                    if (retries >= MAX_RETRIES) {
                        console.error(`Max retries reached for article "${article["Article Title"] || 'Untitled'}"`);
                        // 返回原始文章但添加错误信息
                        result = {
                            ...article,
                            spatialScope: 'Error during extraction',
                            timeRange: 'Error during extraction',
                            subSections: [],
                            };
                    } else {
                        // 等待一段时间后重试
                        await new Promise(resolve => setTimeout(resolve, 1000 * retries));
                    }
                }
            }
            
            // 如果 result 没有获取到，则返回一个默认值
            if (result) {
                return result;
            } else {
                return {
                    ...article,
                    spatialScope: 'No result',
                    timeRange: 'No result',
                    subSections: []
                } as Article;
            }
        });
        
        // 等待当前批次完成
        const batchResults = await Promise.all(batchPromises);
        results.push(...batchResults.filter(Boolean) as Article[]);
    }
    return results;
}
// 从文件加载文章
async function loadArticlesFromFile(filePath: string): Promise<Article[]> {
    try {
        // 读取文件
        const data = fs.readFileSync(filePath, 'utf8');
        const articles: Article[] = JSON.parse(data);
        
        // 过滤有效文章 (必须有标题、摘要和搜索结果)
        const validArticles = articles.filter(article => 
            article && 
            article["Article Title"] && 
            article["searchResults"] && 
            Array.isArray(article["searchResults"]) &&
            article["searchResults"].length > 0
        );
        
        console.log(`Loaded ${validArticles.length} valid articles out of ${articles.length} total`);
        console.log(`Filtered out ${articles.length - validArticles.length} articles without searchResults`);
        
        // 返回有效文章 (所有文章都已确保有searchResults)
        return validArticles;
    } catch (error) {
        console.error(`Error loading articles from file: ${error}`);
        return [];
    }
}

// 保存处理结果
async function saveResults(articles: Article[], outputPath: string): Promise<void> {
    try {
        const outputDir = path.dirname(outputPath);
        
        // 确保输出目录存在
        if (!fs.existsSync(outputDir)) {
            fs.mkdirSync(outputDir, { recursive: true });
        }
        
        fs.writeFileSync(outputPath, JSON.stringify(articles, null, 2), 'utf8');
        console.log(`Results saved to ${outputPath}`);
    } catch (error) {
        console.error(`Error saving results: ${error}`);
    }
}

// 主函数
async function main(): Promise<void> {
    // 命令行参数: 输入文件路径和输出文件路径
    const args = process.argv.slice(2);
    const inputPath = args[0] || './data/process/English_articles_test.json';
    const outputPath = args[1] || './data/output_articles.json';
    
    console.log(`Loading articles from ${inputPath}`);
    const articles = await loadArticlesFromFile(inputPath);
    
    if (articles.length === 0) {
        console.log('No articles found. Exiting.');
        return;
    }
    
    console.log(`Starting batch extraction of ${articles.length} articles`);
    const results = await batchExtractArticles(articles);
    
    console.log(`Extraction completed. Processed ${results.length} articles.`);
    await saveResults(results, outputPath);
}

// 执行主函数
if (require.main === module) {
    main().catch(error => {
        console.error('Error in main function:', error);
        process.exit(1);
    });
}
