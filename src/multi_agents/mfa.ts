import { Client } from '@langchain/langgraph-sdk';
import { RemoteGraph } from '@langchain/langgraph/remote';
import dotenv from 'dotenv';
import PQueue from 'p-queue';
import fs from 'fs';
import path from 'path';

dotenv.config();

// 环境配置
const url = process.env.DEPLOYMENT_URL ?? '';
const apiKey = process.env.LANGSMITH_API_KEY ?? '';
const mfaSearchGraphName = 'mfa_search_agent';
const mergeAgentGraphName = 'merge_agent';

// 定义工具调用接口
interface ToolCall {
  name: string;
  args: any;
}

// 初始化客户端和远程图
const client = new Client({ apiUrl: url, apiKey: apiKey });
const mfaSearchGraph = new RemoteGraph({ 
  graphId: mfaSearchGraphName, 
  url: url, 
  apiKey: apiKey 
});
const mergeAgentGraph = new RemoteGraph({
  graphId: mergeAgentGraphName,
  url: url,
  apiKey: apiKey
});

// 定义期刊列表
const JOURNALS = [
  "JOURNAL OF INDUSTRIAL ECOLOGY",
  "RESOURCES CONSERVATION AND RECYCLING",
  "JOURNAL OF CLEANER PRODUCTION",
  "ENVIRONMENTAL SCIENCE & TECHNOLOGY",
  "BUILDING RESEARCH AND INFORMATION",
  "APPLIED ENERGY",
  "WASTE MANAGEMENT",
  "FRONTIERS IN EARTH SCIENCE",
  "SCIENTIFIC DATA",
  "ENVIRONMENTAL IMPACT ASSESSMENT REVIEW",
  "SUSTAINABILITY"
];

/**
 * 创建期刊专业查询 - 用原始查询加上期刊名
 */
function createJournalQueries(userQuery: string): string[] {
  console.log(`Creating ${JOURNALS.length} journal-specific queries`);
  return JOURNALS.map(journal => `${userQuery} in ${journal}`);
}

/**
 * 对单个查询执行MFA搜索
 */
async function performMfaSearch(query: string) {
  try {
    // 创建新线程
    const thread = await client.threads.create();
    const config = { configurable: { thread_id: thread.thread_id } };
    
    console.log(`Searching for: "${query}"`);
    const result = await mfaSearchGraph.invoke({ messages: [{ role: "user", content: query }] }, config);
    
    return {
      query,
      result: result,
      success: true
    };
  } catch (error: unknown) {
    console.error(`Error searching for "${query}":`, error);
    return {
      query,
      error: error instanceof Error ? error.message : "Unknown error",
      success: false
    };
  }
}

/**
 * 从搜索结果中提取边界数据
 */
function extractBoundaryData(searchResult: any): any[] {
  const boundaryData: any[] = [];
  
  if (!searchResult?.messages) {
    return boundaryData;
  }
  
  // 遍历所有消息寻找边界工具调用
  searchResult.messages.forEach((msg: any) => {
    if (msg?.tool_calls && Array.isArray(msg.tool_calls)) {
      msg.tool_calls.forEach((call: ToolCall) => {
        if (call.name === 'extract_boundary' && call.args) {
          boundaryData.push(call.args);
        }
      });
    }
  });
  
  return boundaryData;
}

/**
 * 合并多个MFA搜索结果（添加重试和批处理功能）
 */
async function mergeBoundaryData(boundaryData: any[]): Promise<any> {
  console.log(`Merging ${boundaryData.length} boundary data items`);
  
  try {
    // 如果数据过多，分批处理
    if (boundaryData.length > 15) {
      console.log("Large dataset detected, using batch processing");
      return await mergeInBatches(boundaryData);
    }
    
    // 创建新线程
    const mergeThread = await client.threads.create();
    const mergeConfig = { 
      configurable: { 
        thread_id: mergeThread.thread_id
      },
      // 增加超时设置
      timeout: 120000, // 120秒
      maxRetries: 3 // 最多重试3次
    };
    
    // 创建包含所有边界数据的消息
    const messages = boundaryData.map(data => ({
      role: "user",
      content: JSON.stringify(data)
    }));
    
    // 调用合并代理
    return await mergeAgentGraph.invoke({ messages }, mergeConfig);
  } catch (error) {
    console.error("Error merging boundary data:", error);
    
    // 出错时返回未合并的原始数据
    console.log("Falling back to unmerged data");
    return {
      messages: [{ 
        role: "assistant", 
        content: JSON.stringify(boundaryData)
      }]
    };
  }
}

/**
 * 分批处理大量边界数据
 */
async function mergeInBatches(boundaryData: any[]): Promise<any> {
  console.log("Starting batch processing");
  
  // 将数据分成小批次
  const batchSize = 10;
  const batches = [];
  for (let i = 0; i < boundaryData.length; i += batchSize) {
    batches.push(boundaryData.slice(i, i + batchSize));
  }
  
  console.log(`Split into ${batches.length} batches`);
  
  // 处理每个批次
  const mergedBatches = [];
  for (let i = 0; i < batches.length; i++) {
    console.log(`Processing batch ${i+1}/${batches.length}`);
    try {
      // 创建新线程
      const mergeThread = await client.threads.create();
      const mergeConfig = { 
        configurable: { thread_id: mergeThread.thread_id },
        timeout: 60000,
        maxRetries: 2
      };
      
      // 创建消息
      const messages = batches[i].map(data => ({
        role: "user",
        content: JSON.stringify(data)
      }));
      
      // 调用合并代理
      const result = await mergeAgentGraph.invoke({ messages }, mergeConfig);
      
      // 解析结果并添加到合并批次
      if (result.messages && result.messages.length > 0) {
        const lastMessage = result.messages[result.messages.length - 1];
        if (typeof lastMessage.content === 'string') {
          const parsed = JSON.parse(lastMessage.content);
          mergedBatches.push(...(Array.isArray(parsed) ? parsed : [parsed]));
        }
      }
      
      // 添加延迟，避免请求过快
      await new Promise(resolve => setTimeout(resolve, 1000));
    } catch (error) {
      console.error(`Error processing batch ${i+1}:`, error);
      // 将该批次未合并的数据添加到结果
      mergedBatches.push(...batches[i]);
    }
  }
  
  // 返回批次合并结果
  return {
    messages: [{ 
      role: "assistant", 
      content: JSON.stringify(mergedBatches)
    }]
  };
}

/**
 * 执行完整的搜索流程：创建期刊查询 -> MFA搜索 -> 合并结果
 */
export async function searchExpandAndMerge(userQuery: string): Promise<any> {
  try {
    console.log(`Starting search process for: "${userQuery}" with journal specialization`);
    
    // 步骤1: 创建期刊查询
    const journalQueries = createJournalQueries(userQuery);
    console.log(`Created ${journalQueries.length} journal-specific queries`);
    
    // 步骤2: 并行执行MFA搜索
    const queue = new PQueue({ concurrency: 3 }); // 限制并发数
    const searchTasks = journalQueries.map((query: string) => 
      queue.add(() => performMfaSearch(query))
    );
    
    // 等待所有搜索完成
    const searchResults = await Promise.all(searchTasks);
    const successfulSearches = searchResults.filter(r => r.success).length;
    console.log(`Completed ${successfulSearches}/${searchResults.length} searches successfully`);
    
    // 步骤3: 收集所有边界数据
    const allBoundaryData: any[] = [];
    searchResults.forEach(result => {
      if (result.success && result.result) {
        const boundaryItems = extractBoundaryData(result.result);
        console.log(`Found ${boundaryItems.length} boundary items for query: "${result.query}"`);
        allBoundaryData.push(...boundaryItems);
      }
    });
    
    console.log(`Total boundary data items collected: ${allBoundaryData.length}`);
    
    // 保存原始边界数据
    const outputDir = path.join(__dirname, '../../outputs');
    if (!fs.existsSync(outputDir)) {
      fs.mkdirSync(outputDir, { recursive: true });
    }
    
    const safeQuery = userQuery.replace(/[^a-zA-Z0-9\u4e00-\u9fa5]/g, '_').substring(0, 20);
    const timestamp = new Date().toISOString().replace(/[:.]/g, '-');
    const boundaryDataPath = path.join(outputDir, `boundary_data_${safeQuery}_${timestamp}.json`);
    fs.writeFileSync(boundaryDataPath, JSON.stringify(allBoundaryData, null, 2));
    console.log(`Raw boundary data saved to: ${boundaryDataPath}`);
    
    if (allBoundaryData.length === 0) {
      return {
        originalQuery: userQuery,
        journalQueries: journalQueries,
        searchCount: searchResults.length,
        successfulSearches,
        error: "No boundary data found in search results"
      };
    }
    
    // 步骤4: 合并边界数据
    const mergeResult = await mergeBoundaryData(allBoundaryData);
    
    // 步骤5: 提取并保存合并结果
    let parsedMergeResult = null;
    if (mergeResult.messages && mergeResult.messages.length > 0) {
      const lastMessage = mergeResult.messages[mergeResult.messages.length - 1];
      if (typeof lastMessage.content === 'string') {
        parsedMergeResult = JSON.parse(lastMessage.content);
      }
    }
    
    // 保存合并结果
    if (parsedMergeResult) {
      const mergedResultPath = path.join(outputDir, `merged_results_${safeQuery}_${timestamp}.json`);
      fs.writeFileSync(mergedResultPath, JSON.stringify(parsedMergeResult, null, 2));
      console.log(`Merged results saved to: ${mergedResultPath}`);
    }
    
    // 步骤6: 返回完整结果
    return {
      originalQuery: userQuery,
      journalQueries: journalQueries,
      searchCount: searchResults.length,
      successfulSearches,
      boundaryItemsCount: allBoundaryData.length,
      mergedResults: mergeResult,
      parsedResults: parsedMergeResult
    };
  } catch (error: unknown) {
    console.error("Error in search and merge process:", error);
    return {
      originalQuery: userQuery,
      error: error instanceof Error ? error.message : "Unknown error"
    };
  }
}

// 如果直接运行此脚本，处理命令行参数
if (require.main === module) {
  const query = process.argv[2];
  if (!query) {
    console.error("请提供一个查询作为命令行参数");
    process.exit(1);
  }
  
  console.log("开始执行完整搜索流程...");
  searchExpandAndMerge(query)
    .then(results => {
      console.log("\n===== 结果摘要 =====");
      console.log(`原始查询: "${results.originalQuery}"`);
      
      if (results.error) {
        console.log(`错误: ${results.error}`);
      } else {
        console.log(`基于 10 个期刊创建的查询`);
        console.log(`成功搜索: ${results.successfulSearches}/${results.searchCount}`);
        console.log(`收集的边界项: ${results.boundaryItemsCount}`);
        
        if (results.parsedResults) {
          console.log(`\n合并后共 ${results.parsedResults.length} 个结果项`);
        } else {
          console.log("\n合并结果解析失败");
        }
      }
      
      console.log("\nJSON结果已保存到outputs目录");
    })
    .catch(error => {
      console.error("处理过程中出现错误:", error);
    });
}