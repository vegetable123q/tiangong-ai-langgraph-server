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
const extractAgentGraphName = 'extract_agent';
const mergeAgentGraphName = 'merge_agent';

// 边界项接口定义
interface BoundaryItem {
  source: string;
  SpatialScope: string;
  timeRange: string;
  policyRecommendations: string[];
  spatialTag?: string;
}

// 初始化客户端和远程图
const client = new Client({ apiUrl: url, apiKey: apiKey });
const mfaSearchGraph = new RemoteGraph({ 
  graphId: mfaSearchGraphName, 
  url: url, 
  apiKey: apiKey 
});
const extractAgentGraph = new RemoteGraph({
  graphId: extractAgentGraphName,
  url: url,
  apiKey: apiKey
});
const mergeAgentGraph = new RemoteGraph({
  graphId: mergeAgentGraphName,
  url: url,
  apiKey: apiKey
});

/**
 * 执行MFA搜索获取内容组
 */
async function performSearchAndGetContentGroups(query: string): Promise<string[]> {
  try {
    console.log(`Executing search for: "${query}"`);
    
    // 创建新线程
    const thread = await client.threads.create();
    const config = { 
      configurable: { thread_id: thread.thread_id },
      recursionLimit: 50
    };
    
    // 调用MFA搜索代理
    const result = await mfaSearchGraph.invoke({ 
      messages: [{ role: "user", content: query }] 
    }, config);
    
    console.log('Search completed successfully');
    
    // 提取内容批次
    const contentGroups: string[] = [];
    
    // 尝试从不同位置提取内容组
    if (result.contentBatches && Array.isArray(result.contentBatches)) {
      // 处理一维或二维数组
      if (Array.isArray(result.contentBatches[0])) {
        // 二维数组的情况
        result.contentBatches.forEach((batch: string[]) => {
          contentGroups.push(...batch);
        });
      } else {
        // 一维数组的情况
        contentGroups.push(...result.contentBatches);
      }
      console.log(`Found ${contentGroups.length} content groups directly in result`);
    } else if (result.messages && Array.isArray(result.messages)) {
      // 尝试从消息中提取
      for (const message of result.messages) {
        // 检查additional_kwargs
        if (message.additional_kwargs?.contentBatches) {
          const batches = message.additional_kwargs.contentBatches;
          if (Array.isArray(batches)) {
            if (Array.isArray(batches[0])) {
              batches.forEach((batch: string[]) => {
                contentGroups.push(...batch);
              });
            } else {
              contentGroups.push(...batches);
            }
            console.log(`Found ${contentGroups.length} content groups in message additional_kwargs`);
          }
        }
      }
    }
    
    console.log(`Total content groups extracted: ${contentGroups.length}`);
    return contentGroups;
  } catch (error) {
    console.error('Error during MFA search:', error);
    throw new Error(`Search failed: ${error instanceof Error ? error.message : 'Unknown error'}`);
  }
}

/**
 * 对单一内容组使用extract_agent提取边界信息
 */
async function extractBoundaryFromContent(content: string, query: string): Promise<BoundaryItem | null> {
  try {
    console.log(`Extracting boundary from content (preview): ${content.substring(0, 100)}...`);
    
    // 调用提取代理
    const extractResult = await extractAgentGraph.invoke({
      content: content,
      query: query
    });
    
    // 检查提取结果
    if (extractResult.isRelevant && extractResult.boundaryItem) {
      console.log('Content is relevant, boundary extracted successfully');
      console.log(`Evaluation score: ${extractResult.evaluationResult?.score?.toFixed(2) || 'N/A'}`);
      console.log(`Cycles completed: ${extractResult.cycleCount || 1}`);
      return extractResult.boundaryItem;
    } else {
      console.log(`Content is ${extractResult.isRelevant ? 'relevant but extraction failed' : 'not relevant'}`);
      return null;
    }
  } catch (error) {
    console.error('Error extracting boundary:', error);
    return null;
  }
}

/**
 * 合并一批边界项
 */
async function mergeBoundaryBatch(boundaryBatch: BoundaryItem[]): Promise<BoundaryItem[]> {
  try {
    console.log(`Merging batch of ${boundaryBatch.length} boundary items`);
    
    // 创建新线程
    const thread = await client.threads.create();
    const config = { 
      configurable: { thread_id: thread.thread_id },
      recursionLimit: 10
    };
    
    // 准备消息
    const messages = boundaryBatch.map(item => ({
      role: "user", 
      content: JSON.stringify(item)
    }));
    
    // 调用合并代理
    const result = await mergeAgentGraph.invoke({ messages }, config);
    
    // 解析结果
    let mergedItems: BoundaryItem[] = [];
    if (result.messages && result.messages.length > 0) {
      const lastMessage = result.messages[result.messages.length - 1];
      if (typeof lastMessage.content === 'string') {
        try {
          const parsedContent = JSON.parse(lastMessage.content);
          if (Array.isArray(parsedContent)) {
            mergedItems = parsedContent;
          } else {
            mergedItems = [parsedContent];
          }
        } catch (e) {
          console.error('Failed to parse merge agent response:', e);
        }
      }
    }
    
    console.log(`Merged batch resulted in ${mergedItems.length} items`);
    return mergedItems;
  } catch (error) {
    console.error('Error merging boundary batch:', error);
    return boundaryBatch; // 失败时返回原始批次
  }
}

/**
 * 执行完整的集成工作流
 */
export async function runIntegratedWorkflow(userQuery: string): Promise<any> {
  try {
    // 创建会话标识和时间戳
    const sessionId = uuidv4().substring(0, 8);
    const timestamp = new Date().toISOString().replace(/[:.]/g, '-');
    const safeQuery = userQuery.replace(/[^a-zA-Z0-9\u4e00-\u9fa5]/g, '_').substring(0, 20);
    
    // 创建输出目录
    const outputDir = path.join(__dirname, '../../outputs');
    if (!fs.existsSync(outputDir)) {
      fs.mkdirSync(outputDir, { recursive: true });
    }
    
    console.log(`==========================================`);
    console.log(`Starting integrated workflow for query: "${userQuery}"`);
    console.log(`Session ID: ${sessionId}`);
    console.log(`==========================================`);
    
    // 步骤1: 搜索并获取内容组
    console.log(`\n====== STEP 1: MFA SEARCH ======`);
    const contentGroups = await performSearchAndGetContentGroups(userQuery);
    
    if (contentGroups.length === 0) {
      console.log('No content groups found, workflow terminated');
      return {
        success: false,
        query: userQuery,
        error: 'No content found from search'
      };
    }
    
    // 保存搜索结果
    const searchResultsPath = path.join(outputDir, `search_results_${safeQuery}_${sessionId}.json`);
    fs.writeFileSync(searchResultsPath, JSON.stringify(contentGroups, null, 2));
    console.log(`Search results saved to: ${searchResultsPath}`);
    
    // 步骤2: 对每个内容组提取边界信息
    console.log(`\n====== STEP 2: BOUNDARY EXTRACTION ======`);
    console.log(`Processing ${contentGroups.length} content groups for extraction`);
    
    // 并行处理内容组，限制并发数
    const extractionPromises: Promise<BoundaryItem | null>[] = [];
    const maxConcurrent = 4;
    let activePromises = 0;
    let index = 0;
    
    while (index < contentGroups.length || activePromises > 0) {
      if (index < contentGroups.length && activePromises < maxConcurrent) {
        // 启动新的提取任务
        console.log(`Starting extraction for content ${index + 1}/${contentGroups.length}`);
        const contentGroup = contentGroups[index];
        
        const promise = extractBoundaryFromContent(contentGroup, userQuery)
          .then(result => {
            activePromises--;
            return result;
          })
          .catch(error => {
            console.error(`Error in extraction promise ${index + 1}:`, error);
            activePromises--;
            return null;
          });
        
        extractionPromises.push(promise);
        activePromises++;
        index++;
      } else {
        // 等待一个任务完成
        await new Promise(resolve => setTimeout(resolve, 100));
      }
    }
    
    // 等待所有提取任务完成
    const allExtractedResults = await Promise.all(extractionPromises);
    
    // 过滤有效结果
    const validBoundaryItems = allExtractedResults.filter(item => item !== null) as BoundaryItem[];
    console.log(`\nExtracted ${validBoundaryItems.length} valid boundary items out of ${contentGroups.length} content groups`);
    
    if (validBoundaryItems.length === 0) {
      console.log('No valid boundary items extracted, workflow terminated');
      return {
        success: false,
        query: userQuery,
        error: 'No valid boundary information could be extracted'
      };
    }
    
    // 保存提取结果
    const extractionResultsPath = path.join(outputDir, `extraction_results_${safeQuery}_${sessionId}.json`);
    fs.writeFileSync(extractionResultsPath, JSON.stringify(validBoundaryItems, null, 2));
    console.log(`Extraction results saved to: ${extractionResultsPath}`);
    
    // 步骤3: 分批处理边界项进行合并
    console.log(`\n====== STEP 3: BATCH MERGING ======`);
    
    // 将边界项分成每批5个的批次
    const batchSize = 5;
    const batches: BoundaryItem[][] = [];
    
    for (let i = 0; i < validBoundaryItems.length; i += batchSize) {
      batches.push(validBoundaryItems.slice(i, i + batchSize));
    }
    
    console.log(`Split ${validBoundaryItems.length} boundary items into ${batches.length} batches (batch size: ${batchSize})`);
    
    // 并行处理每个批次，限制并发数
    const mergePromises: Promise<BoundaryItem[]>[] = [];
    let activeMergePromises = 0;
    let batchIndex = 0;
    
    while (batchIndex < batches.length || activeMergePromises > 0) {
      if (batchIndex < batches.length && activeMergePromises < maxConcurrent) {
        // 启动新的合并任务
        console.log(`Starting merge for batch ${batchIndex + 1}/${batches.length}`);
        const batch = batches[batchIndex];
        
        const promise = mergeBoundaryBatch(batch)
          .then(result => {
            activeMergePromises--;
            return result;
          })
          .catch(error => {
            console.error(`Error in merge promise ${batchIndex + 1}:`, error);
            activeMergePromises--;
            return batch; // 错误时返回原始批次
          });
        
        mergePromises.push(promise);
        activeMergePromises++;
        batchIndex++;
      } else {
        // 等待一个任务完成
        await new Promise(resolve => setTimeout(resolve, 100));
      }
    }
    
    // 等待所有合并任务完成
    const allMergedBatches = await Promise.all(mergePromises);
    
    // 收集所有合并结果
    const allMergedItems: BoundaryItem[] = [];
    allMergedBatches.forEach(batch => {
      allMergedItems.push(...batch);
    });
    
    console.log(`\nMerged into a total of ${allMergedItems.length} boundary items`);
    
    // 步骤4: 最终去重和结果整理
    console.log(`\n====== STEP 4: FINAL DEDUPLICATION ======`);
    
    // 按source进行最终去重
    const finalMap = new Map<string, BoundaryItem>();
    allMergedItems.forEach(item => {
      if (!finalMap.has(item.source)) {
        finalMap.set(item.source, item);
      }
    });
    
    const finalResults = Array.from(finalMap.values());
    console.log(`Final deduplication: ${allMergedItems.length} → ${finalResults.length} unique items`);
    
    // 按空间标签分组
    type GroupedResults = {
      city: BoundaryItem[];
      province: BoundaryItem[];
      national: BoundaryItem[];
      focus: BoundaryItem[];
      untagged: BoundaryItem[];
    };
    
    const groupedResults: GroupedResults = {
      city: [],
      province: [],
      national: [],
      focus: [],
      untagged: []
    };
    
    finalResults.forEach(item => {
      if (!item.spatialTag) {
        groupedResults.untagged.push(item);
      } else if (Object.keys(groupedResults).includes(item.spatialTag)) {
        groupedResults[item.spatialTag as keyof GroupedResults].push(item);
      } else {
        groupedResults.untagged.push(item);
      }
    });
    
    // 保存最终结果
    const finalResultsPath = path.join(outputDir, `final_results_${safeQuery}_${sessionId}.json`);
    fs.writeFileSync(finalResultsPath, JSON.stringify(finalResults, null, 2));
    
    // 保存分组结果
    const groupedResultsPath = path.join(outputDir, `grouped_results_${safeQuery}_${sessionId}.json`);
    fs.writeFileSync(groupedResultsPath, JSON.stringify(groupedResults, null, 2));
    
    console.log(`\nFinal results saved to: ${finalResultsPath}`);
    console.log(`Grouped results saved to: ${groupedResultsPath}`);
    
    // 简化返回，只包含分组后的结果
    return {
      success: true,
      query: userQuery,
      groupedResults: groupedResults
    };
  } catch (error) {
    console.error('Error in integrated workflow:', error);
    return {
      success: false,
      query: userQuery,
      error: error instanceof Error ? error.message : 'Unknown error in workflow'
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
  
  console.log("开始执行集成工作流...");
  runIntegratedWorkflow(query)
    .then(results => {
      if (!results.success) {
        console.log(`\n处理失败: ${results.error}`);
        return;
      }
      
      console.log(`\n===== 分组结果 =====`);
      
      // 输出城市级别结果
      if (results.groupedResults.city.length > 0) {
        console.log(`\n城市级别 (${results.groupedResults.city.length}项):`);
        results.groupedResults.city.forEach((item: BoundaryItem, index:number) => {
          console.log(`[${index + 1}] ${item.source.substring(0, 80)}...`);
          console.log(`    空间范围: ${item.SpatialScope}`);
          console.log(`    时间范围: ${item.timeRange}`);
          console.log(`    政策建议数: ${item.policyRecommendations.length}`);
        });
      }
      
      // 输出省级结果
      if (results.groupedResults.province.length > 0) {
        console.log(`\n省级 (${results.groupedResults.province.length}项):`);
        results.groupedResults.province.forEach((item: BoundaryItem, index:number) => {
          console.log(`[${index + 1}] ${item.source.substring(0, 80)}...`);
          console.log(`    空间范围: ${item.SpatialScope}`);
          console.log(`    时间范围: ${item.timeRange}`);
          console.log(`    政策建议数: ${item.policyRecommendations.length}`);
        });
      }
      
      // 输出国家级别结果
      if (results.groupedResults.national.length > 0) {
        console.log(`\n国家级别 (${results.groupedResults.national.length}项):`);
        results.groupedResults.national.forEach((item: BoundaryItem, index:number) => {
          console.log(`[${index + 1}] ${item.source.substring(0, 80)}...`);
          console.log(`    空间范围: ${item.SpatialScope}`);
          console.log(`    时间范围: ${item.timeRange}`);
          console.log(`    政策建议数: ${item.policyRecommendations.length}`);
        });
      }
      
      // 输出局部焦点结果
      if (results.groupedResults.focus.length > 0) {
        console.log(`\n局部焦点 (${results.groupedResults.focus.length}项):`);
        results.groupedResults.focus.forEach((item: BoundaryItem, index:number) => {
          console.log(`[${index + 1}] ${item.source.substring(0, 80)}...`);
          console.log(`    空间范围: ${item.SpatialScope}`);
          console.log(`    时间范围: ${item.timeRange}`);
          console.log(`    政策建议数: ${item.policyRecommendations.length}`);
        });
      }
      
      // 输出未分类结果
      if (results.groupedResults.untagged.length > 0) {
        console.log(`\n未分类 (${results.groupedResults.untagged.length}项):`);
        results.groupedResults.untagged.forEach((item: BoundaryItem, index:number) => {
          console.log(`[${index + 1}] ${item.source.substring(0, 80)}...`);
          console.log(`    空间范围: ${item.SpatialScope}`);
          console.log(`    时间范围: ${item.timeRange}`);
          console.log(`    政策建议数: ${item.policyRecommendations.length}`);
        });
      }
      
      console.log("\n详细结果已保存到outputs目录");
    })
    .catch(error => {
      console.error("处理过程中出现错误:", error);
    });
}