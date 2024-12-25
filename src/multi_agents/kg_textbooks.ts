import { Client } from '@langchain/langgraph-sdk';
import { RemoteGraph } from '@langchain/langgraph/remote';
import dotenv from 'dotenv';
import fs from 'fs';
import PQueue from 'p-queue';

dotenv.config();

const url = process.env.DEPLOYMENT_URL ?? '';
const apiKey = process.env.LANGSMITH_API_KEY ?? '';
const graphName = process.env.GRAPH_NAME ?? '';
const client = new Client({ apiUrl: url, apiKey: apiKey });
const remoteGraph = new RemoteGraph({ graphId: graphName, url: url, apiKey: apiKey });

async function generateKG(inputs: { rec_id: string; title: string }[]) {
  const thread = await client.threads.create();
  const config = { configurable: { thread_id: thread.thread_id } };
  await remoteGraph.invoke({ inputs: inputs }, config);
}

async function main() {
  const data: { rec_id: string; title: string }[] = JSON.parse(
    fs.readFileSync('./src/data/textbooks.json', 'utf-8'),
  );
  const queue = new PQueue({ concurrency: 10 });

  const tasks = data.map((item) => queue.add(() => generateKG([item])));

  await Promise.all(tasks);
  console.info('All tasks completed.');
}

main().catch(console.error);
