import { parentPort } from 'node:worker_threads';
import { runTextureBakeBatchGroup, type TextureBakeBatchRunInput } from '../texture/bakeBatch';
import { deserializeTextureBakeBatchRunInput } from '../pipeline/textureBakeSerialization';
import type { TextureBakeWorkerRequest, TextureBakeWorkerResponse } from '../pipeline/textureBakeWorkerProtocol';

let input: TextureBakeBatchRunInput | null = null;

function post(message: TextureBakeWorkerResponse): void {
  parentPort?.postMessage(message);
}

function errorMessage(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}

function errorStack(error: unknown): string | undefined {
  return error instanceof Error ? error.stack : undefined;
}

function requireInput(): TextureBakeBatchRunInput {
  if (!input) throw new Error('Texture bake worker has not been initialized.');
  return input;
}

function handleInit(request: Extract<TextureBakeWorkerRequest, { type: 'init' }>): void {
  input = deserializeTextureBakeBatchRunInput(request.input);
  post({ type: 'ready', id: request.id });
}

async function handleRunBatches(request: Extract<TextureBakeWorkerRequest, { type: 'run-batches' }>): Promise<void> {
  const activeInput = requireInput();
  await runTextureBakeBatchGroup(
    { ...activeInput, batches: request.batches },
    (result) => post({ type: 'batch-result', id: request.id, result }),
    (progress) => post({ type: 'batch-progress', id: request.id, progress }),
    { collectResults: false },
  );
  post({ type: 'batches-complete', id: request.id });
}

function handleDispose(request: Extract<TextureBakeWorkerRequest, { type: 'dispose' }>): void {
  input = null;
  post({ type: 'disposed', id: request.id });
}

if (!parentPort) throw new Error('Texture bake worker requires a worker_threads parentPort.');

parentPort.on('message', (request: TextureBakeWorkerRequest) => {
  void (async () => {
    try {
      if (request.type === 'init') handleInit(request);
      else if (request.type === 'run-batches') await handleRunBatches(request);
      else handleDispose(request);
    } catch (error: unknown) {
      const stack = errorStack(error);
      post({
        type: 'error',
        id: request.id,
        message: errorMessage(error),
        ...(stack ? { stack } : {}),
      });
    }
  })();
});
