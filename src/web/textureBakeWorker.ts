import { runTextureBakeBatchGroup, type TextureBakeBatchRunInput } from '../texture/bakeBatch';
import { deserializeTextureBakeBatchRunInput } from '../pipeline/textureBakeSerialization';
import type { TextureBakeWorkerRequest, TextureBakeWorkerResponse } from '../pipeline/textureBakeWorkerProtocol';
import { collectTransferables } from './serialization';

type WorkerLikeScope = {
  postMessage(message: unknown, transfer?: Transferable[]): void;
  onmessage: ((event: MessageEvent<TextureBakeWorkerRequest>) => void) | null;
};

const workerSelf = self as unknown as WorkerLikeScope;
let input: TextureBakeBatchRunInput | null = null;

function post(message: TextureBakeWorkerResponse): void {
  workerSelf.postMessage(message, collectTransferables(message));
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

workerSelf.onmessage = (event: MessageEvent<TextureBakeWorkerRequest>) => {
  const request = event.data;
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
};

export {};
