import type { SerializedTextureBakeBatchRunInput } from './textureBakeSerialization';
import type {
  TextureBakeBatch,
  TextureBakeBatchProgress,
  TextureBakeBatchResult,
} from '../texture/bakeBatch';

export type TextureBakeWorkerRequest =
  | { type: 'init'; id: string; input: SerializedTextureBakeBatchRunInput }
  | { type: 'run-batches'; id: string; batches: TextureBakeBatch[] }
  | { type: 'dispose'; id: string };

export type TextureBakeWorkerResponse =
  | { type: 'ready'; id: string }
  | { type: 'batch-result'; id: string; result: TextureBakeBatchResult }
  | { type: 'batch-progress'; id: string; progress: TextureBakeBatchProgress }
  | { type: 'batches-complete'; id: string }
  | { type: 'disposed'; id: string }
  | { type: 'error'; id: string; message: string; stack?: string };
