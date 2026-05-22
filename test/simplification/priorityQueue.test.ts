import { describe, expect, it } from 'vitest';
import { CandidateQueue } from '../../src/simplification/priorityQueue';

describe('CandidateQueue', () => {
  it('keeps one live candidate per edge when priorities are updated', () => {
    const queue = new CandidateQueue(4);

    queue.upsert(2, 10, 1, 2, 3);
    queue.upsert(2, 5, 4, 5, 6);
    queue.upsert(1, 7, 7, 8, 9);

    expect(queue.size).toBe(2);
    expect(queue.liveCandidateCount()).toBe(2);

    const first = queue.pop();
    expect(first?.edgeId).toBe(2);
    expect(first?.cost).toBe(5);
    expect(first?.x).toBe(4);
    expect(queue.size).toBe(1);
  });

  it('removes inactive edges without leaving stale heap entries', () => {
    const queue = new CandidateQueue(3);

    queue.upsert(0, 3, 0, 0, 0);
    queue.upsert(1, 1, 1, 1, 1);
    queue.upsert(2, 2, 2, 2, 2);
    queue.remove(1);

    expect(queue.size).toBe(2);
    expect(queue.liveCandidateCount()).toBe(2);
    expect(queue.pop()?.edgeId).toBe(2);
    expect(queue.pop()?.edgeId).toBe(0);
    expect(queue.pop()).toBeUndefined();
  });
});
