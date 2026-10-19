"""Microbatch sizes that preserve effective batch 64 with exact accumulation."""
MICROBATCHES = (64, 32, 16, 8, 4, 2, 1)


def accumulation_for(batch_size: int) -> int:
    if batch_size not in MICROBATCHES:
        raise ValueError('Microbatch must be one of 64, 32, 16, 8, 4, 2, 1')
    return 64 // batch_size
