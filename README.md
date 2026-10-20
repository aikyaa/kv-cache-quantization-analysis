# KV-cache quantization for LLM decode: an analytical study

How much do INT8/INT4 KV caches cut memory traffic during decode attention, and at
what context length and batch size does decode move from memory-bound to
compute-bound? Everything here runs on a laptop CPU with open-source Python.

Work in progress.
