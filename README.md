# Software Identity Disambiguation Benchmark

This repository provides a benchmark for evaluating the performance of language models in resolving identity conflicts in research software metadata. It includes gold standard annotations, evaluation scripts, and visualizations comparing automated and human disambiguation.

The benchmark is part of the Software Observatory project and supports the development of robust, FAIR-aligned software metadata integration pipelines.

> **Note:** This benchmark is under active development. The code and results may change as new models, prompts, and evaluation methods are added. For reproducible use, refer to a tagged release.

## Overview

- `data/` contains grouped metadata entries generated from the Software Observatory's ETL pipeline. These entries are used to compose the input messages for identity resolution.
- `src/` contains the core logic for preparing messages and performing inference, adapted from the production ETL pipeline.
- `scripts/` includes model-specific wrappers to trigger message preparation, run inference with different providers, and evaluate results.
- `figures/` contains visualizations such as confusion matrices, performance comparisons, and timing breakdowns.

## Environment Variables

To run the benchmark, a `.env` file is required in the root of the repository. It should define the following variables:

```bash
# Required for metadata and content extraction
GITHUB_TOKEN=...
GITLAB_TOKEN=...

# Required for model inference
OPENROUTER_API_KEY=...
HUGGINGFACE_API_KEY=...
```

This file **should not** be committed to the repository.

## License

This project is licensed under the Apache License 2.0.


## Contact

For questions or contributions, feel free to open an issue or contact the maintainers.