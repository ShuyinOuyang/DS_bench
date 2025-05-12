# DS-bench: Code Generation Benchmark for Data Science Code

##  :round_pushpin: Abstract
We introduce DS-bench, a new benchmark designed to evaluate large language models (LLMs) on complicated data science code generation tasks.
Existing benchmarks, such as DS-1000, often consist of overly simple code snippets, imprecise problem descriptions, and inadequate testing.
DS-bench sources 1,000 realistic problems from GitHub across ten widely used Python data science libraries, offering richer problem descriptions, substantially longer code solutions, and a customizable test suite with 200 test cases per problem by default.
To construct the DS-bench, we develop a two-stage pipeline that combines automatic code selection, test case generation, and problem description synthesis with rigorous manual polishing to ensure alignment, prevent data leakage, and enhance evaluation reliability.
Experimental results demonstrate that DS-bench is substantially more challenging than existing benchmarks such as DS-1000, consistently yielding lower pass@k scores and exhibiting reduced evaluation variance.
Furthermore, DS-bench exhibits robust scaling behavior, where larger models systematically outperform smaller ones, validating its ability to distinguish model capabilities meaningfully.
We believe DS-bench will serve as a rigorous and trustworthy foundation for advancing LLM-based data science programming.

## :rocket: Updates
**05/12/2025:** Code released

## Installation

```

```

## Seed code
[DS-1000](https://github.com/xlang-ai/DS-1000) 
You can download the DS-1000 dataset from [huggingface](https://huggingface.co/datasets/xlangai/DS-1000).
For the details of this dataset, you can see their [github page](https://github.com/xlang-ai/DS-1000)

