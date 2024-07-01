# CodeAlignKGQA
Artefacts related to the CodeAlignKGQA paper. 

# KoPL Logical Forms Generation and Execution

This repository contains code and scripts for generating and executing KoPL programs. It is organized into the following directories:

## Directories

### 1. Code generation
This directory contains code for generating KoPL logical forms. It has three sub-directories:
- **kqapro**: Contains scripts for the KQA Pro dataset.
- **webqsp**: Contains scripts for the WebQSP dataset.
- **metaqa**: Contains scripts for the MetaQA dataset.

#### KQA Pro Dataset

**Manual Setting:**
To generate KoPL programs in the manual setting, run:
```bash
cd Code generation/kqapro/
python3 kqapro_manual_gemini.py
```

**Dynamic Setting:**
To generate KoPL programs in the dynamic setting, run:
```bash
cd Code generation/kqapro/Dynamic self code correction/
python3 kqapro_dynamic_gemini.py
```

Similarly, programs for webqsp and metaqa can be generated. 


### 2. Executor
This directory contains scripts to execute KoPL programs.

#### KQA Pro Dataset

Check Syntactic Accuracy:
```bash
python -m Executor.executor_rule_kqapro 'val' > 'log_filepath'
```

Execute KoPL Programs:
```bash
python -m Executor.executor_rule_kqapro 'exec' > 'log_filepath'
```

#### WebQSP Dataset
Check Syntactic Accuracy:
```bash
python3 -m Executor.executor_rule_webQSP 'generated_KoPL_filepath' 'ground_KoPL_filepath' 'val' > 'log_filepath'
```

Execute KoPL Programs:
```bash
python3 -m Executor.executor_rule_webQSP 'generated_KoPL_filepath' 'ground_KoPL_filepath' 'exec' > 'log_filepath'
```

#### MetaQA Dataset
Follow similar commands as for the WebQSP dataset.

### 3. Question specific facts
This directory contains scripts to retrieve question-specific facts.

#### KQA Pro Dataset
To retrieve facts, run:
```bash
python3 Retrieval_kqapro.py threshold hop > output_filepath
```
*threshold* value can be between 0 to 1.
*hop* is a natural number.

Similarly, you can retrieve facts for the MetaQA and WebQSP datasets.

### 4. Data
This folder contains input data files for three datasets: KQAPro, MetaQA, WebQSP.


## Package Dependencies

Ensure you have the following packages installed:
```bash
pytorch: 1.11.0
transformers: 4.28.1
sentence-transformers: 2.2.2
guidance: 0.0.64
```
