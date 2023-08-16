# RL_for_summarization_based_on_faithful_metrics

This repository contains code snippets used in the thesis: "Reinforcement Learning Approaches for Faithful Abstractive Text Summarization".
No unified interface is provided. Some code (and paths to data) might need to be configured or (un)commented to execute specific parts.

The data needed to train and evaluate included in this repo do not belong to me. We thank the respective authors. More information about the sources is provided in the thesis text.

The code utilized to create plot 5.1 and 5.2 is in the file "Plotting_randomFunction_DatasetLengthRatio.R"

## Ensemble pipeline:
* Unique pairs of documents and summaries are collected into a csv file. Code: "ensemble_data_createUniquePairs"
* Score for the metrics used as features are than added into a copy of this file. Code: "MetricScoresOnEnsembleData" and "AddingMetricsToAggrefactDatasheet". (For the later the Aggrefact data was replaced by this. The csv of collected paits, follows the same pattern as the aggrefact.csv so some code was dual used, particularly the docker based stuff.)
* These cach is than used to train the ensemble. Code: "EnsembleTraining"

## Eval on Aggrefact
Aggrefact has a two stage pipeline. Step 1: add metric scores to a csv. Step 2: eval the csv.
Step 1:
* Adding ensemble scores. Code: "EvalEnsemble" (Also used to plot ensemble training information stored in a save of a training run)
* ChatGPT-based metrics were added using the Code in: "AddingChatGPTMetricsToAggrefactDatasheet"
* Score of remaining metrics are added using Code in: "AddingMetricsToAggrefactDatasheet"
Step 2:
Collect all created csv metric score files into one file. Evaluate this file / or files using Code in: "aggrefact_eval.py"

## Caution
A lot of metrics provided from other repositories are utilized in this work. Most need to be installed first. Some are set up using a docker pipeline which handles version issues and setup.
Information about the utilized repositories and version is documented in the thesis.

As this code is the result of rapid prototyping and execution on multiple servers it is not perfectly structured and contains alot of Code duplicates. However, if anybody wants to understand or reuse any part of the Code, feel free to ask for advice. We are happy if this project provides benefit to somebody.