# How to develop applications with `LangChain`? 

Author: Zechang Sun

Date: May 23, 2023

for Data Science Club in DoA, Tsinghua

Ref: https://python.langchain.com/en/latest

## What is `LangChain`?

[`LangChain`](https://python.langchain.com/en/latest/) is a framework for developing applications powered by language models. We believe that the most powerful and differentiated applications will not only call out to a language model, but will also be:

1. Data-aware: connect a language model to other sources of data;
2. Agents: allow a language model to interact with its environment.

## Key Components of  `LangChain`

* `Models` 

  * `LLMs`: large language models, which take a text string as input, and return a text string as output;
  * `Chat Models`: usually backed by a language model, but their APIs are more structured, these models take a list of Chat Messages as input, and return a Chat Message;
  * `Text Embedding Models`: take text as input and return a list of floats.
* `Prompts`

  * `PromptValue`: an input to a model;

  * `Promapt Templates`: in charge of constructing a PromptValue;

  * `Example Selectors`: hardcoded or dynamically selected examples in prompts;

  * `Output Parsers`: get more structured information than just text back, instruct the model how output should be formatted and parse output into the desired format if necessary.
* `Indexes`: how to structure documents so that LLMs can best interact with them
  * `Document Loaders`:  classes responsible for loading documents from various sources;
  * `Text Splitters`: classes responsible for splitting text into smaller chunks;
  * `VectorStores`: the most common type of index, which relies on embeddings;
  * `Retrievers`: interface for fetching relevant documents to combine with language models.

* `Memory`: store and retrieve data in the process of a conversation.
  * Two main methods:
    * Based on the input, fetch any relevant pieces of data;
    * Based on the input and output, update state accordingly.

  * Two main types of memory:
    * Short term: how to pass data in the context of a singluar conversion;
    * Long term: how to fetch and update information between conversations.

* `Chain`: an incredibly generic concept which returns to a sequence of modular components (or other chains), combined in a particular way to accomplish a common use case;
* `Agents`: has access to a suite of tools, depending on the user input, the agent can decide which, if any, of these tools to call.

## Key Concept for developing applications 

*  Chain of Thought
  * encourage the model to generate a series of intermediate reasoning steps
  * simply include "Let's think step-by-step" in the prompt
*  Action Plan Generation
  * use a language model to generate actions to take
  * results of these actions can then be fed back into the language model to generate a subsequent action
* ReAct
  * combine Chain-of-Thought prompting with action plan generation
  * induce the model to think about what action to take, then take it
*  Self-ask
  * the model explicitly ask itself follow-up questions, which are then answered by an external search engine
* Prompt Chaining
  * combing multiple LLM calls, with the output of one-step being the input to the next
* Memetic Proxy
  * encourage the LLm to respond in a certain way framing the discussion in a context that the model knows of and that will result in that type of response
* Self Consistency
  * sample a diverse set of reasoning paths and then selects the most consistent answer
  * is most effective when combined with Chain-of-thought prompting
* Inception
  * also called `First Person Instruction`
  * encourage the model to think a certain way by including the start of the model's response in the prompt
* MemPrompt
  * maintains a memory of errors and user feedback
  * use them to prevent repetition of mistakes



