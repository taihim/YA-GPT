# Introduction
In this post, we will go over developing a character level language model that will generate shakespeare like text.

The main way this differs from standard LLM's that you may interact with daily is that we will process one character after each time step while the current standard is to process subword tokens (more on this in a bit).

All ML models perform calculations on numbers. Natural language needs to be converted into a numerical form before we can do anything interesting with it.


Put simply, subword tokenization (e.g. Byte Pair Encoding or Wordpiece) is when text is broken down into meaningful chunks that are larger than individual characters but smaller than full words. 
This lets us encode sequences of text in a smaller amount of tokens.

What do we mean by encoding? It simply means converting characters/subwords as a number, since models can only operate on numbers. 



# Data
