# Introduction
In this post, we will go over developing a character level language model that will generate shakespeare like text.

The main way this differs from standard LLM's that you may interact with daily is that we will process one character after each time step while the current standard is to process subword tokens.

Put simply, subword tokenization (e.g. Byte Pair Encoding or Wordpiece) is when text is broken down into meaningful chunks that are larger than individual characters but smaller than words. 



# Data
