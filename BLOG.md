# Introduction
In this post, we will go over developing a character level language model that will generate shakespeare like text.

The main way this differs from standard LLM's that you may interact with daily is that we will process one character after each time step while the current standard is to process subword tokens (more on this in a bit).

# Data
All ML models perform calculations on numbers. Natural language needs to be converted into a numerical form before we can do anything interesting with it.
The process of converting text to a numerical format for processing is called encoding. 


Before we can encode our text, we first need to know all the possible values present in our input data. This collection is commonly called the 'Vocabulary' of the model. 
Just like a person can only properly speak/understand words from their vocabulary, a model can only generate elements that are present in its vocabulary. 

e.g. if our input data is just the text "Hello, world!", our vocabulary would look like ['H', 'e', 'l', 'o', ',', ' ', 'w', 'r', 'd', '!']. Notice how we also take into account the space and punctuation characters.
A model trained on this input would only be able to generate some combination of these characters. 

If we take a look at all the unique characters contained in the Shakespeare dataset we get this:

`['\n', ' ', '!', '$', '&', "'", ',', '-', '.', '3', ':', ';', '?', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']`

So our model has access to the whole alphabet and many punctuation/formatting characters. It can generate quite a lot of different combinations with this larger vocabulary (total length 65).

Now that we have a vocabulary for our model, we can define an encoding scheme to convert the data. We can just map each character to an integer value, starting from 0.
This ends up looking like this:

`{'\n': 0, ' ': 1, '!': 2, '$': 3, '&': 4, "'": 5, ',': 6, '-': 7, '.': 8, '3': 9, ':': 10, ';': 11, '?': 12, 'A': 13, 'B': 14, 'C': 15, 'D': 16, 'E': 17, 'F': 18, 'G': 19, 'H': 20, 'I': 21, 'J': 22, 'K': 23, 'L': 24, 'M': 25, 'N': 26, 'O': 27, 'P': 28, 'Q': 29, 'R': 30, 'S': 31, 'T': 32, 'U': 33, 'V': 34, 'W': 35, 'X': 36, 'Y': 37, 'Z': 38, 'a': 39, 'b': 40, 'c': 41, 'd': 42, 'e': 43, 'f': 44, 'g': 45, 'h': 46, 'i': 47, 'j': 48, 'k': 49, 'l': 50, 'm': 51, 'n': 52, 'o': 53, 'p': 54, 'q': 55, 'r': 56, 's': 57, 't': 58, 'u': 59, 'v': 60, 'w': 61, 'x': 62, 'y': 63, 'z': 64}`

And we can save the reverse mapping to convert back to text: 

`{0: '\n', 1: ' ', 2: '!', 3: '$', 4: '&', 5: "'", 6: ',', 7: '-', 8: '.', 9: '3', 10: ':', 11: ';', 12: '?', 13: 'A', 14: 'B', 15: 'C', 16: 'D', 17: 'E', 18: 'F', 19: 'G', 20: 'H', 21: 'I', 22: 'J', 23: 'K', 24: 'L', 25: 'M', 26: 'N', 27: 'O', 28: 'P', 29: 'Q', 30: 'R', 31: 'S', 32: 'T', 33: 'U', 34: 'V', 35: 'W', 36: 'X', 37: 'Y', 38: 'Z', 39: 'a', 40: 'b', 41: 'c', 42: 'd', 43: 'e', 44: 'f', 45: 'g', 46: 'h', 47: 'i', 48: 'j', 49: 'k', 50: 'l', 51: 'm', 52: 'n', 53: 'o', 54: 'p', 55: 'q', 56: 'r', 57: 's', 58: 't', 59: 'u', 60: 'v', 61: 'w', 62: 'x', 63: 'y', 64: 'z'}`


So, using our new lookup table, we can now encode "Hello, World!" by iterating over the string and mapping each character to its integer representation:

`[20, 43, 50, 50, 53, 6, 2, 36, 54, 57, 51, 43, 3]`

So, with our character level encoding scheme, the encoded sequence has the same length as the input sequence. 
For transformer based models this is an issue because the attention mechanism scales quadratically with sequence length. Therefore it is important that we limit sequence length if we can while preserving the meaning of the text. 

Luckily, there is another tokenization scheme called subword tokenization (e.g. Byte Pair Encoding or Wordpiece) that attempts to address this issue. In this scheme, text is broken down into meaningful chunks that are larger than individual characters but smaller than full words. 
This leads to a larger vocabulary but it lets us encode sequences of text in a smaller amount of tokens.

Modern models like GPT use the Byte Pair Encoding algorithm. GPT4 uses an improved version of BPE called cl100k_base.
This tokenizer is more complicated to implement and the vocabulary is constructed greedily using various rules (approx. 100k vocab size). (Seperate blogpost on BPE) 
OpenAI allows us to use their encoder via a python package called [tiktoken](https://github.com/openai/tiktoken).

When we encode the same with BPE, we get this:

`'Hello, world!' -> [9906, 11, 1917, 0]`

So instead of 13 integers, we encoded it using 4 (almost a 3x reduction in tokens).

