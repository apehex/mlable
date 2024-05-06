# TODO

## Common

- [x] replicate Keras modules in base Tensorflow
    - [x] initializers
    - [x] layers
    - [x] losses
    - [x] optimizers
- [x] inputs
    - [x] tokenizer operate directly on tensors instead of looping => *no*, make tokenization framework agnostic
- [x] sampling
- [x] pyTorch variants:
    - [x] MLP
    - [x] CNN
    - [x] SAT

## GPT

- [x] positional embedding:
    - [x] normalize? *no*
- [x] self attention layer:
    - [x] key
    - [x] query
    - [x] value
    - [x] masking + softmax
- [x] residual block:
    - [x] layer norm
    - [x] self attention
    - [x] residual

## Tokenizers

- [x] ngrams
- [x] BPE
    - [x] basic merging
    - [ ] split the data into chunks of similar (uni)codes that can be merged
- [ ] autoencoder:
    - [ ] from UTF-8?
    - [ ] avoid the overlap between successive tokens (shared information)
    - [ ] dimensionality reduction
    - [ ] losses:
        - [ ] synonyms are close
        - [ ] variations are close (plural, upper / lower case, tense)
- [ ] GAN? colony of models?
- [ ] other:
    - [ ] keep track of the token's parts / parents? => enforce the similarity
    - [ ] density: encoding that actually uses the full range of uint256 / float32
- [ ] solidity:
    - [ ] specific tokens for solidity assembly
    - [ ] specific tokens for solidity source code
- [ ] view tokenization results like tiktokensizer
- [ ] properties / loss functions:
    - [ ] embedding space:
        - [ ] the token embedding is dense (contrary to one-hot encoding)
        - [ ] split & merged tokens are close to their siblings
        - [ ] synonyms are close
        - [ ] the loss for numbers varies with delta
- [ ] tokens:
    - [ ] numbers are represented as a single token:
        - [ ] integers
        - [ ] hex numbers (cute into chunks of 32 bytes = 256 bits)
        - [ ] floats in any format (scientific etc)
        - [ ] represented as a vector of length 256
    - [ ] keywords are single tokens
    - [ ] variables are special tokens ?

## TokUn

### Objectives

- [x] dense embeddings, rather than sparse "one-hot"
- [x] guiding without fixing: no frozen dictionary, context agnostic
- [x] tokenization independant of the input partitioning / shift
- [x] dense encoding != one-hot vectors on the vocabulary
- [x] composite tokens have parent / child relation: "splitting" carries the information of "split" and "ing"
- [x] reduce token dimension: from several 100k to 256!

### Dataviz

- [x] spatial repartition of tokens
- [ ] embeddings of child <=> parent tokens
- [x] limit embedding size = when fidelity starts to drop = max compression (64 UTF-32 bytes?)

### Curriculum

- [ ] shift training data by 1, 2, ..., G - 1 ticks along the time / context axis
- [ ] switch between equivalent formats:
    - [x] byte shift
    - [ ] abbreviations: "can't" <=> "cannot"
    - [ ] change number format (while keeping the same value)
- [ ] random perturbations on the inputs:
    - [ ] letter capitalization
    - [ ] byte replacement
    - [ ] byte insertion
    - [ ] reversing order in groups?
- [ ] equivalence 1 <=> 4 <=> 4 x 4:
    - [ ] pad data with 0s to fill bigger tokens until they match their parts

### Issues

Trying to solve:

- [x] variable length encoding (UTF-8) that screws fixed size model shapes
- [x] part / global unrelated: knowledge about tokens doesn't transfer to their siblings
- [x] better support for eastern languages

### Blocks

- [x] tokenization:
    - [x] simplify: divide + position + merge = reshape + dense (position = dense bias on the merged vector)
    - [x] move data from axis 0 to axis -1 in the end: (B * G, E) => (B, G * E)
- [x] detokenization
    - [x] simplify: same as the tokenization block
- [x] head

### Models

- [x] VAE
- [x] VAE + CNN
- [x] VAE + CNN + attention
- [x] VAE + hierarchical CNN
- [x] VAE + hierarchical CNN + attention
- [x] VAE + hierarchical CNN + attention + normalization
