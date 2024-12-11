# Interpretable SAE Feature Extraction
CS 7643 Deep Learning at Georgia Institute of Technology

## Project Overview
This project explores the use of Sparse Autoencoders (SAEs) to extract interpretable features from language models, specifically TinyStories-33M and Llama-2-3B-Instruct. By leveraging the TransformerLens and SAE-Lens libraries, we train SAEs to identify meaningful patterns in the MLP layers of these models.

## Models Analyzed
- **TinyStories-33M**: A small language model trained on simple stories
- **Llama-2-3B-Instruct**: A larger instruction-tuned language model

## Tools Used
- **TransformerLens**: For accessing and analyzing transformer model internals
- **SAE-Lens**: For training and visualizing sparse autoencoders
- **PyTorch**: As the underlying deep learning framework

## Methodology
1. Used TransformerLens to load and access the internal activations of both models
2. Trained sparse autoencoders on the MLP layer outputs using SAE-Lens
3. Analyzed the learned features using visualization tools provided by SAE-Lens
4. Documented interpretable features found in different model layers

## Key Findings
### TinyStories-33M Features
- Layer 0 discovered features related to:
  - Arts and crafts materials
  - Speaking verbs (said, cried, shouted)
  - Possessive adjectives (my, your, his, her)
  - Time-related words
  - Character names
  
[Additional findings to be added]

### Llama-2-3B Features
[Your findings for Llama-2 to be added]

## Repository Structure
- `notebooks/`: Jupyter notebooks containing analysis
- `visualizations/`: Generated HTML visualizations of features
- [Additional structure to be added]

## Setup and Usage
[Instructions for setting up and running your code]

## Acknowledgments
This project builds upon the work and tools provided by:
- [TransformerLens](https://github.com/neelnanda-io/TransformerLens)
- [SAE-Lens](https://github.com/ArthurConmy/sae-lens) 
