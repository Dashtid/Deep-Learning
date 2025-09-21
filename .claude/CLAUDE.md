# Deep Learning for Medical Image Segmentation - Claude Code Guidelines

## Project Overview

This project explores the quantification of differences in expert segmentations from CT and MRI images using deep learning, based on the QUBIQ Challenge. It implements U-Net architectures for medical image segmentation across multiple datasets including brain-growth, brain-tumor, kidney, and prostate imaging tasks.

## Development Environment

**Operating System**: Windows 11
**Shell**: Git Bash / PowerShell / Command Prompt
**Important**: Always use Windows-compatible commands:
- Use `dir` instead of `ls` for Command Prompt
- Use PowerShell commands when appropriate
- File paths use backslashes (`\`) in Windows
- Use `python -m http.server` for local development server
- Git Bash provides Unix-like commands but context should be Windows-aware

## Development Guidelines

### Code Quality
- Follow Python PEP 8 style guidelines
- Use meaningful variable and function names
- Implement proper error handling and logging
- Add comprehensive docstrings for functions and classes
- Use type hints where appropriate
- Maintain clean, readable code
- Follow language-specific best practices

### Security
- No sensitive information in the codebase
- Use HTTPS for all external resources
- Regular dependency updates
- Follow security best practices for the specific technology stack

### Machine Learning Specific Guidelines
- Document model architectures and hyperparameters clearly
- Implement proper data preprocessing and augmentation
- Use appropriate evaluation metrics (DICE score, etc.)
- Ensure reproducibility with random seeds
- Implement proper train/validation/test splits
- Document experimental results and findings

## Learning and Communication
- Always explain coding actions and decisions to help the user learn
- Describe why specific approaches or technologies are chosen
- Explain the purpose and functionality of code changes
- Provide context about best practices and coding patterns used
- Provide detailed explanations in the console when performing tasks, as many concepts may be new to the user