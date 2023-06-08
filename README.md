# DocInterrogator

The `DocInterrogator` class provides a way to extract answers from PDFs and images. It supports three different model types: 'donut', 'layoutlm', and 'pix2struct'.

## Installation

```shell
pip install docinterrogator 
```

## Usage

To use the `DocInterrogator` class, follow these steps:

```python
from docinterrogator.DocInterrogator import DocInterrogator
di = DocInterrogator(model_type='donut')
image_path = 'path/to/image.png'
questions = ['What is the title?', 'Who is the author?']
answers = doc_interrogator.extract_answers(image_path, questions)
```
