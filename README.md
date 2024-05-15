# BCQA (Benchmarking Complex QA)

BCQA is a benchmark for a wide range of complex Qa tasks. It also aims to provide a easy to use framework for evaluating retrieval and reasoning approaches for answering complex multi-hop questions.


# Setup
1) Clone the repo <br />
2) Create a conda environment conda create -n bcqa  <br />
3) pip install -e .<br />

# Running Evaluation
The evaluation scripts for retreival and LLMs are in the evaluation folder 

For instance to run dpr retreival for Wikimultihopqa run <br/>
python3 evaluation/wikimultihop/run_dpr_inference.py <br />

Before running the above script make sure you have configured the correct paths for the data and corpus files in evaluation/config.ini <br />

Example: 
wikimultihopqa = /home/bcqa/BCQA/2wikimultihopQA <br />
wikimultihopqa-corpus = /home/bcqa/BCQA/wiki_musique_corpus.json <br />


## Coding Practices

### Auto-formatting code
1. Install `black`: ```pip install black``` or ```conda install black```
2. In your IDE: Enable formatting on save.
3. Install `isort`: ```pip install isort``` or ```conda install isort```
4. In your IDE: Enable sorting import on save.

In VS Code, you can do this using the following config:
```json
{
    "editor.formatOnSave": true,
    "editor.codeActionsOnSave": {
        "source.organizeImports": true
    }
}
```

### Type hints
Use [type hints](https://docs.python.org/3/library/typing.html) for __everything__! No exceptions.

### Docstrings
Write a docstring for __every__ function (except the main function). We use the [Google format](https://github.com/NilsJPWerner/autoDocstring/blob/HEAD/docs/google.md). In VS Code, you can use [autoDocstring](https://marketplace.visualstudio.com/items?itemName=njpwerner.autodocstring).

### Example
```python
def sum(a: float, b: float) -> float:
    """Compute the sum of a and b.

    Args:
        a (float): First number.
        b (float): Second number.
    
    Returns:
        float: The sum of a and b.
    """

    return a + b
```
