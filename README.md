# BCQA (Benchmarking Complex QA)

BCQA is a benchmark for a wide range of complex Qa tasks. It also aims to provide a easy to use framework for evaluating retrieval and reasoning approaches for answering complex multi-hop questions.


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
