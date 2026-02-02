# Extraction Models in AI Browser Agents

## What is an Extraction Model?

An **Extraction Model** is a structured schema (blueprint) that defines exactly what information the AI Agent should retrieve from a website. Instead of getting a generic text summary, you get structured data (like JSON) that matches specific fields you care about.

In the context of this project (using `browser-use` and `LangChain`), extraction models are typically defined using **Pydantic** classes.

## Why use them?

1.  **Precision**: Forces the LLM to look for specific data points (e.g., "price", "sku", "author").
2.  **Reliability**: Ensures the output is always in a predictable format (e.g., JSON) rather than free-form text.
3.  **Type Safety**: Guarantees that numbers are numbers, lists are lists, etc.
4.  **Automation**: Makes it easy to save data directly to databases or CSV files without complex parsing.

## How it works in code

When you define a model, you pass it to the Agent or Controller. The Agent then uses its "vision" and DOM analysis to find data on the page that fits your model.

### Example

If you want to extract product details from an e-commerce site:

```python
from pydantic import BaseModel
from typing import List, Optional

# 1. Define the shape of a single item
class ProductItem(BaseModel):
    title: str
    price: float
    currency: str = "USD"
    in_stock: bool
    rating: Optional[float]

# 2. Define the output structure (e.g., a list of items)
class ProductExtraction(BaseModel):
    products: List[ProductItem]
    total_found: int
```

### Integration in `CustomController`

Your `CustomController` already supports an `output_model` parameter:

```python
# src/controller/custom_controller.py

class CustomController(Controller):
    def __init__(self, output_model: Optional[Type[BaseModel]] = None, ...):
        super().__init__(output_model=output_model, ...)
```

When you initialize the controller with this model, the agent will prioritize actions that help fill this schema.

## Summary

Think of an Extraction Model as a **form** you give to the AI Agent. Instead of just "browsing," the Agent becomes a "data entry clerk" filling out that form based on what it sees on the website.