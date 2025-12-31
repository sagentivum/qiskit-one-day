# Code Annotation Guide

This guide defines the standards for code annotations (type hints and docstrings) across the project.

## Type Hints

### Requirements
- All functions MUST have type hints for parameters and return values
- Use `from __future__ import annotations` at the top of files (enables forward references and cleaner syntax)
- Use modern type hints: `list[Type]` instead of `List[Type]`, `dict[str, int]` instead of `Dict[str, int]`
- Use `Optional[Type]` or `Type | None` for optional values (note: `Optional[Type]` from typing module is also acceptable)
- Use `Tuple[Type1, Type2]` for tuples (from typing module) or `tuple[Type1, Type2]` for modern syntax
- Note: Both `List[Type]` from typing and `list[Type]` modern syntax are acceptable, but prefer consistency within a file

### Examples
```python
from __future__ import annotations
from typing import Optional, Tuple

def example_function(
    param1: str,
    param2: int,
    optional_param: Optional[str] = None
) -> tuple[list[int], float]:
    ...
```

## Docstrings

### Module-Level Docstrings
- All modules SHOULD have a module-level docstring describing the module's purpose
- Place immediately after imports, before any code

### Function Docstrings
- All functions MUST have docstrings
- Use triple-quoted strings (""")
- First line: brief one-line summary (no period unless multiple sentences)
- Optional blank line, then detailed description if needed
- Include "Args:" section describing all parameters
- Include "Returns:" section for functions that return values (describe return type and meaning)
- Include "Raises:" section if the function raises exceptions

### Class Docstrings
- All classes MUST have docstrings
- Describe the class purpose and any important attributes
- For dataclasses, document the purpose and fields

### Format Style
- Use Google-style docstrings (not NumPy or reStructuredText)
- Keep docstrings concise but informative

### Examples

```python
def solve_tsp_qaoa(
    D: np.ndarray,
    reps: int = 1,
    shots: int = 4096,
) -> TspQaoaResult:
    """
    Solve TSP using Qiskit Optimization's Tsp -> QuadraticProgram -> MinimumEigenOptimizer(QAOA).

    Returns:
        TspQaoaResult containing raw binary solution, decoded tour, and tour length.
    """
    ...
```

```python
@dataclass(frozen=True)
class City:
    """
    Represents a city with geographic coordinates.
    
    Attributes:
        name: City name
        lat: Latitude in decimal degrees
        lon: Longitude in decimal degrees
    """
    name: str
    lat: float
    lon: float
```

## Private Functions

- Private functions (prefixed with `_`) SHOULD have docstrings explaining their purpose
- Type hints are still required

## Inline Comments

- Use inline comments sparingly for non-obvious logic
- Prefer descriptive variable names over comments when possible
- Comments should explain "why" not "what"

