# Issues / TODOs

## inoculations.py: DRY violation and unclear function distinction

**File:** `src/core/inoculations.py`

### Problem

Two functions do nearly the same thing:

1. `load_inoculation_prompt(name)` - raises `ValueError` on error, silent on success
2. `resolve_inoculation(name)` - calls `sys.exit(1)` on error, prints confirmation message

This violates DRY since both functions duplicate the core logic of loading from YAML and validating the name exists. The distinction between them is also not documented, making it unclear which to use.

### Current behavior

```python
def load_inoculation_prompt(name: str) -> str:
    """Load an inoculation prompt by name from inoculations.yaml."""
    # ... raises ValueError on unknown name

def resolve_inoculation(name: str) -> str:
    """Load and validate an inoculation prompt by name, exiting on error."""
    # ... calls sys.exit(1) on unknown name, prints status message
```

### Suggested fix

Refactor so `resolve_inoculation` wraps `load_inoculation_prompt`:

```python
def resolve_inoculation(name: str) -> str:
    """CLI wrapper: load inoculation, print status, exit on error."""
    try:
        prompt = load_inoculation_prompt(name)
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)
    print(f"[Inoculation Prefill] Using '{name}'")
    return prompt
```

This keeps the library function (`load_inoculation_prompt`) clean and composable, while the CLI helper (`resolve_inoculation`) handles user-facing concerns.
