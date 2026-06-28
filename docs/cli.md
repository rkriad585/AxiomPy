# CLI Tool

The `axiompy` package installs a command-line tool for quick calculations and introspection — no Python import needed.

## Subcommands

| Command | Description |
|---------|-------------|
| `eval` | Evaluate a math expression |
| `factors` | Show prime factorization |
| `constants` | List or search built-in constants |
| `convert` | Show a number in multiple formats |
| `stats` | Cache and system statistics |
| `help` | Show detailed help on a topic |
| `shell` | Interactive REPL with `A` pre-imported |
| `demo` | Run all example scripts |
| `info` | Version, backend, config |

## Examples

```bash
# Arithmetic evaluation
axiompy eval "2 * pi * 5^2"
# 2 * pi * 5^2 = 157.07963267948966

# Prime factorization
axiompy factors 84
# 84 = 2 × 2 × 3 × 7

# List all constants (or filter by name)
axiompy constants
axiompy constants speed

# Number conversion (binary, hex, roman, words)
axiompy convert 42

# Cache statistics
axiompy stats

# Help on a topic
axiompy help magic
axiompy help constants

# Interactive shell
axiompy shell
```
