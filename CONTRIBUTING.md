# Contributing to QuantumCortex

Thank you for your interest in contributing to QuantumCortex!

## How to Contribute

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/amazing-feature`
3. Commit your changes: `git commit -m 'Add amazing feature'`
4. Push to the branch: `git push origin feature/amazing-feature`
5. Open a Pull Request

## Development Setup

```bash
# Clone your fork
git clone https://github.com/YOUR_USERNAME/QuantumCortex.git
cd QuantumCortex

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac

# Install with dev dependencies
pip install -e ".[dev]"

# Run tests
pytest tests/ -v
```

## Code Style

- Follow PEP 8
- Use type hints where possible
- Write docstrings for all public functions
- Add tests for new features

## Pull Request Checklist

- [ ] Tests pass
- [ ] Code is properly formatted
- [ ] Documentation is updated
- [ ] Changes are backward compatible
