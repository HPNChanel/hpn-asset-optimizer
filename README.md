# Asset Optimizer (`aopt`)

> **Ultra-high-performance, offline CLI for image optimization**

[![Python 3.11+](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Built with Typer](https://img.shields.io/badge/Built%20with-Typer-ff69b4.svg)](https://typer.tiangolo.com/)

## ✨ Features

- 🚀 **Speed**: SIMD-optimized image processing with concurrent batch operations
- 🔒 **Privacy**: 100% offline - no data leaves your machine
- 🎨 **Beautiful CLI**: Rich dashboard with real-time progress
- 🤖 **AI-Powered**: OCR-based PII detection, smart content-aware resizing
- 📦 **Presets**: Built-in optimization profiles (web, print, social, email)

## 🚀 Quick Start

```bash
# Install with Poetry
poetry install

# Compress an image
aopt compress image.jpg --quality 80

# Convert to WebP
aopt convert image.png --format webp

# Batch process a directory
aopt batch ./images/ --output ./optimized/

# Apply a preset
aopt preset apply web image.jpg
```

## 📖 Commands

| Command | Description |
|---------|-------------|
| `aopt compress <path>` | Compress images (lossy/lossless) |
| `aopt convert <path>` | Convert image format |
| `aopt batch <dir>` | Process entire directory |
| `aopt shield <path>` | Detect & redact PII |
| `aopt info <path>` | Show image metadata |
| `aopt strip <path>` | Remove all metadata |
| `aopt preset list` | Show available presets |
| `aopt preset apply` | Apply optimization preset |

## 🛠️ Development

```bash
# Install dev dependencies
poetry install --with dev,ai

# Run tests
poetry run pytest

# Lint code
poetry run ruff check src/
```

## 📄 License

MIT License - HPN Corporation
