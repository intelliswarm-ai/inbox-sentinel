# Archived Files

This directory contains the original files from before the refactoring. They have been preserved for reference but are no longer part of the active codebase.

## Directory Structure

- **original_files/**: Original implementation files
  - `mcp_*.py`: Original MCP server implementations
  - `*_workflow.py`: Original workflow implementations
  - `train_*.py`: Original training scripts
  - `email_preprocessor.py`: Original preprocessor

- **test_files/**: Original test scripts
  - Various test and demo scripts

## Migration Notes

The functionality from these files has been reorganized into the new package structure:

- MCP servers → `inbox_sentinel/servers/mcp/`
- ML models → `inbox_sentinel/ml/models/`
- Preprocessing → `inbox_sentinel/ml/preprocessing/`
- Training scripts → `inbox_sentinel/scripts/`
- Tests → `tests/`

These files are kept for reference only and should not be used directly.