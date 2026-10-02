#!/bin/bash
# Schedule execution of many runs
# Run from root folder with: bash scripts/schedule.sh

uv run train-command trainer.max_epochs=5 logger=csv

uv run train-command trainer.max_epochs=10 logger=csv
