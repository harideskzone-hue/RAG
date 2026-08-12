#!/bin/bash
MODEL_FREE=true VLM_PROVIDER=none REASONING_PROVIDER=none PYTHONPATH=. .venv/bin/python scripts/chat_cli.py --url http://localhost:8001/api/v1/chat << 'CLI_INPUT'
What can you tell me about the person in the blue shirt?
exit
CLI_INPUT
