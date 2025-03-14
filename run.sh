#!/bin/bash

PYTHONPATH=. python run.py --root . --config invokeai.yaml > logs/invokeai.log 2>&1 &