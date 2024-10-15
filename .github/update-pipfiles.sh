#!/bin/bash

for v in "3.9" "3.10" "3.11" "3.12" "3.13"; do
  uv pip compile --python-version $v requirements.txt tests/requirements.txt .github/requirements.txt > .github/requirements$v.txt
done
