# Amazon ML Challenge 2026

## Business Entity Resolution

This repository contains our solution for the Amazon ML Challenge 2026.

The task is to identify matching business entities across three independent
data sources containing noisy and inconsistent business information.

## Pipeline

Data Loading
→ Preprocessing
→ Candidate Generation / Blocking
→ Feature Engineering
→ ML Model
→ Inference
→ Output Generation

## Project Structure

```text
src/
├── preprocessing/
├── blocking/
├── features/
├── model/
├── inference/
└── evaluation/

notebooks/
tests/
docs/
output/