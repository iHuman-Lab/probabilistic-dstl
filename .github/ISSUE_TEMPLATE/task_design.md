---
name: Task Design & Alignment
about: Propose a task, design its architecture, and justify the validation before coding.
title: '[TASK]: '
labels: design-review
assignees: ''
---

> **Workflow:** Fill out Sections 1, 2, and 3. Wait for PI alignment/approval before creating a branch or writing code.

## 1. Scope & Architecture
* **Objective:** *(1-2 sentences on the specific problem this solves).*
* **Config Updates:** List any new parameters needed in the central config file.
* **Entry Point Impacts:** What needs to change in `main.py` to route this task?

## 2. Implementation Blueprint & LLM Guardrails
* **Logical Steps:**
    1. [ ]
    2. [ ]
* **LLM Boundary:** What *specific* function/syntax are you delegating? *(No structural copy-pasting).*

## 3. Validation & Adversarial Justification
* **The Validation Method:** *(The exact command, script, or test to execute to check correctness).*
* **The Success Condition:** *(What specific, measurable output proves it worked? e.g., falsifiable metrics, tensor shapes).*
* **Justification (Why is this sufficient?):** *Defend your verification method. Address these specific targets:*
    * How does this validation cover the failure space (not just the obvious successes)?
    * What is a case where this validation could pass, but the system is still wrong? How do we catch that?