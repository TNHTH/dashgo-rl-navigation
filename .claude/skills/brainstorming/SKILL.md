---
name: brainstorming
description: "You MUST use this before any creative work - creating features, building components, adding functionality, or modifying behavior. Explores user intent, requirements and design before implementation through structured questioning and alternative exploration."
---

# Brainstorming Ideas Into Designs

## Overview

Help turn ideas into fully formed designs and specs through natural collaborative dialogue.

Start by understanding the current project context, then ask questions one at a time to refine the idea. Once you understand what you're building, present the design in small sections (200-300 words), checking after each section whether it looks right so far.

## The Process

**Understanding the idea:**
- Check out the current project state first (files, docs, recent commits)
- Ask questions one at a time to refine the idea
- Prefer multiple choice questions when possible, but open-ended is fine too
- Only one question per message - if a topic needs more exploration, break it into multiple questions
- Focus on understanding: purpose, constraints, success criteria

**Exploring approaches:**
- Propose 2-3 different approaches with trade-offs
- Present options conversationally with your recommendation and reasoning
- Lead with your recommended option and explain why

**Presenting the design:**
- Once you believe you understand what you're building, present the design
- Break it into sections of 200-300 words
- Ask after each section whether it looks right so far
- Cover: architecture, components, data flow, error handling, testing
- Be ready to go back and clarify if something doesn't make sense

## After the Design

**Documentation:**
- Write the validated design to `D:\cursor\file\Si Yuan\claude\plans\YYYY-MM-DD-<topic>-design.md`
- Create the plans directory if it doesn't exist
- Include the design rationale and alternatives considered
- Commit the design document to git if in a git repository

**Implementation (if continuing):**
- Ask: "Ready to set up for implementation?"
- Follow the file organization rules:
  - Project-specific implementations go to project directories
  - Standalone documents go to `Si Yuan\claude\`
  - Temporary files go to `.claude-temp\`

## Key Principles

- **One question at a time** - Don't overwhelm with multiple questions
- **Multiple choice preferred** - Easier to answer than open-ended when possible
- **YAGNI ruthlessly** - Remove unnecessary features from all designs
- **Explore alternatives** - Always propose 2-3 approaches before settling
- **Incremental validation** - Present design in sections, validate each
- **Be flexible** - Go back and clarify when something doesn't make sense

## Windows-Specific Notes

When saving designs on Windows:
- Use backslashes in paths: `D:\cursor\file\Si Yuan\claude\plans\`
- Ensure UTF-8 encoding for markdown files
- Use PowerShell commands for directory creation if needed:
  ```powershell
  New-Item -ItemType Directory -Force -Path "D:\cursor\file\Si Yuan\claude\plans"
  ```

## Example Usage

**User**: "I want to add a feature to my app"

**Brainstorming Process**:
1. Check current project state
2. Ask: "What should this feature do?" (one question)
3. Ask: "Who will use it?" (one question)
4. Present 3 approaches with trade-offs
5. Design in 200-300 word sections
6. Validate each section
7. Save to `D:\cursor\file\Si Yuan\claude\plans\2026-01-13-feature-design.md`
8. Ask if ready to implement

## Integration with File Organization Rules

This skill follows your established file organization rules:

1. **Design documents** → `D:\cursor\file\Si Yuan\claude\plans\`
2. **Project-specific docs** → Project directory (e.g., `multi-agent-system\docs\`)
3. **Temporary files** → `D:\cursor\file\.claude-temp\`

Always ask if you're unsure about the correct location for a file.
