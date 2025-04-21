# Agent(s) Architecture

```ascii
┌─────────────┐      ┌────────────────┐     ┌─────────────────┐
│ System Data │─────►│ Primary Agent  │────►│Regex x AST Valid│
└─────────────┘      │ (LLM)          │     └────────┬────────┘
                     └────────────────┘              │
                                                     ▼
┌─────────────┐      ┌────────────────┐     ┌─────────────────┐
│ Tool        │◄─────│ Confirmation   │◄────│ Validated Tool  │
│ Execution   │      │ Agent (LLM)    │     │ Calls           │
└─────────────┘      └────────────────┘     └─────────────────┘
```

A secure multi-agent system implementing a two-phase validation pattern for AI-initiated system operations through coordinated primary and confirmation agents.

## Core Axioms

### 1. Dual-Agent Authority

Two independent AI agents operate through a secure validation mechanism:
- **Primary Agent**: Analyzes data and proposes specific tool actions based on predefined criteria
- **Confirmation Agent**: Independently validates proposed actions to prevent potentially harmful operations

The separation of powers between these agents creates defense-in-depth through isolation of decision-making authority.

### 2. Ephemeral Action Authorization

Each proposed action must pass a distinct, three-phase approval process:
- **Phase 1: Primary Analysis & Proposal**: The primary agent analyzes system data and proposes specific tool invocations
- **Phase 2: Structural Validation**: Each proposed action is validated against strict regex patterns
- **Phase 3: Semantic Confirmation**: A separate confirmation agent evaluates the nature and impact of proposed actions

Only the union of all three approvals permits execution, with any single denial preventing action completion.

### 3. Virtualized Execution Environment

Code execution occurs in an isolated, precisely controlled environment with:
- AST-based code validation prior to execution
- Restricted access to builtins and limited import functionality
- Path validation to prevent traversal attacks
- Runtime controls including stdout/stderr capture and sanitization

## Transaction Flow

```
┌─────────────┐      ┌────────────────┐     ┌─────────────────┐
│ System Data │─────►│ Primary Agent  │────►│ Regex Validator │
└─────────────┘      │ (LLM)          │     └────────┬────────┘
                     └────────────────┘              │
                                                     ▼
┌─────────────┐      ┌────────────────┐     ┌─────────────────┐
│ Tool        │◄─────│ Confirmation   │◄────│ Validated Tool  │
│ Execution   │      │ Agent (LLM)    │     │ Calls           │
└─────────────┘      └────────────────┘     └─────────────────┘
```

## Security Implementation

### Zero Standing Privileges
- Each authorization is strictly ephemeral and bound to a single proposed action
- No long-lived tokens or credentials grant direct access to sensitive operations
- All approvals expire immediately upon use or rejection

### Isolated Contexts
- Clear security boundaries separate agents, validation, and execution phases
- Independent validation contexts prevent cross-contamination of authorization logic
- AST validation enforces boundary between code validation and execution environments

### Reduced Attack Surface
- Structural validation acts as a first-pass filter before semantic evaluation
- Secondary confirmation agent provides independent judgment on action safety
- AST validation prevents unsafe code execution patterns before runtime

### Enhanced Audit Trail
- Comprehensive logging of all decisions and actions for accountability
- Full capture of proposed, validated, and executed operations
- Clear decision boundaries establish responsibility for authorization

## Implementation Requirements

### Agent Configuration
- Models must be configured for structured output format adherence (low temperature)
- System prompts explicitly discourage explanation in favor of specific action proposals
- Confirmation agent receives minimal context to force independent judgment

### Validation Architecture 
- Strict regex patterns statically validate structural conformance
- AST validation provides static analysis prior to execution
- Built-in mechanism rejects any action failing validation at any phase

### Execution Isolation
- Path validation restricts file access to designated workspace
- Limited builtins prevent access to sensitive system functionality
- Resource limits cap output volume to prevent flooding

---

In essence: The Agent(s) Architecture creates trustworthy AI system operations through validation layers. By requiring multiple independent approvals, enforcing structural conformity, and isolating execution, it achieves robust security with operational flexibility. Every action undergoes rigorous scrutiny before execution, establishing accountable automation for sensitive operations; all logged locally via DuckDB
