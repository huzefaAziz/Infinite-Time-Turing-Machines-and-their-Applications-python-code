# Infinite Time Turing Machines and Universal State Machine

Python implementation based on the paper: "Infinite Time Turing Machines and their Applications" (arXiv:2506.05351)

This repository contains implementations of:
- **Classical Turing Machines** - Basic computational model
- **Infinite Time Turing Machines (ITTMs)** - Extension to transfinite ordinal steps
- **Universal State Machine (USM)** - Novel computational paradigm with:
  - Dynamic, queryable knowledge graphs
  - Voyager Calibration Algorithm
  - Electron Synthesis Algorithm
- **Transformer architectures** - Reinterpreted through the ITTM framework

## Features

### Turing Machine (`turing_machine.py`)
- Classical Turing Machine implementation
- State transitions and tape operations
- Example: Binary counter

### Infinite Time Turing Machine (`ittm.py`)
- Computation through transfinite ordinal steps
- Limit ordinal handling with limsup computation
- Hypercomputation capabilities

### Universal State Machine (`universal_state_machine.py`)
- **Knowledge Graph**: Dynamic, queryable graph structure
- **Electron Synthesis Algorithm**: Efficient information routing with quadratic speedup
- **Voyager Calibration Algorithm**: Fast convergence optimization
- Modular, interpretable computation

### Transformer Attention (`transformer_attention.py`)
- Attention mechanisms reinterpreted through ITTM framework
- Transformer blocks as ITTM computational steps
- Conversion to USM representation

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Basic Examples

Run the example file to see all implementations in action:

```bash
python examples.py
```

### Using a Turing Machine

```python
from turing_machine import create_binary_counter

tm = create_binary_counter()
tm.load_input("101")  # Binary number 5
result = tm.run()
print(f"Result: {result}")
print(f"Tape: {tm.get_tape_string()}")
```

### Using an Infinite Time Turing Machine

```python
from ittm import InfiniteTimeTuringMachine, Direction

ittm = InfiniteTimeTuringMachine(
    states=["q0", "q1", "limit", "accept"],
    alphabet=["0", "1"],
    limit_state="limit"
)

ittm.add_transition("q0", "0", "1", Direction.RIGHT, "q0")
ittm.load_input("1010")
ittm.run()
```

### Using the Universal State Machine

```python
from universal_state_machine import UniversalStateMachine

usm = UniversalStateMachine()

# Add knowledge
usm.add_knowledge("cat", {"type": "animal"}, ["mammal", "pet"])

# Query the graph
result = usm.query("cat", depth=2)

# Process information
activations = usm.process_information(["cat", "dog"])

# Learn patterns
target = {"average_activation": 0.7, "connectivity": 2.0}
usm.learn_pattern(target)
```

### Using Transformers as ITTM

```python
from transformer_attention import TransformerAsITTM

transformer = TransformerAsITTM(
    vocab_size=100,
    d_model=64,
    d_ff=256,
    num_layers=3
)

logits, attention_weights = transformer.forward([1, 5, 10, 15])
```

## Architecture

### Knowledge Graph Structure
The USM uses a dynamic knowledge graph where:
- **Nodes** represent concepts with state and metadata
- **Edges** represent relationships with weights
- The graph evolves in real-time through learning

### Electron Synthesis Algorithm
Efficiently routes information through the knowledge graph:
- Propagates activations from source nodes
- Uses edge weights for routing decisions
- Achieves quadratic speedup over naive attention

### Voyager Calibration Algorithm
Optimizes the knowledge graph structure:
- Transfinite-scale optimization
- Rapid convergence to target patterns
- Exponential speedup over gradient descent

## Key Concepts from the Paper

1. **ITTMs extend computation** beyond finite steps to transfinite ordinals
2. **Deep Learning as ITTM**: Modern architectures can be understood through ITTM principles
3. **USM advantages**:
   - Logarithmic query/update operations
   - Modular, composable structure
   - Real-time adaptation
   - Interpretable knowledge representation

## File Structure

```
.
├── turing_machine.py          # Classical Turing Machine
├── ittm.py                     # Infinite Time Turing Machine
├── universal_state_machine.py # USM core implementation
├── transformer_attention.py    # Transformer/attention mechanisms
├── examples.py                 # Usage examples
├── requirements.txt            # Dependencies
└── README.md                   # This file
```

## References

- Paper: [Infinite Time Turing Machines and their Applications](https://arxiv.org/pdf/2506.05351)
- Original ITTM paper: Hamkins & Lewis (2000)

## License

This implementation is provided for educational and research purposes.

