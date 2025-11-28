"""
Example usage of the ITTM and USM implementations
Demonstrates the key concepts from the paper.
"""

from turing_machine import TuringMachine, create_binary_counter, Direction
from ittm import InfiniteTimeTuringMachine
from universal_state_machine import UniversalStateMachine
from transformer_attention import TransformerAsITTM, create_causal_mask
import numpy as np


def example_turing_machine():
    """Example: Binary Counter Turing Machine"""
    print("=" * 60)
    print("Example 1: Classical Turing Machine - Binary Counter")
    print("=" * 60)
    
    tm = create_binary_counter()
    tm.load_input("101")  # Binary number 5
    
    print(f"Initial tape: {tm.get_tape_string()}")
    print(f"Initial state: {tm.current_state}")
    
    result = tm.run(max_steps=100)
    print(f"Final tape: {tm.get_tape_string()}")
    print(f"Final state: {tm.current_state}")
    print(f"Result: {result}")
    print(f"Steps taken: {tm.step_count}")
    print()


def example_ittm():
    """Example: Infinite Time Turing Machine"""
    print("=" * 60)
    print("Example 2: Infinite Time Turing Machine")
    print("=" * 60)
    
    ittm = InfiniteTimeTuringMachine(
        states=["q0", "q1", "limit", "accept"],
        alphabet=["0", "1"],
        blank_symbol="B",
        initial_state="q0",
        limit_state="limit"
    )
    
    # Simple transition: flip bits
    ittm.add_transition("q0", "0", "1", Direction.RIGHT, "q0")
    ittm.add_transition("q0", "1", "0", Direction.RIGHT, "q0")
    ittm.add_transition("q0", "B", "B", Direction.LEFT, "accept")
    
    ittm.load_input("1010")
    
    print(f"Initial tape: {ittm.get_tape_string()}")
    print(f"Initial state: {ittm.current_state}")
    print(f"Initial time step: {ittm.time_step}")
    
    # Run for a few steps
    for i in range(10):
        ittm.step()
        if ittm.time_step % 5 == 0:
            print(f"Time step {ittm.time_step}: {ittm.get_tape_string()}, state={ittm.current_state}")
    
    print(f"Final time step: {ittm.time_step}")
    print()


def example_usm_basic():
    """Example: Basic USM usage"""
    print("=" * 60)
    print("Example 3: Universal State Machine - Basic Usage")
    print("=" * 60)
    
    usm = UniversalStateMachine()
    
    # Add knowledge about concepts
    usm.add_knowledge("cat", {"type": "animal", "legs": 4}, ["mammal", "pet"])
    usm.add_knowledge("dog", {"type": "animal", "legs": 4}, ["mammal", "pet"])
    usm.add_knowledge("bird", {"type": "animal", "legs": 2}, ["mammal"])
    usm.add_knowledge("mammal", {"type": "category"}, ["animal"])
    usm.add_knowledge("pet", {"type": "category"}, [])
    usm.add_knowledge("animal", {"type": "root_category"}, [])
    
    print("Knowledge graph created with concepts: cat, dog, bird, mammal, pet, animal")
    
    # Query the knowledge graph
    result = usm.query("cat", depth=2)
    print(f"\nQuery 'cat' (depth=2):")
    print(f"  Nodes found: {list(result['nodes'].keys())}")
    print(f"  Edges: {len(result['edges'])}")
    
    # Process information
    activations = usm.process_information(["cat", "dog"], max_depth=3)
    print(f"\nProcessing information from ['cat', 'dog']:")
    print(f"  Activated nodes: {len(activations)}")
    for node, activation in sorted(activations.items(), key=lambda x: x[1], reverse=True)[:5]:
        print(f"    {node}: {activation:.3f}")
    
    # Get statistics
    stats = usm.get_statistics()
    print(f"\nUSM Statistics:")
    print(f"  Nodes: {stats['node_count']}")
    print(f"  Edges: {stats['edge_count']}")
    print(f"  Average connectivity: {stats['average_connectivity']:.2f}")
    print()


def example_usm_learning():
    """Example: USM learning with Voyager Calibration"""
    print("=" * 60)
    print("Example 4: USM Learning with Voyager Calibration")
    print("=" * 60)
    
    usm = UniversalStateMachine()
    
    # Initialize with some structure
    for i in range(10):
        usm.add_knowledge(f"node_{i}", {"value": i * 0.1})
        if i > 0:
            usm.knowledge_graph.add_edge(f"node_{i-1}", f"node_{i}", weight=0.5)
    
    print("Initial graph created with 10 nodes")
    
    # Define target pattern
    target_pattern = {
        "average_activation": 0.7,
        "connectivity": 2.0
    }
    
    print(f"Target pattern: {target_pattern}")
    
    # Learn the pattern
    result = usm.learn_pattern(target_pattern, convergence_threshold=0.05)
    
    print(f"\nLearning completed:")
    print(f"  Iterations: {result['iterations']}")
    print(f"  Final error: {result['final_error']:.4f}")
    print(f"  Converged: {result['final_error'] < 0.05}")
    print()


def example_transformer_ittm():
    """Example: Transformer as ITTM implementation"""
    print("=" * 60)
    print("Example 5: Transformer as ITTM Implementation")
    print("=" * 60)
    
    # Create a small transformer
    transformer = TransformerAsITTM(
        vocab_size=100,
        d_model=64,
        d_ff=256,
        num_layers=3,
        num_heads=1
    )
    
    # Example input tokens
    token_ids = [1, 5, 10, 15, 20]
    
    print(f"Input tokens: {token_ids}")
    
    # Forward pass
    logits, attention_weights = transformer.forward(token_ids)
    
    print(f"Output logits shape: {logits.shape}")
    print(f"Number of attention layers: {len(attention_weights)}")
    
    # Show attention pattern from first layer
    if attention_weights:
        print(f"\nAttention pattern (first layer, first token):")
        first_attention = attention_weights[0][0]
        print(f"  Shape: {first_attention.shape}")
        print(f"  Max attention: {first_attention.max():.3f}")
        print(f"  Min attention: {first_attention.min():.3f}")
    
    # Convert to USM representation
    usm = UniversalStateMachine()
    node_mapping = transformer.to_usm_representation(usm)
    
    print(f"\nTransformer converted to USM:")
    print(f"  USM nodes: {usm.get_statistics()['node_count']}")
    print(f"  USM edges: {usm.get_statistics()['edge_count']}")
    print()


def example_electron_synthesis():
    """Example: Electron Synthesis Algorithm for information routing"""
    print("=" * 60)
    print("Example 6: Electron Synthesis - Information Routing")
    print("=" * 60)
    
    usm = UniversalStateMachine()
    
    # Create a knowledge graph about a topic
    # Topic: Machine Learning
    concepts = {
        "neural_network": ["deep_learning", "backpropagation"],
        "deep_learning": ["transformer", "cnn", "rnn"],
        "transformer": ["attention", "self_attention"],
        "attention": ["query", "key", "value"],
        "cnn": ["convolution", "pooling"],
        "rnn": ["lstm", "gru"],
        "backpropagation": ["gradient_descent"],
        "gradient_descent": ["optimization"]
    }
    
    # Build the graph
    for concept, connections in concepts.items():
        usm.add_knowledge(concept, {"topic": "machine_learning"})
        for conn in connections:
            usm.knowledge_graph.add_edge(concept, conn, weight=1.0)
    
    print("Knowledge graph created about Machine Learning concepts")
    
    # Route a query
    query = "How does attention work in transformers?"
    context = ["transformer", "attention"]
    
    results = usm.route_query(query, context, top_k=5)
    
    print(f"\nQuery: '{query}'")
    print(f"Context: {context}")
    print(f"\nTop relevant concepts:")
    for i, (concept, score) in enumerate(results, 1):
        print(f"  {i}. {concept}: {score:.3f}")
    print()


def main():
    """Run all examples"""
    print("\n" + "=" * 60)
    print("ITTM and USM Implementation Examples")
    print("Based on: Infinite Time Turing Machines and their Applications")
    print("=" * 60 + "\n")
    
    example_turing_machine()
    example_ittm()
    example_usm_basic()
    example_usm_learning()
    example_transformer_ittm()
    example_electron_synthesis()
    
    print("=" * 60)
    print("All examples completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()

