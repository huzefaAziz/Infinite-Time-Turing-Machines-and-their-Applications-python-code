"""
Classical Turing Machine Implementation
Based on the theoretical foundation described in the paper.
"""

from enum import Enum
from typing import Dict, Tuple, Optional, List
from dataclasses import dataclass


class Direction(Enum):
    """Tape movement directions"""
    LEFT = -1
    RIGHT = 1
    STAY = 0


@dataclass
class Transition:
    """Represents a state transition rule"""
    read_symbol: str
    write_symbol: str
    move: Direction
    next_state: str


class TuringMachine:
    """
    Classical Turing Machine implementation.
    
    A Turing Machine consists of:
    - A finite set of states
    - An infinite tape with cells containing symbols
    - A read/write head that can move left or right
    - A transition function that determines state changes
    """
    
    def __init__(
        self,
        states: List[str],
        alphabet: List[str],
        blank_symbol: str = "B",
        initial_state: str = "q0",
        accept_state: str = "accept",
        reject_state: str = "reject"
    ):
        self.states = set(states)
        self.alphabet = set(alphabet)
        self.blank_symbol = blank_symbol
        self.initial_state = initial_state
        self.accept_state = accept_state
        self.reject_state = reject_state
        
        # Transition function: (state, symbol) -> Transition
        self.transitions: Dict[Tuple[str, str], Transition] = {}
        
        # Current configuration
        self.current_state = initial_state
        self.tape: Dict[int, str] = {0: blank_symbol}
        self.head_position = 0
        self.step_count = 0
        
    def add_transition(
        self,
        state: str,
        read_symbol: str,
        write_symbol: str,
        move: Direction,
        next_state: str
    ):
        """Add a transition rule to the machine"""
        if state not in self.states:
            raise ValueError(f"State {state} not in states")
        if read_symbol not in self.alphabet and read_symbol != self.blank_symbol:
            raise ValueError(f"Symbol {read_symbol} not in alphabet")
            
        self.transitions[(state, read_symbol)] = Transition(
            read_symbol=read_symbol,
            write_symbol=write_symbol,
            move=move,
            next_state=next_state
        )
    
    def load_input(self, input_string: str):
        """Initialize the tape with input string"""
        self.tape = {}
        for i, symbol in enumerate(input_string):
            self.tape[i] = symbol
        self.head_position = 0
        self.current_state = self.initial_state
        self.step_count = 0
    
    def get_current_symbol(self) -> str:
        """Get the symbol at the current head position"""
        return self.tape.get(self.head_position, self.blank_symbol)
    
    def step(self) -> bool:
        """
        Execute one computation step.
        Returns True if computation continues, False if halted.
        """
        if self.current_state in [self.accept_state, self.reject_state]:
            return False
        
        current_symbol = self.get_current_symbol()
        key = (self.current_state, current_symbol)
        
        if key not in self.transitions:
            # No transition defined - halt and reject
            self.current_state = self.reject_state
            return False
        
        transition = self.transitions[key]
        
        # Write symbol
        self.tape[self.head_position] = transition.write_symbol
        
        # Move head
        self.head_position += transition.move.value
        
        # Update state
        self.current_state = transition.next_state
        self.step_count += 1
        
        return True
    
    def run(self, max_steps: int = 10000) -> str:
        """
        Run the machine until it halts or max_steps is reached.
        Returns 'accept', 'reject', or 'timeout'
        """
        while self.step_count < max_steps:
            if not self.step():
                break
        
        if self.current_state == self.accept_state:
            return "accept"
        elif self.current_state == self.reject_state:
            return "reject"
        else:
            return "timeout"
    
    def get_tape_string(self, start: int = -10, end: int = 10) -> str:
        """Get a string representation of the tape around the head"""
        result = []
        for i in range(start, end + 1):
            symbol = self.tape.get(i, self.blank_symbol)
            if i == self.head_position:
                result.append(f"[{symbol}]")
            else:
                result.append(symbol)
        return " ".join(result)
    
    def __repr__(self):
        return (f"TuringMachine(state={self.current_state}, "
                f"head={self.head_position}, steps={self.step_count})")


def create_binary_counter() -> TuringMachine:
    """
    Example: Binary Counter Turing Machine
    Increments a binary number on the tape.
    """
    tm = TuringMachine(
        states=["q0", "q1", "q2", "accept"],
        alphabet=["0", "1"],
        blank_symbol="B",
        initial_state="q0"
    )
    
    # State q0: Move right to find the end
    tm.add_transition("q0", "0", "0", Direction.RIGHT, "q0")
    tm.add_transition("q0", "1", "1", Direction.RIGHT, "q0")
    tm.add_transition("q0", "B", "B", Direction.LEFT, "q1")
    
    # State q1: Increment from right to left
    tm.add_transition("q1", "0", "1", Direction.LEFT, "q2")
    tm.add_transition("q1", "1", "0", Direction.LEFT, "q1")
    tm.add_transition("q1", "B", "1", Direction.RIGHT, "accept")
    
    # State q2: Copy remaining bits
    tm.add_transition("q2", "0", "0", Direction.LEFT, "q2")
    tm.add_transition("q2", "1", "1", Direction.LEFT, "q2")
    tm.add_transition("q2", "B", "B", Direction.RIGHT, "accept")
    
    return tm

