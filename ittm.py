"""
Infinite Time Turing Machine (ITTM) Implementation
Extends classical computation into transfinite ordinal steps.
"""

from typing import Dict, Tuple, Optional, Set, List
from dataclasses import dataclass
from enum import Enum
from turing_machine import TuringMachine, Direction, Transition


class LimitState(Enum):
    """States at limit ordinals"""
    LIMIT = "limit"
    SUCCESSOR = "successor"


@dataclass
class ITTMConfiguration:
    """Configuration of an ITTM at a given time step"""
    state: str
    tape: Dict[int, str]
    head_position: int
    time_step: int  # Can be transfinite ordinal
    is_limit: bool = False


class InfiniteTimeTuringMachine:
    """
    Infinite Time Turing Machine (ITTM)
    
    Extends classical Turing Machines to operate through:
    - Finite successor ordinals (normal steps)
    - Limit ordinals (special limit behavior)
    - Transfinite ordinals (beyond infinity)
    
    Key properties:
    - At limit ordinals, the state becomes the limit state
    - The tape cell values become the limsup of previous values
    - Enables hypercomputation beyond classical limits
    """
    
    def __init__(
        self,
        states: List[str],
        alphabet: List[str],
        blank_symbol: str = "B",
        initial_state: str = "q0",
        limit_state: str = "limit",
        accept_state: str = "accept",
        reject_state: str = "reject"
    ):
        self.states = set(states)
        self.alphabet = set(alphabet)
        self.blank_symbol = blank_symbol
        self.initial_state = initial_state
        self.limit_state = limit_state
        self.accept_state = accept_state
        self.reject_state = reject_state
        
        # Transition function
        self.transitions: Dict[Tuple[str, str], Transition] = {}
        
        # History of configurations for limit ordinal computation
        self.configuration_history: List[ITTMConfiguration] = []
        
        # Current configuration
        self.current_state = initial_state
        self.tape: Dict[int, str] = {0: blank_symbol}
        self.head_position = 0
        self.time_step = 0
        
    def add_transition(
        self,
        state: str,
        read_symbol: str,
        write_symbol: str,
        move: Direction,
        next_state: str
    ):
        """Add a transition rule"""
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
        self.time_step = 0
        self.configuration_history = []
        self._save_configuration()
    
    def _save_configuration(self):
        """Save current configuration to history"""
        config = ITTMConfiguration(
            state=self.current_state,
            tape=self.tape.copy(),
            head_position=self.head_position,
            time_step=self.time_step,
            is_limit=self._is_limit_ordinal(self.time_step)
        )
        self.configuration_history.append(config)
    
    def _is_limit_ordinal(self, ordinal: int) -> bool:
        """
        Check if an ordinal is a limit ordinal.
        In practice, we use a simple heuristic: ordinals divisible by omega
        For full implementation, this would need proper ordinal arithmetic.
        """
        # Simplified: treat certain ordinals as limits
        # In full implementation, would use proper ordinal notation
        return ordinal > 0 and ordinal % 100 == 0
    
    def _compute_limsup(self, position: int) -> str:
        """
        Compute the limsup (limit superior) of a tape cell at a limit ordinal.
        This is the maximum value that appears infinitely often.
        """
        if not self.configuration_history:
            return self.blank_symbol
        
        # Get all values at this position throughout history
        values = []
        for config in self.configuration_history:
            if position in config.tape:
                values.append(config.tape[position])
            else:
                values.append(self.blank_symbol)
        
        # Find the limsup: the maximum value that appears infinitely often
        # Simplified implementation: return the maximum value seen
        if not values:
            return self.blank_symbol
        
        # Order symbols: blank < 0 < 1 < ... (lexicographic)
        symbol_order = {self.blank_symbol: 0}
        for i, sym in enumerate(sorted(self.alphabet), 1):
            symbol_order[sym] = i
        
        # Find the maximum symbol that appears frequently
        max_symbol = max(values, key=lambda s: symbol_order.get(s, 0))
        return max_symbol
    
    def _handle_limit_ordinal(self):
        """
        Handle computation at a limit ordinal.
        - State becomes the limit state
        - Each tape cell becomes the limsup of its previous values
        - Head position becomes 0 (or limsup of positions)
        """
        self.current_state = self.limit_state
        
        # Compute limsup for each tape position that has been used
        all_positions = set()
        for config in self.configuration_history:
            all_positions.update(config.tape.keys())
        
        new_tape = {}
        for position in all_positions:
            new_tape[position] = self._compute_limsup(position)
        
        self.tape = new_tape
        self.head_position = 0  # Reset to start at limit
    
    def get_current_symbol(self) -> str:
        """Get the symbol at the current head position"""
        return self.tape.get(self.head_position, self.blank_symbol)
    
    def step(self) -> bool:
        """
        Execute one computation step (successor or limit).
        Returns True if computation continues, False if halted.
        """
        if self.current_state in [self.accept_state, self.reject_state]:
            return False
        
        # Check if we're at a limit ordinal
        if self._is_limit_ordinal(self.time_step):
            self._handle_limit_ordinal()
        else:
            # Normal successor ordinal step
            current_symbol = self.get_current_symbol()
            key = (self.current_state, current_symbol)
            
            if key not in self.transitions:
                # No transition defined
                if self.current_state == self.limit_state:
                    # From limit state, transition to first state
                    self.current_state = self.initial_state
                else:
                    self.current_state = self.reject_state
                    return False
            else:
                transition = self.transitions[key]
                
                # Write symbol
                self.tape[self.head_position] = transition.write_symbol
                
                # Move head
                self.head_position += transition.move.value
                
                # Update state
                self.current_state = transition.next_state
        
        self.time_step += 1
        self._save_configuration()
        return True
    
    def run(self, max_steps: int = 10000) -> str:
        """
        Run the ITTM until it halts or max_steps is reached.
        Returns 'accept', 'reject', or 'timeout'
        """
        while self.time_step < max_steps:
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
        return (f"ITTM(state={self.current_state}, "
                f"head={self.head_position}, time={self.time_step})")

