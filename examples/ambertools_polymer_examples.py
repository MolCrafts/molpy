"""
Example usage of AmberTools polymer builder for molpy.
"""

import logging
from pathlib import Path
import molpy as mp
from molpy.builder.polymer_ambertools import (
    AmberToolsPolymerBuilder,
    build_polymer_with_ambertools,
    BuilderStep
)

# Set up logging to see workflow progress
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')


def example_simple_polymer():
    """Example: Build a simple polyethylene chain."""
    print("\n=== Building Simple Polyethylene Chain ===")
    
    try:
        frame = build_polymer_with_ambertools(
            monomer_smiles="CC",  # Ethylene monomer
            sequence="A",         # Single monomer type
            n_repeat=5,          # 5 repeat units
            workdir="./examples/polyethylene"
        )
        
        print(f"✓ Built polymer with {len(frame['atoms']['id'])} atoms")
        print(f"✓ Topology file: {frame.props.get('prmtop')}")
        print(f"✓ Coordinate file: {frame.props.get('inpcrd')}")
        
        return frame
        
    except Exception as e:
        print(f"✗ Failed to build polymer: {e}")
        return None


def example_copolymer():
    """Example: Build an alternating copolymer."""
    print("\n=== Building Alternating Copolymer ===")
    
    try:
        # Build alternating ethylene-propylene copolymer
        frame = build_polymer_with_ambertools(
            monomer_smiles="CC",  # Ethylene (could be extended for multiple monomers)
            sequence="A-B-A-B",  # Alternating pattern
            n_repeat=3,          # 3 repetitions of the A-B-A-B pattern
            workdir="./examples/copolymer"
        )
        
        print(f"✓ Built copolymer with {len(frame['atoms']['id'])} atoms")
        print(f"✓ Sequence pattern repeated 3 times")
        
        return frame
        
    except Exception as e:
        print(f"✗ Failed to build copolymer: {e}")
        return None


def example_custom_workflow():
    """Example: Create a custom workflow with additional steps."""
    print("\n=== Custom Workflow with Additional Steps ===")
    
    class OptimizationStep(BuilderStep):
        """Custom step to add geometry optimization."""
        
        @property
        def name(self) -> str:
            return "optimization"
        
        def run(self, context):
            """Add geometry optimization using AMBER."""
            print("  → Running geometry optimization (simulated)")
            # In real implementation, this would call sander or pmemd
            # context['optimized_coords'] = run_optimization(context['inpcrd_file'])
            return context
    
    class AnalysisStep(BuilderStep):
        """Custom step to analyze the built polymer."""
        
        @property
        def name(self) -> str:
            return "analysis"
        
        def run(self, context):
            """Analyze polymer properties."""
            frame = context.get('frame')
            if frame:
                n_atoms = len(frame['atoms']['id'])
                print(f"  → Analysis: Polymer has {n_atoms} atoms")
                context['analysis'] = {'n_atoms': n_atoms}
            return context
    
    try:
        # Create builder with custom steps
        builder = AmberToolsPolymerBuilder()
        
        # Add custom steps
        builder.add_step(OptimizationStep(), position=-1)  # Before final step
        builder.add_step(AnalysisStep())  # At the end
        
        print(f"Workflow steps: {builder.get_step_names()}")
        
        frame = builder.build(
            monomer_smiles="CCO",  # Ethanol monomer
            sequence="A",
            n_repeat=4,
            workdir="./examples/custom_workflow"
        )
        
        print(f"✓ Built polymer with custom workflow")
        print(f"✓ Final frame has {len(frame['atoms']['id'])} atoms")
        
        return frame
        
    except Exception as e:
        print(f"✗ Custom workflow failed: {e}")
        return None


def example_error_handling():
    """Example: Demonstrate error handling."""
    print("\n=== Error Handling Example ===")
    
    try:
        # This should fail with invalid SMILES
        frame = build_polymer_with_ambertools(
            monomer_smiles="INVALID_SMILES",
            sequence="A",
            n_repeat=2,
            workdir="./examples/error_test"
        )
        
    except Exception as e:
        print(f"✓ Correctly caught error: {type(e).__name__}: {e}")


def main():
    """Run all examples."""
    print("AmberTools Polymer Builder Examples")
    print("===================================")
    
    # Check if molq is available
    try:
        import molq
        print("✓ molq is available")
    except ImportError:
        print("✗ molq is not available - examples will fail")
        print("  Install with: pip install molq")
        return
    
    # Run examples
    examples = [
        example_simple_polymer,
        example_copolymer,
        example_custom_workflow,
        example_error_handling
    ]
    
    results = {}
    for example in examples:
        try:
            result = example()
            results[example.__name__] = result
        except Exception as e:
            print(f"✗ Example {example.__name__} failed: {e}")
            results[example.__name__] = None
    
    # Summary
    print("\n=== Summary ===")
    for name, result in results.items():
        status = "✓" if result is not None else "✗"
        print(f"{status} {name}")
    
    print("\nFor more advanced usage, see the AmberToolsPolymerBuilder class documentation.")


if __name__ == "__main__":
    main()
