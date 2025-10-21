"""
Example: Lc -> p K pi decay with phase space generation and angle calculations

This example demonstrates:
1. Using phasespace package to generate Lc -> p K pi phase space events
2. Computing all helicity angles for all possible decay topologies
3. Computing Wigner rotations between different topologies
"""

import phasespace as phsp
import numpy as np
import json
from typing import Dict, List, Tuple, Any
from pydantic import BaseModel, Field
from decayangle.decay_topology import TopologyCollection
from decayangle.config import config as cfg

# Set configuration
cfg.sorting = "off"  # Disable sorting for consistent results

# Pydantic models for data documentation
class HelicityAngles(BaseModel):
    """Helicity angles for a specific isobar-bachelor pair"""
    phi: List[float] = Field(..., description="Azimuthal angles in radians")
    theta: List[float] = Field(..., description="Polar angles in radians")

class WignerAngles(BaseModel):
    """Wigner rotation angles between topologies"""
    phi_rf: List[float] = Field(..., description="Azimuthal Wigner angles in radians")
    theta_rf: List[float] = Field(..., description="Polar Wigner angles in radians")
    psi_rf: List[float] = Field(..., description="Third Euler angles in radians")

class TopologyResult(BaseModel):
    """Results for a single decay topology"""
    topology_name: str = Field(..., description="String representation of the topology")
    helicity_angles: Dict[str, HelicityAngles] = Field(..., description="Helicity angles for each isobar-bachelor pair")

class WignerRotationResult(BaseModel):
    """Wigner rotation results between two topologies"""
    from_topology: str = Field(..., description="Source topology")
    to_topology: str = Field(..., description="Target topology")
    particle_angles: Dict[str, WignerAngles] = Field(..., description="Wigner angles for each particle")

class DecayAnalysisResult(BaseModel):
    """Complete analysis results for Lc -> p K pi decay"""
    decay_info: Dict[str, Any] = Field(..., description="Decay configuration and metadata")
    phase_space_info: Dict[str, Any] = Field(..., description="Phase space generation information")
    topology_results: List[TopologyResult] = Field(..., description="Results for each decay topology")
    wigner_rotations: List[WignerRotationResult] = Field(..., description="Wigner rotations between topologies")
    analysis_metadata: Dict[str, Any] = Field(..., description="Analysis metadata and timestamps")

def numpy_to_list(data: np.ndarray) -> List[float]:
    """Convert numpy array to list of floats"""
    return data.tolist()

def save_results_to_json(helicity_angles: Dict, wigner_rotations: Dict, 
                        decay_info: Dict, phase_space_info: Dict, 
                        filename: str = "Lc_decay_analysis.json"):
    """Save analysis results to JSON file using Pydantic models"""
    
    # Convert topology results
    topology_results = []
    for topology, angles in helicity_angles.items():
        helicity_dict = {}
        for (isobar, bachelor), (phi, theta) in angles.items():
            key = f"{isobar}->{bachelor}"
            helicity_dict[key] = HelicityAngles(
                phi=numpy_to_list(phi),
                theta=numpy_to_list(theta)
            )
        
        topology_results.append(TopologyResult(
            topology_name=str(topology),
            helicity_angles=helicity_dict
        ))
    
    # Convert Wigner rotation results
    wigner_results = []
    for (top1, top2), rotations in wigner_rotations.items():
        particle_angles = {}
        for particle, angles in rotations.items():
            particle_angles[str(particle)] = WignerAngles(
                phi_rf=numpy_to_list(angles.phi_rf),
                theta_rf=numpy_to_list(angles.theta_rf),
                psi_rf=numpy_to_list(angles.psi_rf)
            )
        
        wigner_results.append(WignerRotationResult(
            from_topology=str(top1),
            to_topology=str(top2),
            particle_angles=particle_angles
        ))
    
    # Create complete result
    result = DecayAnalysisResult(
        decay_info=decay_info,
        phase_space_info=phase_space_info,
        topology_results=topology_results,
        wigner_rotations=wigner_results,
        analysis_metadata={
            "analysis_type": "Lc -> p K pi decay analysis",
            "software_versions": {
                "decayangle": "latest",
                "phasespace": "latest"
            },
            "configuration": {
                "sorting": "off",
                "n_events": phase_space_info.get("n_events", 0)
            }
        }
    )
    
    # Save to JSON
    with open(filename, 'w') as f:
        json.dump(result.dict(), f, indent=2)
    
    print(f"Results saved to {filename}")
    return result

def main():
    # Particle masses for Lc -> p K pi decay (in GeV)
    m_Lc = 2.28646  # Lc+ mass
    m_p = 0.93827   # proton mass  
    m_K = 0.49368   # K+ mass
    m_pi = 0.13957  # pi+ mass
    
    print("=== Lc -> p K pi Decay Analysis ===")
    print(f"Mother particle (Lc): {m_Lc:.5f} GeV")
    print(f"Daughter particles: p={m_p:.5f} GeV, K={m_K:.5f} GeV, π={m_pi:.5f} GeV")
    print()
    
    # Generate phase space events
    print("Generating phase space events...")
    n_events = 100
    weights, momenta_dict = phsp.nbody_decay(m_Lc, [m_p, m_K, m_pi]).generate(n_events, seed=42)

    # convert to numpy array
    momenta_dict = {k: np.array(v) for k, v in momenta_dict.items()}
    weights = np.array(weights)

    # Convert to decayangle format (particle indices: 1=p, 2=K, 3=pi)
    momenta = {
        1: np.array(momenta_dict["p_0"]),  # proton
        2: np.array(momenta_dict["p_1"]),  # K+
        3: np.array(momenta_dict["p_2"])   # pi+
    }
    
    print(f"Generated {n_events} events")
    print(f"Phase space weights range: [{weights.min():.6f}, {weights.max():.6f}]")
    print()
    
    # Store decay and phase space information for JSON output
    decay_info = {
        "mother_particle": {"name": "Lc+", "mass_gev": m_Lc},
        "daughter_particles": [
            {"name": "p", "mass_gev": m_p},
            {"name": "K+", "mass_gev": m_K},
            {"name": "pi+", "mass_gev": m_pi}
        ],
        "decay_channel": "Lc+ -> p K+ pi+"
    }
    
    phase_space_info = {
        "n_events": n_events,
        "weight_range": [float(weights.min()), float(weights.max())],
        "weight_mean": float(weights.mean()),
        "weight_std": float(weights.std()),
        "seed": 42
    }
    
    # Create topology collection for 3-body decay
    tg = TopologyCollection(0, [1, 2, 3])
    print("Available decay topologies:")
    for i, topology in enumerate(tg.topologies):
        print(f"  {i+1}. {topology}")
    print()
    
    # Transform momenta to rest frame of mother particle
    reference_topology = tg.topologies[0]
    momenta_rest = reference_topology.to_rest_frame(momenta)
    momenta_rest = reference_topology.align_with_daughter(momenta_rest)
    
    helicity_angles = {}
    for i, topology in enumerate(tg.topologies):
        angles = topology.helicity_angles(momenta_rest)
        helicity_angles[topology] = angles
    
    wigner_rotations = {}
    
    # Compute relative Wigner angles between all topology pairs
    for i, topology1 in enumerate(tg.topologies):
        for j, topology2 in enumerate(tg.topologies):
            if i != j:
                relative_angles = topology1.relative_wigner_angles(topology2, momenta_rest)
                wigner_rotations[(topology1, topology2)] = relative_angles
    
    # Save results to JSON
    print("\n=== Saving Results to JSON ===")
    result = save_results_to_json(helicity_angles, wigner_rotations, 
                                decay_info, phase_space_info)
    
    return helicity_angles, wigner_rotations, result



if __name__ == "__main__":
    helicity_angles, wigner_rotations, result = main()
    
    # Print summary of saved data
    print(f"\n=== Analysis Summary ===")
    print(f"Total topologies analyzed: {len(result.topology_results)}")
    print(f"Total Wigner rotation pairs: {len(result.wigner_rotations)}")
    print(f"JSON file contains structured data with Pydantic validation")
    print(f"Use result.dict() to access the complete data structure")
