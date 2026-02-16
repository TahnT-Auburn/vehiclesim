import numpy as np
import copy
from vehiclesim.vehicle_configs.veh_params import vp

def perturb_parameters(nominal_params, percentage, distribution='uniform', seed=None):
    """
    Create a perturbed copy of the nominal parameters.
    
    Args:
        nominal_params: Dictionary of nominal parameter values
        percentage: Percentage variation (e.g., 0.20 for ±20% variation)
        distribution: 'normal' or 'uniform'
        seed: Random seed for reproducibility (optional)
        
    Returns:
        Dictionary with perturbed parameters
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Create a deep copy to avoid modifying the original
    perturbed = copy.deepcopy(nominal_params)
    
    for param_name, nominal_value in nominal_params.items():
        # Handle list parameters (like cs, n_t)
        if isinstance(nominal_value, list):
            perturbed[param_name] = [
                sample_value(nom, percentage * abs(nom), distribution)
                for nom in nominal_value
            ]
        else:
            # Scalar parameter
            sigma = percentage * abs(nominal_value)
            perturbed[param_name] = sample_value(
                nominal_value, sigma, distribution
            )
    
    return perturbed


def sample_value(nominal, sigma, distribution):
    """
    Sample a new value given nominal and sigma.
    
    Args:
        nominal: Nominal/base value
        sigma: Standard deviation or range
        distribution: 'normal' or 'uniform'
        
    Returns:
        Sampled value
    """
    if distribution == 'normal':
        return nominal + np.random.normal(0, sigma)
    elif distribution == 'uniform':
        # Uniform distribution ±sigma
        return np.random.uniform(nominal - sigma, nominal + sigma)
    else:
        raise ValueError(f"Unknown distribution: {distribution}")


# Example usage:
if __name__ == "__main__":
    # Simple one-line usage:
    perturbed_vp = perturb_parameters(vp, percentage=0.20, seed=42)
