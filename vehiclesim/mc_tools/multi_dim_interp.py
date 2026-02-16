import numpy as np
from numpy.typing import NDArray

def multi_dim_interp(
        mc_data:NDArray,
        target_len:int,
):
    """
    Interpolates a 1D slice of data along all other dimensions.
    Use case is for matching monte carlo data with different sequence lengths
    (e.g, filter MCs at 40Hz and VIO only MCs at 10Hz).

    Args:
        mc_data (NDArray): The monte carlo data to be interpolated. Assumes shape (N, L_MC, L), where L is the dimension to be interpolated.
        target_len (int): The target length such that the output data returns (N, L_MC, L_new).

        Returns:
            interp_mc_data (NDArray): The interpolated MC data with shape (N, L_MC, L_new).
    """
    L_new = target_len
    L_orig = mc_data.shape[2]

    def interp_1d_slice(slice_1d, L_orig, L_new):
        """Interpolates a 1D slice to a new length."""
        # Define original and new x-coordinates for interpolation
        xp = np.arange(L_orig)
        x = np.linspace(0, L_orig - 1, L_new)
        # Perform linear interpolation
        return np.interp(x, xp, slice_1d)
    
    interp_mc_data = np.apply_along_axis(
        interp_1d_slice,
        axis=2,
        arr=mc_data,
        L_orig=L_orig,
        L_new=L_new
    )

    return interp_mc_data