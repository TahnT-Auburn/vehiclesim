def calc_error_statics(truth_states:list[list], filter_states:list[list]) -> list[list]:
    """Calculates the error statics using a filter's outputs and the truth
    
    Args:
        truth_states (list[list]): Truth states with state labels.
        filter_states (list[list]): Filter states with state labels.
    Returns:
        state_errors (list[list]): A list of errors for each state
    """
    assert len(truth_states) == len(filter_states),\
        f'Lengths of inputs must match. truth_states has length {len(truth_states)} and filter_staes has length {len(filter_states)}.'
    
    state_errors = []
    for truth_state, filter_state in zip(truth_states, filter_states):
        label = filter_state[1]
        unit = filter_state[2]
        state_error = filter_state[0] - truth_state[0]
        state_errors.append([state_error, label, unit])
        
    return state_errors
    