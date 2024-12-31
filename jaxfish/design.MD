
function: run_simulation
this function will run one simulation with pre-defined max-length
it will need the following data:

 - terrain related, static
   - terrain_map: ndarray, (9, 9)
   - margin: int

 - simulation related, static
   - max_simulation_length: int
   - food_num: int, static, use 1 or 2
 
 - fish related, static
   - max_health: float, static  
   - food_rate: float, static
   - land_penalty_rate: float, static
   - health_decay_rate: float, static
   - move_penalty_rate: float, static
   - mother_name: str, static
   - name: str, static

 - brain related, static
   - rf_weights: ndarray, (n_eye, n_position), float, each row is an eye, each column is the weight of each receptive field pixel, static  
   - rf_positions: ndarray, (n_eye, n_position, 2), int, the last dimension is (row, col) of relative location of each receptive field pixel to fish center, static  
   - baselines: ndarray, (n_neuron), float, baseline firing rate, static
   - refractory_periods: ndarray, (n_neuron), float, static
   - connection_directions, (n_connections, 2), int, each row is (pre_neuron_index, pos_neuron_index), static
   - connection_weights, (n_connections,), float, static
   - connection_waveforms, (n_connections, 3), int, each row is (latency, rise_time, decay_time), static
 
 - Dynamic during simulation
   - food_positions: 2d array, (length, food_num, 2), int, each row is the (row, col) of a food pellet.
   - health_history: ndarray, (length,)
   - position_history: ndarray, (length, 2), each row is the (row, col) of fish center at a given time point
   - firing_histories: ndarray, (n_neuron, length), int/bool, each row is a neuron, indicating if the neuron has fired at a given time point
   - psp_waveforms: ndarray, (n_neuron, length)
 

functions:

 - generate_terrain_minimap(shape:List[int]=[9, 9], food_num:int=1) -> ndarray:

 - create_simulation_cache(neurons, connections, )

