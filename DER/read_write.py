"""
read_write.py
Functions to Read and Write Simulations as pickle files
"""

import pickle
import os
import re
import datetime

from DER.simulation import Simulation


def read(filepath: str) -> Simulation:
    """Write a simulation to a file"""
    with open(filepath, 'rb') as f:
        simulation = pickle.load(f)
    return simulation

load = read

def write(
    simulation: Simulation, 
    filepath: str, 
    create_path_if_not_exists:bool = True,
    add_time_stamp:bool = False
    ):
    """Write a simulation to a file"""
    # Create the path if it does not exist and enabled
    if create_path_if_not_exists:
        if not os.path.exists(os.path.dirname(filepath)):
            os.mkdir(os.path.dirname(filepath))
    # Add the time stamp if enabled
    if add_time_stamp:
        filepath = os.path.splitext(os.path.abspath(filepath))
        time_stamp = str(re.sub(r"\D", "-", datetime.datetime.now().isoformat(timespec='seconds')))
        filepath = filepath[0] + "-" + time_stamp + filepath[1]
    # Wire the file
    with open(filepath, 'wb') as f:
        pickle.dump(simulation, f)
    # print(f"Saved at '{filepath}'")
