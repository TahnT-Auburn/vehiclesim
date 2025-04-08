#%%
import numpy as np
import os
from vehiclesim.tractor_trailer import TractorTrailer

#%%
def genData(directory_path):
    # decalre utility instance to help load trucksim data
    veh_config_file = 'C:\\Users\\pzt0029\\Documents\\Vehicle_Simulations\\vehiclesim\\tests\\veh_config\\tractor_trailer\\5a_config.yaml'
    tt_util = TractorTrailer(veh_config_file=veh_config_file)
    dirs, filenames = fileLoop(directory_path)
    dir_dict = {}
    for idx, name in enumerate(filenames):
            dir_dict[name] = tt_util.genTsData(dirs[idx])

    return dir_dict

def fileLoop(directory_path):
    items = os.listdir(directory_path)
    dirs = []
    filenames = []
    for item in items:
        # construct full path to item and populate list of directories
        item_path = os.path.join(directory_path,item)
        dirs.append(item_path)
        # parse filenames
        filename = os.path.basename(item_path)
        filename_no_ext = os.path.splitext(filename)[0]
        filenames.append(filename_no_ext)

    return dirs, filenames