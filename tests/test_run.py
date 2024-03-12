#%%
from vehiclesim.tractor_trailer import TractorTrailer
from python_utilities.plotters_class import Plotters
import matplotlib.pyplot as plt

if __name__ == '__main__':

    test_instance = TractorTrailer(ts_data_file='data/Run103.csv')
    ts_data = test_instance.ts_data

    x = ts_data.Time
    y = ts_data.AVx_L1
    y2 = ts_data.AVx_L2
    y3 = ts_data.AVx_R1
    plot = Plotters()
    plot.matPlot(sig_count='multi', signals=[[x,y], [x,y2], [x,y3]],\
            label=[y.name, x.name, y.name])
    