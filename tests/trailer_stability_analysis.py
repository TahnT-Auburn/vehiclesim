#%%
import numpy as np
import matplotlib
matplotlib.use('ipympl')
import matplotlib.pyplot as plt
# import matlab.engine

from vehiclesim.tractor_trailer import TractorTrailer
from trailer_stability_data_loader import genData
from filter_tools.estimators import Estimators

#%%
# call generate data function
data_dir = 'C:\\Users\\pzt0029\\Documents\\Vehicle_Simulations\\vehiclesim\\tests\data\\tire_test'
dir_dict = genData(data_dir)

#%%
# Perform analysis
def rmsd(x_ref, x_est):
    return np.sqrt(np.mean((x_ref - x_est)**2))

# roll angle RMSD from baseline
roll_ref = dir_dict['full_nl'].Roll_2
roll_rmsd_full_nl = rmsd(roll_ref, dir_dict['full_nl'].Roll_2)
roll_rmsd_full_5k = rmsd(roll_ref, dir_dict['full_5k'].Roll_2)
roll_rmsd_full_10k = rmsd(roll_ref, dir_dict['full_10k'].Roll_2)
roll_rmsd_full_20k = rmsd(roll_ref, dir_dict['full_20k'].Roll_2)

roll_rmsd_hw_nl = rmsd(roll_ref, dir_dict['hw_nl'].Roll_2)
roll_rmsd_hw_5k = rmsd(roll_ref, dir_dict['hw_5k'].Roll_2)
roll_rmsd_hw_10k = rmsd(roll_ref, dir_dict['hw_10k'].Roll_2)
roll_rmsd_hw_20k = rmsd(roll_ref, dir_dict['hw_20k'].Roll_2)

roll_rmsd_fw_nl = rmsd(roll_ref, dir_dict['fw_nl'].Roll_2)
roll_rmsd_fw_5k = rmsd(roll_ref, dir_dict['fw_5k'].Roll_2)
roll_rmsd_fw_10k = rmsd(roll_ref, dir_dict['fw_10k'].Roll_2)
roll_rmsd_fw_20k = rmsd(roll_ref, dir_dict['fw_20k'].Roll_2)

payloads = ['0', '5K', '10K', '20K']
full_roll = [roll_rmsd_full_nl, roll_rmsd_full_5k, roll_rmsd_full_10k, roll_rmsd_full_20k]
hw_roll = [roll_rmsd_hw_nl, roll_rmsd_hw_5k, roll_rmsd_hw_10k, roll_rmsd_hw_20k]
fw_roll = [roll_rmsd_fw_nl, roll_rmsd_fw_5k, roll_rmsd_fw_10k, roll_rmsd_fw_20k]

# roll rate RMSD from baseline
rr_ref = dir_dict['full_nl'].AVx_2
rr_rmsd_full_nl = rmsd(rr_ref, dir_dict['full_nl'].AVx_2)
rr_rmsd_full_5k = rmsd(rr_ref, dir_dict['full_5k'].AVx_2)
rr_rmsd_full_10k = rmsd(rr_ref, dir_dict['full_10k'].AVx_2)
rr_rmsd_full_20k = rmsd(rr_ref, dir_dict['full_20k'].AVx_2)

rr_rmsd_hw_nl = rmsd(rr_ref, dir_dict['hw_nl'].AVx_2)
rr_rmsd_hw_5k = rmsd(rr_ref, dir_dict['hw_5k'].AVx_2)
rr_rmsd_hw_10k = rmsd(rr_ref, dir_dict['hw_10k'].AVx_2)
rr_rmsd_hw_20k = rmsd(rr_ref, dir_dict['hw_20k'].AVx_2)

rr_rmsd_fw_nl = rmsd(rr_ref, dir_dict['fw_nl'].AVx_2)
rr_rmsd_fw_5k = rmsd(rr_ref, dir_dict['fw_5k'].AVx_2)
rr_rmsd_fw_10k = rmsd(rr_ref, dir_dict['fw_10k'].AVx_2)
rr_rmsd_fw_20k = rmsd(rr_ref, dir_dict['fw_20k'].AVx_2)

full_rr = [rr_rmsd_full_nl, rr_rmsd_full_5k, rr_rmsd_full_10k, rr_rmsd_full_20k]
hw_rr = [rr_rmsd_hw_nl, rr_rmsd_hw_5k, rr_rmsd_hw_10k, rr_rmsd_hw_20k]
fw_rr = [rr_rmsd_fw_nl, rr_rmsd_fw_5k, rr_rmsd_fw_10k, rr_rmsd_fw_20k]

# Hitch angle RMSD from baseline
hitch_ref = dir_dict['full_nl'].Art_H
hitch_rmsd_full_nl = rmsd(hitch_ref, dir_dict['full_nl'].Art_H)
hitch_rmsd_full_5k = rmsd(hitch_ref, dir_dict['full_5k'].Art_H)
hitch_rmsd_full_10k = rmsd(hitch_ref, dir_dict['full_10k'].Art_H)
hitch_rmsd_full_20k = rmsd(hitch_ref, dir_dict['full_20k'].Art_H)

hitch_rmsd_hw_nl = rmsd(hitch_ref, dir_dict['hw_nl'].Art_H)
hitch_rmsd_hw_5k = rmsd(hitch_ref, dir_dict['hw_5k'].Art_H)
hitch_rmsd_hw_10k = rmsd(hitch_ref, dir_dict['hw_10k'].Art_H)
hitch_rmsd_hw_20k = rmsd(hitch_ref, dir_dict['hw_20k'].Art_H)

hitch_rmsd_fw_nl = rmsd(hitch_ref, dir_dict['fw_nl'].Art_H)
hitch_rmsd_fw_5k = rmsd(hitch_ref, dir_dict['fw_5k'].Art_H)
hitch_rmsd_fw_10k = rmsd(hitch_ref, dir_dict['fw_10k'].Art_H)
hitch_rmsd_fw_20k = rmsd(hitch_ref, dir_dict['fw_20k'].Art_H)

full_hitch = [hitch_rmsd_full_nl, hitch_rmsd_full_5k, hitch_rmsd_full_10k, hitch_rmsd_full_20k]
hw_hitch = [hitch_rmsd_hw_nl, hitch_rmsd_hw_5k, hitch_rmsd_hw_10k, hitch_rmsd_hw_20k]
fw_hitch = [hitch_rmsd_fw_nl, hitch_rmsd_fw_5k, hitch_rmsd_fw_10k, hitch_rmsd_fw_20k]

# Hitch rate RMSD from baseline
hr_ref = dir_dict['full_nl'].ArtR_H
hr_rmsd_full_nl = rmsd(hr_ref, dir_dict['full_nl'].ArtR_H)
hr_rmsd_full_5k = rmsd(hr_ref, dir_dict['full_5k'].ArtR_H)
hr_rmsd_full_10k = rmsd(hr_ref, dir_dict['full_10k'].ArtR_H)
hr_rmsd_full_20k = rmsd(hr_ref, dir_dict['full_20k'].ArtR_H)

hr_rmsd_hw_nl = rmsd(hr_ref, dir_dict['hw_nl'].ArtR_H)
hr_rmsd_hw_5k = rmsd(hr_ref, dir_dict['hw_5k'].ArtR_H)
hr_rmsd_hw_10k = rmsd(hr_ref, dir_dict['hw_10k'].ArtR_H)
hr_rmsd_hw_20k = rmsd(hr_ref, dir_dict['hw_20k'].ArtR_H)

hr_rmsd_fw_nl = rmsd(hr_ref, dir_dict['fw_nl'].ArtR_H)
hr_rmsd_fw_5k = rmsd(hr_ref, dir_dict['fw_5k'].ArtR_H)
hr_rmsd_fw_10k = rmsd(hr_ref, dir_dict['fw_10k'].ArtR_H)
hr_rmsd_fw_20k = rmsd(hr_ref, dir_dict['fw_20k'].ArtR_H)

full_hr = [hr_rmsd_full_nl, hr_rmsd_full_5k, hr_rmsd_full_10k, hr_rmsd_full_20k]
hw_hr = [hr_rmsd_hw_nl, hr_rmsd_hw_5k, hr_rmsd_hw_10k, hr_rmsd_hw_20k]
fw_hr = [hr_rmsd_fw_nl, hr_rmsd_fw_5k, hr_rmsd_fw_10k, hr_rmsd_fw_20k]

#%%
# # plots
# plot roll/roll rate RMSD curves
ax1 = plt.subplot(211)
ax1.plot(payloads,full_roll, '-o', c='cyan', label='Normal')
ax1.plot(payloads,hw_roll, '-o', c='k', label='Half Worn')
ax1.plot(payloads,fw_roll, '-o', c='red', label='Full Worn')
ax1.legend()
# ax1.set_title('RMSD From Baseline')
# ax1.set_xlabel('Payload')
ax1.set_ylabel('Roll Angle RMSD (Deg)')
ax1.tick_params(axis='x',labelsize=13)
ax1.tick_params(axis='y',labelsize=13)
ax1.grid()
ax2 = plt.subplot(212)
ax2.plot(payloads,full_rr, '-o', c='cyan', label='Normal')
ax2.plot(payloads,hw_rr, '-o', c='k', label='Half Worn')
ax2.plot(payloads,fw_rr, '-o', c='red', label='Full Worn')
# ax2.legend()
ax2.set_xlabel('Payload (kg)')
ax2.set_ylabel('Roll Rate RMSD (Deg/s)')
ax2.tick_params(axis='x',labelsize=13)
ax2.tick_params(axis='y',labelsize=13)
ax2.grid()
plt.tight_layout()
plt.show()

# plot roll/roll rate RMSD curves
ax1 = plt.subplot(211)
ax1.plot(payloads,full_hitch, '-o', c='cyan', label='Normal')
ax1.plot(payloads,hw_hitch, '-o', c='k', label='Half Worn')
ax1.plot(payloads,fw_hitch, '-o', c='red', label='Full Worn')
ax1.legend()
# ax1.set_title('RMSD From Baseline')
# ax1.set_xlabel('Payload')
ax1.set_ylabel('Hitch Angle RMSD (Deg)')
ax1.tick_params(axis='x',labelsize=13)
ax1.tick_params(axis='y',labelsize=13)
ax1.grid()
ax2 = plt.subplot(212)
ax2.plot(payloads,full_hr, '-o', c='cyan', label='Normal')
ax2.plot(payloads,hw_hr, '-o', c='k', label='Half Worn')
ax2.plot(payloads,fw_hr, '-o', c='red', label='Full Worn')
# ax2.legend()
ax2.set_xlabel('Payload (kg)')
ax2.set_ylabel('Hitch Rate RMSD (Deg/s)')
ax2.tick_params(axis='x',labelsize=13)
ax2.tick_params(axis='y',labelsize=13)
ax2.grid()
plt.tight_layout()
plt.show()
#%%
