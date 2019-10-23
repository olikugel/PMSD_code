import filehandling
import dataconversions
from plotting import print_dict as pd

BASEP = '/home/oliverschoppe/Documents/Synced/_Promotion/Projects/Leo/results/IC2dn2/' # path to 12TB HDD with all the scans

#%% Playground

prediction = filehandling.pload(BASEP + 'reviewed_prediction')
candidates = prediction['metastases']

potentially_TPs = dataconversions.filter_dicts(candidates,'evaluation-manually_confirmed',True)
surely_FPs = dataconversions.filter_dicts(candidates,'evaluation-manually_confirmed',False)