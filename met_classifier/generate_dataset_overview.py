CODEPATH = '/home/okugel/PMSD/PMSD_code'
DATAPATH = '/home/okugel/PMSD/PMSD_data'
import sys
sys.path.insert(0, CODEPATH + '/helperfunctions')
import filehandling


samplecards = filehandling.pload(DATAPATH + '/mice_metadata/' + 'list_of_samplecards.pickledump')


TP_count = 0
FP_count = 0

for samplecard in samplecards:

    print('Mouse:', samplecard['mouse'], ', Patch:', samplecard['patch_id'], ', Met:', samplecard['met_id'], ', Label:', samplecard['label'])

    if samplecard['label'] == 'true positive':
        TP_count += 1
    else:
        FP_count += 1


print()
print('Total number of samples:',len(samplecards))
print('Number of true positives:',TP_count)
print('Number of false positives:',FP_count)
