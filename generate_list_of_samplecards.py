import sys
import basepaths

CODEPATH, DATAPATH = basepaths.get_basepaths()

sys.path.insert(0, CODEPATH + '/helperfunctions')

import filehandling

#%%

global_count = 0

TP_count = 0
FP_count = 0
unclear_count = 0
no_file_count = 0

mice = ['H2030IC10dn573','IC2dn2','IC6dn1','IC6dn2', 'IC14dn1', 'MCF7IC21dn528', 'PC14dn04595']

LIST_OF_SAMPLECARDS = []


for mouse in mice:    
    decisions = filehandling.pload(DATAPATH + '/mice_metadata/' + mouse + '/reviewed_via_GUI.pickledump')
    for decision in decisions:
        SAMPLECARD = {}
        SAMPLECARD['sample_id'] = global_count
        SAMPLECARD['mouse'] = mouse
        SAMPLECARD['patch_id'] = decision['patch_id']
        SAMPLECARD['met_id'] = decision['id']
        try: 
            SAMPLECARD['label'] = decision['evaluation']['reviewed_via_GUI']
            if not (SAMPLECARD['label'] == 'true positive' or SAMPLECARD['label'] == 'false positive'):
                raise Exception('Should not be added to list because it is neither TP nor FP')   
        except: 
            continue
        LIST_OF_SAMPLECARDS.append(SAMPLECARD)
        global_count += 1
        
        if SAMPLECARD['label'] == 'true positive': 
            TP_count += 1
        elif SAMPLECARD['label'] == 'false positive':
            FP_count += 1
        elif SAMPLECARD['label'] == 'unclear':
            unclear_count += 1
        else:
            no_file_count += 1


filehandling.psave(DATAPATH + '/mice_metadata/' + 'list_of_samplecards.pickledump', LIST_OF_SAMPLECARDS)