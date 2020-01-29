import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "..")) 
import basepaths
CODEPATH, DATAPATH = basepaths.get_basepaths()

sys.path.insert(0, CODEPATH + '/helperfunctions')
import filehandling
from filehandling import pad_ID

import csv



with open('falsely_predicted_samples.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')

    for line in csv_reader:
        if not line: 
            continue
        elif 'Cuda' in line[0]: 
            continue
        elif 'Mouse' in line[0]:
            continue
        else:
            mouse    = line[0].strip()
            patch_ID = int(line[1].strip())
            met_ID   = int(line[2].strip())

            filepath_C00 = DATAPATH + '/Potential_TP_Metastases/' + mouse + '/C00/ROI_50/PNG/'
            filepath_C01 = DATAPATH + '/Potential_TP_Metastases/' + mouse + '/C01/ROI_50/PNG/'
            filepath_C02 = DATAPATH + '/Potential_TP_Metastases/' + mouse + '/C02/ROI_50/PNG/'
            filename_x = 'patch' + pad_ID(patch_ID) + '_met' + pad_ID(met_ID) + '_x.png'
            filename_z = 'patch' + pad_ID(patch_ID) + '_met' + pad_ID(met_ID) + '_z.png'
            
            print(filepath_C00 + filename_x)
            print(filepath_C00 + filename_z)
            print(filepath_C01 + filename_x)
            print(filepath_C01 + filename_z)
            print(filepath_C02 + filename_x)
            print(filepath_C02 + filename_z)


# then in bash:
# for file in $(<falsely_predicted.txt); do cp "$file" some_folder; done


