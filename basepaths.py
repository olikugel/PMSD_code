import sys
import subprocess


def get_basepaths():

    bashCommand = "hostnamectl"
    process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
    output, error = process.communicate()
    output = str(output)
    output = output.replace("b'", "").replace("'", "")
    lines = output.split('\\n')
    for line in lines:
        if "Machine ID" in line:
            machine_ID = line.strip().replace("Machine ID: ", "")
            print("\nMachine:\t", machine_ID)
    
    if machine_ID == '83011c94751c4629a5d4c426051b3041':   # running on translatum machine
        CODEPATH = '/home/olikugel/PMSD/PMSD_code'
        DATAPATH = '/home/olikugel/PMSD/PMSD_data'  
    elif machine_ID == '14bef0a2877e464f877ca6ab665b81e8': # running on remote server
        CODEPATH = '/home/oschoppe/Documents/OKugel/PMSD_code'
        DATAPATH = '/home/oschoppe/Documents/OKugel/PMSD_data'  
    elif machine_ID == '3e372380154944dc87b8154c932a2624': # running on OlisLaptop
        CODEPATH = '/home/olikugel/Coding/PMSD/PMSD_code'
        DATAPATH = '/home/olikugel/Coding/PMSD/PMSD_data'  
    else:
        sys.exit('Running on unknown machine. Cannot set basepaths. Exiting.')

    return CODEPATH, DATAPATH
    
