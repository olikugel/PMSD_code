from uuid import getnode as get_mac

def get_basepath(printInfo=False):
    mac_address = get_mac()
    mac_address = str(hex(mac_address)).replace('0x','')
    if printInfo: print('\nMAC address of local machine: ' + mac_address)
    
    if mac_address == 'c4b301d23369': # mac address of OlisLaptop
        BASEPATH = '/Users/olikugel/Git/OlisIDP/' # base path at OlisLaptop
        if printInfo: print('---> running on OlisLaptop')
    else:
        BASEPATH = '/home/olikugel/OlisIDP/' # base path at TranslaTUM
        if printInfo: print ('---> running on TranslaTUM machine')

    if printInfo: print('---> base path was set accordingly')
    return BASEPATH