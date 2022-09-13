import matplotlib.pyplot as plt
from datetime import datetime
import pyvisa as pvisa
import pandas as pd
import os
sds_addresses = ['TCPIP0::192.168.2.10::inst0::INSTR', 'TCPIP0::192.168.2.20::inst0::INSTR']
path = os.path.join("D:","1ScopeGrabs")



def main():
    count = 1
    for sds_address in sds_addresses:
        time_of_grab = datetime.now().strftime('%y-%m-%d-%H-%M-%S') + "("+str(count)+")"+".bmp"
        _rm = pvisa.ResourceManager()
        sds = _rm.open_resource(sds_address)
        files = os.path.join(path, time_of_grab)
        sds.write("SCDP")
        result_str = sds.read_raw()
        f = open(files,"wb")
        f.write(result_str)
        f.flush()
        f.close()
        count+=1

if __name__ == '__main__':
    main()
