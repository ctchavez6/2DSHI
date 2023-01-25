import matplotlib.pyplot as plt
from datetime import datetime
import pyvisa as pvisa
import pandas as pd
import os
from playsound import playsound # import required module
# sds_addresses = ['TCPIP0::192.168.2.10::inst0::INSTR', 'TCPIP0::192.168.2.20::inst0::INSTR', 'TCPIP0::192.168.2.30::inst0::INSTR']
# sds_addresses = ['TCPIP0::192.168.2.30::inst0::INSTR']
# sds_addresses = ['TCPIP0::192.168.2.20::inst0::INSTR', 'TCPIP0::192.168.2.30::inst0::INSTR']
# sds_addresses = ['TCPIP0::10.0.0.5::inst0::INSTR', 'TCPIP0::10.0.0.6::inst0::INSTR', 'TCPIP0::10.0.0.7::inst0::INSTR']
sds_addresses = ['TCPIP0::192.168.2.20::inst0::INSTR', 'TCPIP0::192.168.2.30::inst0::INSTR']

path = os.path.join("C:/Users/Zap/Desktop","ScopeGrabs")






def main():

    for sds_address in sds_addresses:
        time_of_grab_a = datetime.now().strftime('%y-%m-%d-%H-%M-%S')
        time_of_grab_b = time_of_grab_a + ".bmp"

        channel_data = {
            "1": {"time": None, "voltage": None},
            "2": {"time": None, "voltage": None},
            "3": {"time": None, "voltage": None},
            "4": {"time": None, "voltage": None}
        }

        for channel in channel_data.keys():
            _rm = pvisa.ResourceManager()
            sds = _rm.open_resource(sds_address)
            sds.open()
            sds.write("chdr off")
            vdiv = sds.query("c" + channel + ":vdiv?")
            ofst = sds.query("c" + channel + ":ofst?")
            tdiv = sds.query("tdiv?")
            sara = sds.query("sara?")

            sara_unit = {'G': 1E9, 'M': 1E6, 'k': 1E3}
            for unit in sara_unit.keys():
                if sara.find(unit) != -1:
                    sara = sara.split(unit)
                    sara = float(sara[0]) * sara_unit[unit]
                    break
            sara = float(sara)
            sds.timeout = 30000  # default value is 2000(2s)
            sds.chunk_size = 20 * 1024 * 1024  # default value is 20*1024(20k bytes)
            sds.write("c" + channel + ":wf? dat2")
            recv = list(sds.read_raw())[15:]
            recv.pop()
            recv.pop()

            volt_value = []
            for data in recv:
                if data > 127:
                    data = data - 256
                else:
                    pass
                volt_value.append(data)
            channel_data[channel]["voltage"] = volt_value

            time_value = []

            for idx in range(0, len(volt_value)):
                volt_value[idx] = volt_value[idx] / 25 * float(vdiv) - float(ofst)
                time_data = -(float(tdiv) * 14 / 2) + idx * (1 / sara)
                time_value.append(time_data)

            channel_data[channel]["time"] = time_value

        df_as_dict = {
            "ch1_time": channel_data["1"]["time"][1:],
            "ch1_voltage": channel_data["1"]["voltage"][1:],
            #"ch2_time": channel_data["2"]["time"],
            "ch2_voltage": channel_data["2"]["voltage"][1:],
            #"ch3_time": channel_data["3"]["time"],
            "ch3_voltage": channel_data["3"]["voltage"][1:],
            #"ch4_time": channel_data["4"]["time"],
            "ch4_voltage": channel_data["4"]["voltage"][1:],

        }
        df = pd.DataFrame(df_as_dict)
        df.to_csv(os.path.join(path,"4chnl_%s.csv" %time_of_grab_a))

        files = os.path.join(path, time_of_grab_b)
        sds.write("SCDP")
        result_str = sds.read_raw()
        f = open(files,"wb")
        f.write(result_str)
        f.flush()
        f.close()
    playsound('C:/Windows/Media/tada.wav')
    quit()

main()
