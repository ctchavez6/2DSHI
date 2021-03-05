import matplotlib.pyplot as plt
from datetime import datetime
import pyvisa as pvisa
import pandas as pd
sds_addresses = ['TCPIP0::192.168.2.10::inst0::INSTR', 'TCPIP0::192.168.2.20::inst0::INSTR']



def main():

    for sds_address in sds_addresses:

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
        df.to_csv("4chnl_%s.csv" %datetime.now().strftime('%y-%m-%d-%H-%M-%S'))

        x = df_as_dict['ch1_time']
        y4 = df_as_dict['ch4_voltage']
        y3 = df_as_dict['ch3_voltage']
        y2 = df_as_dict['ch2_voltage']
        y1 = df_as_dict['ch1_voltage']

        figure, axis = plt.subplots(2,2)
        plot = False
        scatter = False if plot else True

        if plot:
            axis[0, 0].plot(x,y1)
            axis[0, 0].set_title("Channel 1")
            axis[0, 1].plot(x, y2)
            axis[0, 1].set_title("Channel 2")
            axis[1, 0].plot(x, y3)
            axis[1, 0].set_title("Channel 3")
            axis[1, 1].plot(x, y4)
            axis[1, 1].set_title("Channel 4")
            plt.tight_layout(h_pad=5, w_pad=5)
        elif scatter:
            axis[0, 0].scatter(x,y1, s=.01)
            axis[0, 0].set_title("Channel 1")
            axis[0, 1].scatter(x, y2, s=.01)
            axis[0, 1].set_title("Channel 2")
            axis[1, 0].scatter(x, y3, s=.01)
            axis[1, 0].set_title("Channel 3")
            axis[1, 1].scatter(x, y4, s=.01)
            axis[1, 1].set_title("Channel 4")
            plt.tight_layout(h_pad=5, w_pad=5)


        plt.savefig("4chnl_%s_plot" %datetime.now().strftime('%y-%m-%d-%H-%M-%S'))


if __name__ == '__main__':
    main()
