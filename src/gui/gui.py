from tkinter.filedialog import askopenfilenames, askdirectory
from tkinter import ttk
from tkinter import Tk, StringVar
import threading
import os
from tkinter import TOP, NW

os.environ['TK_SILENCE_DEPRECATION'] = '1'


def get_latest_run_direc(path_override=False, path_to_exclude=None):
    if not path_override:
        data_directory = os.path.join(os.path.join(os.getcwd(), 'test'), 'D')  # Test 'D' Drive
        # data_directory = os.path.join("D:")  # Windows PC @ Franks' House
    else:
        data_directory = os.path.abspath(os.path.join(os.path.join("D:"), os.pardir))

    all_runs_lst = sorted([os.path.join(data_directory, path) for path in os.listdir(data_directory)
                       if os.path.isdir(os.path.join(data_directory, path))
                       and path not in ["$RECYCLE.BIN",
                                        "System Volume Information",
                                        "BaslerCameraData",
                                        ".tmp.drivedownload",
                                        "Recovery"]])

    all_runs = {os.path.basename(os.path.normpath(run)): run for run in all_runs_lst}

    return all_runs


class MyFirstGUI:

    def __init__(self, master, width=500, height=500):
        self.master = master
        self.master.lift()
        self.master.attributes("-topmost", True)
        self.master.configure(background='#EAEDED')
        self.runs = get_latest_run_direc()
        self.options = sorted(list(self.runs.keys()), reverse=True)
        self.master.title("2D SHI")
        # get screen width and height
        screen_width = self.master.winfo_screenwidth()  # width of the screen
        screen_height = self.master.winfo_screenheight()  # height of the screen

        # calculate x and y coordinates for the Tk root window
        x = 0  # (screen_width / 2) - (width / 2)
        y = 0  # (screen_height / 2) - (width / 2)

        # set the dimensions of the screen
        # and where it is placed
        #self.master.geometry('%dx%d+%d+%d' % (width, height, x, y))
        #self.label = tk.Label(self.master, text="2D Simple Harmonic Interferometer!")

        #self.label.pack()

        self.greet_button = ttk.Button(self.master, text="Greet", command=self.greet)
        self.greet_button.grid(row=0, column=0)

        self.close_button = ttk.Button(self.master, text="Close", command=self.master.quit)
        self.close_button.grid(row=0, column=2)

        self.run_selected_base_path = StringVar(self.master)
        self.run_selected_base_path.set(self.options[-1])  # default value
        self.run_selected_full_path = self.runs[self.run_selected_base_path.get()]

        self.select_run_directory = ttk.OptionMenu(self.master, self.run_selected_base_path, *self.options)
        self.select_run_directory.grid(row=0, column=3)

        v = StringVar(root, value='default text')
        run_name = ttk.Entry(root, textvariable=v)
        run_name.grid(row=1, column=0)

    def chose_run_by_file_explorer(self):
        pass

    def greet(self):
        user_choice_full_path = self.different_file()
        self.run_selected_base_path.set(os.path.basename(os.path.normpath(user_choice_full_path)))

    def different_file(self):
        return askdirectory(title='Choose a directory')



root = Tk()
my_gui = MyFirstGUI(root)
root.mainloop()
