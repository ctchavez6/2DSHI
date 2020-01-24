from tkinter import Tk
import sys
from tkinter.filedialog import askopenfilename, askopenfilenames

root = Tk()
Tk().withdraw() # we don't want a full GUI, so keep the root window from appearing

groups = dict()
satisfaction = False
while not satisfaction:
    group_name = input("Please select a name for the current grouping: ")
    _ = input("Press Enter to proceed to pick all your r_matrices (or q to Quit)")
    filez = askopenfilenames(parent=root,title='Choose a file')
    group = root.tk.splitlist(filez)
    print(type(group))
    print(group)


