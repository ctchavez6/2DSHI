import tkinter as tk
import threading

class App(threading.Thread):

    def __init__(self):
        threading.Thread.__init__(self)
        self.start()
        self.foo = 1
        self.at_front = False

    def callback(self):
        self.root.quit()

    def scale_onChange(self, value):
        self.foo = float(value)


    def run(self):
        self.root = tk.Tk()
        self.root.protocol("WM_DELETE_WINDOW", self.callback)

        label = tk.Label(self.root, text="Sigma")
        label.pack()

        scale = tk.Scale(from_=1.00, to=2.50, tickinterval=0.0001, resolution = 0.25, digits = 3,orient=tk.HORIZONTAL, command=self.scale_onChange).pack()
        #scale.pack()

        #self.foo = label
        self.root.mainloop()

    def bring_to_front(self):
        self.root.lift()
        self.at_front = True