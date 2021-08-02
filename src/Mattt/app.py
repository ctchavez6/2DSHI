from tkinter import *

class App():
    def __init__(self):
        self.root = Tk(className="button_click_label")
        #self.l1 = Label(self.root, text="hi")



    #def press(self):
        #self.l1.config(text="hello")

    def select_calibration_directory(self):
        self.calibration_directory, self.alpha, self.v = ccc.characterize()
        print(self.calibration_directory, self.alpha, self.v)
        self.calibration_directory_label.config(text="hellllo")

    def run(self):
        self.root.geometry("200x200")

        #message = StringVar()
        #message.set('hi')

        #b1 = Button(self.root, text="clickhere", command=self.press).pack()

        #self.l1.pack()

        self.root.mainloop()
