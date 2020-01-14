import tkinter as tk
from datetime import datetime
import time
import threading
now = datetime.now()
later = datetime.now()
difference = later-now
difference = difference.seconds


class App(threading.Thread):

   def __init__(self):
      threading.Thread.__init__(self)
      self.start()
      self.foo = 1

   def callback(self):
      self.root.quit()

   def scale_onChange(self, value):
      self.foo = int(value)


   def run(self):
      self.root = tk.Tk()
      self.root.protocol("WM_DELETE_WINDOW", self.callback)

      label = tk.Label(self.root, text="Hello World")
      label.pack()

      scale = tk.Scale(from_=1, to=4, tickinterval=0.5, orient=tk.HORIZONTAL, command=self.scale_onChange)
      scale.pack()

      #self.foo = label
      self.root.mainloop()


app = App()
#app.run()
print('Now we can continue running code while mainloop runs!')
#app.run()

while difference < 5:

   time.sleep(0.5)
   later = datetime.now()
   difference = later - now
   difference = difference.seconds
   print(app.foo)
   print(difference)


app.callback()
"""


mainloop_pressed = True
root.mainloop()
root = tk.Tk()



   def sel():
      selection = "Value = " + str(var.get())
      label.config(text = selection)

   tk.Button(root, text="Quit", command=root.destroy).pack()
   var = tk.DoubleVar()
   scale = tk.Scale( root, variable = var )
   scale.pack(anchor = tk.CENTER)

   button = tk.Button(root, text = "Get Scale Value", command = sel)
   button.pack(anchor = tk.CENTER)

   label = tk.Label(root)
   label.pack()



"""

