import tkinter as tk
import threading
import traceback
import tkinter.ttk as ttk


class App(threading.Thread):

    def __init__(self):
        threading.Thread.__init__(self)
        self.start()
        self.foo = 1
        self.at_front = False
        self.is_toggle_pressed_down = False
        self.stop_streaming_override = False
        self.sub_sigma = 0.20
        self.ofset_delta = 5
        self.horizontal_offset = 0
        self.vertical_offset = 0


    def callback(self):
        self.root.quit()

    def destroy(self):
        self.root.destroy()

    def scale_onChange(self, value):
        self.foo = float(value)

    def scale_onChange_b(self, value):
        self.sub_sigma = float(value)

    def toggle(self):
        if self.stop_streaming_override is True:
                #self.toggle_btn.config('relief')[-1] == 'sunken':
            self.stop_streaming_override = False
            #self.toggle_btn.config(relief="raised")
            #print("self.is_toggle_pressed_down: ", self.is_toggle_pressed_down)
        else:
            #self.toggle_btn.config(relief="sunken")
            self.stop_streaming_override = True
            #print("self.is_toggle_pressed_down: ", self.is_toggle_pressed_down)

    def move_roi_left(self):
        self.horizontal_offset += self.ofset_delta

    def move_roi_right(self):
        self.horizontal_offset -= self.ofset_delta

    def move_roi_up(self):
        self.vertical_offset -= self.ofset_delta

    def move_roi_down(self):
        self.vertical_offset += self.ofset_delta

    def run(self):
        self.root = tk.Tk()
        self.root.protocol("WM_DELETE_WINDOW", self.callback)

        label = tk.Label(self.root, text="Sigma")
        label.pack()
        tk.Scale(from_=1.00, to=2.50, tickinterval=0.0001, resolution = 0.25, digits = 3,orient=tk.HORIZONTAL, command=self.scale_onChange).pack()

        tk.Button(text="Toggle", width=12, relief="raised", command=self.toggle).pack()

        label2 = tk.Label(self.root, text="Sub Sigma")
        label2.pack()
        tk.Scale(from_=0.20, to=1.0, tickinterval=0.0001, resolution = 0.20, digits = 3,orient=tk.HORIZONTAL, command=self.scale_onChange_b).pack()

        tk.Button(repeatdelay=250, repeatinterval=100, text="<", width=12, relief="raised", command=self.move_roi_right).pack()
        tk.Button(repeatdelay=250, repeatinterval=100, text=">", width=12, relief="raised", command=self.move_roi_left).pack()
        tk.Button(repeatdelay=250, repeatinterval=100, text="^", width=12, relief="raised", command=self.move_roi_up).pack()
        tk.Button(repeatdelay=250, repeatinterval=100, text="v", width=12, relief="raised", command=self.move_roi_down).pack()

        self.root.mainloop()

    def bring_to_front(self):
        self.root.lift()
        self.at_front = True


def kill_app(app):
    try:
        if app:
            if not app.at_front:
                app.bring_to_front()
    except Exception:
        pass

    try:
        app.callback()
        app.destroy()
    except RuntimeError:
        pass
    except Exception as e:
        print("Error while calling app.callback() or app.destroy()")
        traceback.print_exc()


def bring_to_front(app):
    try:
        if app:
            if not app.at_front:
                app.bring_to_front()
    except Exception:
        pass

def attempt_to_quit(app):
    bring_to_front(app)
    kill_app(app)