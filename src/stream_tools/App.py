import tkinter as tk
import threading
import traceback
import tkinter.ttk as ttk
import os
import pickle

class App(threading.Thread):

    def __init__(self, stream_object=None):
        threading.Thread.__init__(self)

        self.slider_maximum = stream_object.max_n_sigma
        prev_n_sigma_path = os.path.join(stream_object.prv_run_dir, "n_sigma.p")
        prev_n_sigma_exists = os.path.exists(prev_n_sigma_path)

        prev_v_offset_path = os.path.join(stream_object.prv_run_dir, "v_offset.p")
        prev_v_offset_exists = os.path.exists(prev_v_offset_path)

        prev_h_offset_path = os.path.join(stream_object.prv_run_dir, "h_offset.p")
        prev_h_offset_exists = os.path.exists(prev_h_offset_path)


        print("prev_n_sigma_exists: ", prev_n_sigma_exists)
        if prev_n_sigma_exists and stream_object is not None:
            with open(prev_n_sigma_path, 'rb') as fp:
                self.foo = pickle.load(fp)
                print("prev_v_sigma: ", self.foo)

        else:
            self.foo = 1

        print("prev_v_offset_exists: ", prev_h_offset_exists)
        if prev_v_offset_exists:
            with open(prev_v_offset_path, 'rb') as fp:
                self.vertical_offset = pickle.load(fp)
                print("prev_v_offset: ", self.vertical_offset)

        else:
            self.vertical_offset = 0

        print("prev_h_offset_exists: ", prev_h_offset_exists)
        if prev_h_offset_exists:
            with open(prev_h_offset_path, 'rb') as fp:
                self.horizontal_offset = pickle.load(fp)
                print("prev_h_offset: ", self.horizontal_offset)
        else:
            self.horizontal_offset = 0

        # NOT LOADING PREV N_SIGMA or V/H OFFSETS
        # PRINT LINE DEBUG To FIND OUT WHYYYYY?


        self.start()

        self.at_front = False
        self.is_toggle_pressed_down = False
        self.stop_streaming_override = False
        self.sub_sigma = 0.20
        self.ofset_delta = 5

        self.sigma_slider = None
        self.left_offset_button = None
        self.right_offset_button = None
        self.up_offset_button = None
        self.down_offset_button = None

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

        print("App.run() -> slider max is : ", self.slider_maximum)

        if self.slider_maximum > 1.00:
            self.sigma_slider = tk.Scale(from_=1.00, to=self.slider_maximum, tickinterval=0.0001, resolution = 0.25, digits = 3,orient=tk.HORIZONTAL, command=self.scale_onChange)


        else:
            self.sigma_slider = tk.Scale(from_=1.00, to=1.00, tickinterval=0.0001, resolution = 0.25, digits = 3,orient=tk.HORIZONTAL, command=self.scale_onChange)
            #self.sigma_slider = tk.Scale(from_=1.00, to=2.50, tickinterval=0.0001, resolution = 0.25, digits = 3,orient=tk.HORIZONTAL, command=self.scale_onChange)
            #self.sigma_slider.set(self.foo)
            #self.sigma_slider.pack()

        self.sigma_slider.set(self.foo)
        self.sigma_slider.pack()

        tk.Button(text="Toggle", width=12, relief="raised", command=self.toggle).pack()

        label2 = tk.Label(self.root, text="Sub Sigma")
        label2.pack()
        tk.Scale(from_=0.20, to=1.0, tickinterval=0.0001, resolution = 0.20, digits = 3,orient=tk.HORIZONTAL, command=self.scale_onChange_b).pack()

        self.left_offset_button = tk.Button(repeatdelay=250, repeatinterval=100, text="<", width=12, relief="raised", command=self.move_roi_right)
        self.left_offset_button.pack()

        self.right_offset_button = tk.Button(repeatdelay=250, repeatinterval=100, text=">", width=12, relief="raised", command=self.move_roi_left)
        self.right_offset_button.pack()

        self.up_offset_button = tk.Button(repeatdelay=250, repeatinterval=100, text="^", width=12, relief="raised", command=self.move_roi_up)
        self.up_offset_button.pack()

        self.down_offset_button = tk.Button(repeatdelay=250, repeatinterval=100, text="v", width=12, relief="raised", command=self.move_roi_down)
        self.down_offset_button.pack()


        #tk.Button(repeatdelay=250, repeatinterval=100, text="<", width=12, relief="raised", command=self.move_roi_right).pack()
        #tk.Button(repeatdelay=250, repeatinterval=100, text=">", width=12, relief="raised", command=self.move_roi_left).pack()
        #tk.Button(repeatdelay=250, repeatinterval=100, text="^", width=12, relief="raised", command=self.move_roi_up).pack()
        #tk.Button(repeatdelay=250, repeatinterval=100, text="v", width=12, relief="raised", command=self.move_roi_down).pack()

        self.root.mainloop()

    def bring_to_front(self):
        self.root.lift()
        self.at_front = True

    def disable_sigma_slider(self):
        self.sigma_slider.configure(state='disabled')

    def disable_offsets(self):
        self.sigma_slider.configure(state='disabled')

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