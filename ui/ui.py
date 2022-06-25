import tkinter
import data

lib = data.Library.build()

win = tkinter.Tk()
win.geometry('500x200')
dataset_idx = tkinter.IntVar(value=0)

lib.gui_chooser(win, dataset_idx)

win.mainloop()
