import tkinter
from data import Library
from portfolio import Portfolio

win = tkinter.Tk()
win.geometry('1000x800')

lib = Library.build()
portfolio = Portfolio(lib)
portfolio.gui_create(win)
win.mainloop()
