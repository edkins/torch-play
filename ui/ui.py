import tkinter
from data import Library
from portfolio import PortfolioGui

win = tkinter.Tk()
win.geometry('1000x800')

lib = Library.build()
portfolio = PortfolioGui(win, lib, column=0, row=0)
win.mainloop()
