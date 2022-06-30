import tkinter
from data import Library
from portfolio import PortfolioGui
from tasks import TaskManager

win = tkinter.Tk()
win.geometry('1000x800')
win.title('Machine learning visualizer')

lib = Library.build()
task_manager = TaskManager(win)
portfolio = PortfolioGui(win, lib, task_manager, column=0, row=0)
win.mainloop()
