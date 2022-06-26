import tkinter as tk
from tkinter import ttk
from typing import Sequence

class ButtonColumn:
    def __init__(self, master: tk.Widget, column: int, row: int, onchange: callable=lambda value:None):
        self.onchange = onchange
        self.frame = tk.Frame(master)
        self.frame.grid(column=column, row=row)
        self.buttons = []
        self.labels = ()
        self.selected = tk.StringVar(master)
        self.selected.trace_add('write', lambda a,b,c:self.onchange(self.selected.get()))

    def set_labels(self, labels: Sequence[str]):
        labels = tuple(labels)
        if self.labels != labels:
            for button in self.buttons:
                button.destroy()
            self.buttons = [tk.Radiobutton(self.frame, text=label, indicatoron=0, value=label, variable=self.selected) for label in labels]
            self.labels = labels
            for i, button in enumerate(self.buttons):
                button.grid(column=0, row=i)

class Dropdown:
    def __init__(self, master: tk.Widget, column: int, row: int, onchange: callable=lambda value:None):
        self.onchange = onchange
        self.master = master
        self.labels = ()
        self.selected = tk.StringVar(master)
        self.combo = ttk.Combobox(master, textvariable=self.selected, state='readonly', values=self.labels)
        self.combo.grid(column=column, row=row)
        self.combo.bind('<<ComboboxSelected>>', lambda event: self.onchange(self.selected.get()))
    
    def set_labels(self, labels: Sequence[str]):
        labels = tuple(labels)
        if self.labels != labels:
            self.combo.config(values=labels)
            self.labels = labels

class Hider:
    def __init__(self, master: tk.Widget, func: callable, column: int, row: int):
        self.master = master
        self.func = func
        self.frame = None
        self.column = column
        self.row = row
        self.gui = None

    def set_visibility(self, visible: bool):
        was_visible = self.frame != None
        if visible and not was_visible:
            self.frame = tk.Frame(self.master)
            self.frame.grid(column=self.column, row=self.row)
            self.gui = self.func(self.frame)
        elif not visible and was_visible:
            self.frame.destroy()
            self.frame = None
            self.gui = None

class PrompterButton:
    def __init__(self, master: tk.Widget, text: str, window_title: str, column: int, row: int, command: callable, validator: callable):
        self.master = master
        self.text = text
        self.window_title = window_title
        self.command = command
        self.validator = validator
        self.button = tk.Button(master, text=text, command=self.click)
        self.button.grid(column=column, row=row)
        self.win = None
        self.entry = None

    def click(self):
        if self.win != None:
            self.win.destroy()
        self.win = tk.Toplevel(self.master)
        self.win.title(self.window_title)
        self.entry = tk.Entry(self.win)
        self.entry.grid(column=0, row=0)
        button = tk.Button(self.win, text='OK', command=self.click_ok)
        button.grid(column=1, row=0)
        cancel = tk.Button(self.win, text='Cancel', command=self.click_cancel)
        cancel.grid(column=2, row=0)
        self.message = tk.Label(self.win, text='')
        self.message.grid(column=3, row=0)
        self.entry.focus_set()
        # Press OK button when enter is pressed
        self.entry.bind('<Return>', lambda event: self.click_ok())
        # Press Cancel button when escape is pressed
        self.entry.bind('<Escape>', lambda event: self.click_cancel())

    def click_ok(self):
        value = self.entry.get()
        if self.validator(value):
            self.win.destroy()
            self.win = None
            self.entry = None
            self.command(value)
        else:
            self.message.config(text='Invalid value')

    def click_cancel(self):
        self.win.destroy()
        self.win = None
        self.entry = None

def tkgridinfo(widget):
    try:
        info = widget.grid_info()
        return f'{info["column"]},{info["row"]}'
    except AttributeError:
        return 'x'

def tkdump(widget, depth=0):
    print(' ' * depth + str(widget) + ' -- ' + tkgridinfo(widget))
    for child in widget.grid_slaves():
        tkdump(child, depth+1)
