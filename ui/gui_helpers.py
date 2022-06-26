import tkinter as tk
from tkinter import ttk
from typing import Sequence, Literal

class ButtonSet:
    def __init__(self,
            master: tk.Widget,
            column: int, row: int,
            vertical: bool,
            selection: Literal['name','index'],
            labels: Sequence[str],
            onchange: callable=lambda value:None):
        self.onchange = onchange
        self.frame = tk.Frame(master)
        self.frame.grid(column=column, row=row)
        self.buttons = []
        self.labels = ()
        self.selection = selection
        if self.selection == 'name':
            self.selected = tk.StringVar(master)
            self.selected.trace_add('write', lambda a,b,c:self.onchange(self.selected.get()))
        elif self.selection == 'index':
            self.selected = tk.IntVar(master)
            self.selected.trace_add('write', lambda a,b,c:self.onchange(self.selected.get()))
        else:
            raise ValueError(f'Invalid selection type: {self.selection}')

        self.vertical = vertical
        self.set_labels(labels)

    def set_labels(self, labels: Sequence[str]):
        labels = tuple(labels)
        if self.labels != labels:
            for button in self.buttons:
                button.destroy()

            if self.selection == 'name':
                self.buttons = [tk.Radiobutton(self.frame, text=label, indicatoron=0, value=label, variable=self.selected) for label in labels]
            elif self.selection == 'index':
                self.buttons = [tk.Radiobutton(self.frame, text=label, indicatoron=0, value=i, variable=self.selected) for i, label in enumerate(labels)]

            self.labels = labels
            for i, button in enumerate(self.buttons):
                if self.vertical:
                    button.grid(column=0, row=i)
                else:
                    button.grid(column=i, row=0)

class ButtonColumn(ButtonSet):
    def __init__(self,
            master: tk.Widget,
            column: int, row: int,
            selection: Literal['name','index'],
            labels: Sequence[str],
            onchange: callable=lambda value:None):
        super().__init__(master, column=column, row=row, vertical=True, selection=selection, labels=labels, onchange=onchange)

class ButtonRow(ButtonSet):
    def __init__(self,
            master: tk.Widget,
            column: int, row: int,
            selection: Literal['name','index'],
            labels: Sequence[str],
            onchange: callable=lambda value:None):
        super().__init__(master, column=column, row=row, vertical=False, selection=selection, labels=labels, onchange=onchange)

class Dropdown:
    def __init__(self, master: tk.Widget, column: int, row: int, labels:Sequence[str]=(), onchange: callable=lambda value:None):
        self.onchange = onchange
        self.master = master
        self.labels = ()
        self.selected = tk.StringVar(master)
        self.combo = ttk.Combobox(master, textvariable=self.selected, state='readonly', values=self.labels)
        self.combo.grid(column=column, row=row)
        self.selected.trace_add('write', lambda a,b,c:self.onchange(self.selected.get()))
        self.set_labels(labels)
    
    def set_labels(self, labels: Sequence[str]):
        labels = tuple(labels)
        if self.labels != labels:
            self.combo.config(values=labels)
            self.labels = labels

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

def frame(master: tk.Widget, column: int, row: int):
    frame = tk.Frame(master)
    frame.grid(column=column, row=row)
    return frame

def label(master: tk.Widget, text: str, column: int, row: int):
    label = tk.Label(master, text=text)
    label.grid(column=column, row=row)
    return label