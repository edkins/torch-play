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
            self._selected = tk.StringVar(master)
            self._selected.trace_add('write', lambda a,b,c:self.onchange(self._selected.get()))
        elif self.selection == 'index':
            self._selected = tk.IntVar(master)
            self._selected.trace_add('write', lambda a,b,c:self.onchange(self._selected.get()))
        else:
            raise ValueError(f'Invalid selection type: {self.selection}')

        self.vertical = vertical
        self.set_labels(labels)

    def get(self) -> str | int:
        return self._selected.get()

    def set(self, value: str | int):
        self._selected.set(value)

    def set_labels(self, labels: Sequence[str]):
        labels = tuple(labels)
        if self.labels != labels:
            for button in self.buttons:
                button.destroy()

            if self.selection == 'name':
                self.buttons = [tk.Radiobutton(self.frame, text=label, indicatoron=0, value=label, variable=self._selected) for label in labels]
            elif self.selection == 'index':
                self.buttons = [tk.Radiobutton(self.frame, text=label, indicatoron=0, value=i, variable=self._selected) for i, label in enumerate(labels)]

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
    def __init__(self, master: tk.Widget, column: int, row: int, selection:Literal['name','index'], labels:Sequence[str]=(), onchange: callable=lambda value:None):
        self.onchange = onchange
        self.master = master
        self.labels = ()
        self.selection = selection
        self._selected = tk.StringVar(master)
        self.combo = ttk.Combobox(master, textvariable=self._selected, state='readonly', values=self.labels)
        self.combo.grid(column=column, row=row)
        self._selected.trace_add('write', lambda a,b,c:self.onchange(self._selected.get()))
        self.set_labels(labels)
    
    def set_labels(self, labels: Sequence[str]):
        labels = tuple(labels)
        if self.labels != labels:
            self.combo.config(values=labels)
            self.labels = labels

    def set(self, value: str | int):
        if self.selection == 'name':
            self._selected.set(value)
        elif self.selection == 'index':
            self.combo.current(value)
        else:
            raise ValueError(f'Invalid selection type: {self.selection}')

    def get(self) -> str | int:
        if self.selection == 'name':
            return self._selected.get()
        elif self.selection == 'index':
            return self.combo.current()
        else:
            raise ValueError(f'Invalid selection type: {self.selection}')

class TextPrompt:
    def __init__(self, master: tk.Widget, column: int, row: int, validator: callable, command: callable):
        self.validator = validator
        self.command = command
        self.frame = frame(master, column=column, row=row)
        self.entry = tk.Entry(self.frame)
        self.entry.grid(column=0, row=0)
        self.message = label(self.frame, text='', column=0, row=1)
        self.entry.focus_set()

    def submit(self) -> bool:
        value = self.entry.get()
        if self.validator(value):
            self.command(value)
            return True
        else:
            self.message.config(text='Invalid value')
            return False

class PrompterButton:
    def __init__(self, master: tk.Widget, text: str, window_title: str, column: int, row: int, prompt: callable, **kwargs):
        self.master = master
        self.text = text
        self.window_title = window_title
        self.prompt = prompt
        self.button = tk.Button(master, text=text, command=self.click)
        self.button.grid(column=column, row=row)
        self.win = None
        self.prompt_frame = None
        self.kwargs = kwargs

    def click(self):
        if self.win != None:
            self.win.destroy()
        self.win = tk.Toplevel(self.master)
        self.win.title(self.window_title)
        self.prompt_frame = self.prompt(master=self.win, column=0, row=0, **self.kwargs)
        button = tk.Button(self.win, text='OK', command=self.click_ok)
        button.grid(column=1, row=0)
        cancel = tk.Button(self.win, text='Cancel', command=self.click_cancel)
        cancel.grid(column=2, row=0)
        self.message = tk.Label(self.win, text='')
        self.message.grid(column=3, row=0)
        # Press OK button when enter is pressed
        self.win.bind('<Return>', lambda event: self.click_ok())
        # Press Cancel button when escape is pressed
        self.win.bind('<Escape>', lambda event: self.click_cancel())

    def click_ok(self):
        if self.prompt_frame.submit():
            self.win.destroy()
            self.win = None
            self.prompt_frame = None

    def click_cancel(self):
        self.win.destroy()
        self.win = None
        self.prompt_frame = None

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