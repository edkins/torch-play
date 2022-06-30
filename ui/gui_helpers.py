import tkinter as tk
from tkinter import ttk
from typing import Sequence, Literal, Optional
from PIL import Image, ImageTk
import numpy as np
import math

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
                    # Stretch horizontally
                    button.grid(column=0, row=i, sticky=tk.W + tk.E)
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

def frame(master: tk.Widget, column: int, row: int, columnspan: int=1, rowspan: int=1):
    frame = tk.Frame(master)
    frame.grid(column=column, row=row, columnspan=columnspan, rowspan=rowspan)
    return frame

def label(master: tk.Widget, text: str, column: int, row: int):
    label = tk.Label(master, text=text)
    label.grid(column=column, row=row)
    return label

class Picture:
    def __init__(self, master: tk.Widget, column: Optional[int]=None, row: Optional[int]=None):
        self.label = tk.Label(master)
        if column != None:
            self.label.grid(column=column, row=row)
        self.pil_image = None
        self.tk_image = None
        self.width = 1
        self.height = 1

    def place(self, x: int, y: int, width: int, height: int):
        self.label.place(x=x, y=y, width=width, height=height)
        self.width = width
        self.height = height

    def destroy(self):
        self.label.destroy()
    
    def set(self, params: tuple[np.ndarray, Literal['L','RGB']]):
        array, mode = params
        self.pil_image = Image.fromarray(array, mode=mode)
        self.tk_image = ImageTk.PhotoImage(self.pil_image.resize((self.width, self.height), Image.NEAREST))
        self.label.config(image=self.tk_image)
        #self.label.config(text='foo')

class ScrollableHGrid:
    def __init__(self, master: tk.Widget, constructor: callable, column:int, row:int, width: int, height: int, child_width: int, child_height: int, value_fetcher: callable, big_count: int):
        self.frame = tk.Frame(master, width=width, height=height)
        self.frame.grid(column=column, row=row)
        self.count = math.ceil(width / child_width) + 1
        self.scroll = 0
        self.value_fetcher = value_fetcher
        self.constructor = constructor
        self.big_count = big_count
        self.child_width = child_width
        self.child_height = child_height
        self.children = [None] * self.count
        self.child_indices = [None] * self.count
        self.update()

    def xview_scroll(self, delta: int, unit: str):
        self.scroll += delta
        self.update()

    def update(self):
        map_min = self.scroll // self.child_width
        map_max = map_min + self.count
        map_min = max(0, map_min)
        map_max = min(self.big_count, map_max)

        child_indices = [None] * self.count
        for i in range(map_min, map_max):
            child_indices[i % self.count] = i

        for j in range(self.count):
            if self.child_indices[j] == child_indices[j]:
                # already correct, but may need moving
                self.children[j].place(x=child_indices[j] * self.child_width - self.scroll, y=0, width=self.child_width, height=self.child_height)
            elif self.child_indices[j] == None:
                self.children[j] = self.constructor(master=self.frame)
                self.children[j].place(x=child_indices[j] * self.child_width - self.scroll, y=0, width=self.child_width, height=self.child_height)
                self.children[j].set(self.value_fetcher(child_indices[j]))
            elif child_indices[j] == None:
                self.children[j].destroy()
                self.children[j] = None
            else:
                self.children[j].place(x=child_indices[j] * self.child_width - self.scroll, y=0, width=self.child_width, height=self.child_height)
                self.children[j].set(self.value_fetcher(child_indices[j]))

        self.child_indices = child_indices
