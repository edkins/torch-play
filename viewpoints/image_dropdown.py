import tkinter as tk
from tkinter import ttk
from typing import Callable
from PIL import ImageTk

class ImageDropdown(ttk.Button):
    def __init__(self, master: tk.Misc, text: str, values:list, renderer:Callable, onchange:Callable, **kwargs):
        ttk.Button.__init__(self, master, text=text, command=self.click_button, **kwargs)
        self.text = text
        self.values = values
        self.renderer = renderer
        self.onchange = onchange
        self.popup = None
        self.images = []

    def click_button(self):
        if self.popup is None:
            self.popup = tk.Toplevel(self.master)
            self.popup.wm_title(self.text)
            self.popup.geometry('600x300')
            self.popup.transient(self.master)
            self.popup.grab_set()
            self.popup.focus_set()
            self.popup.bind('<Escape>', lambda e: self.close())

            self.images = []

            for i, value in enumerate(self.values):
                img = ImageTk.PhotoImage(self.renderer(value))
                self.images.append(img)
                ttk.Button(
                    self.popup,
                    image=img,
                    command=self.onchange_with(value)
                ).grid(column=i % 20, row=i // 20)
        else:
            self.close()

    # Add an extra function here so that value gets captured correctly
    def onchange_with(self, value):
        def lamb():
            self.close()
            self.onchange(value)
        return lamb

    def close(self):
        if self.popup != None:
            self.popup.destroy()
            self.popup = None
            self.images = []
