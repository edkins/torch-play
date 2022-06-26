from __future__ import annotations
import tkinter as tk
from tkinter import ttk
from data import Library
from gui_helpers import Dropdown

class ProjectGui:
    def __init__(self, master):
        self.master = master
        self.frame_0 = tk.Frame(master)
        self.frame_0.grid(column=0, row=0)
        text = tk.Label(self.frame_0, text='Dataset:')
        text.grid(column=0, row=0)
        self.dataset = tk.StringVar(master)
        #self.dataset_idx.trace('w', self.dataset_idx_changed)
        self.dataset_dropdown = Dropdown(self.frame_0, column=1, row=0)

class Project:
    def __init__(self, name: str, dataset_name: str):
        self.name = name
        self.dataset_name = dataset_name

    def gui_update(self, gui: ProjectGui, library: Library):
        gui.dataset_dropdown.set_labels(library.options())
        gui.dataset_dropdown.selected.set(self.dataset_name)
