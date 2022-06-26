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
        self.project_heading = tk.Label(self.frame_0, text='')
        self.project_heading.grid(column=0, row=0)

        self.frame_1 = tk.Frame(master)
        self.frame_1.grid(column=0, row=1)
        text = tk.Label(self.frame_1, text='Dataset:')
        text.grid(column=0, row=0)
        self.dataset = tk.StringVar(master)
        #self.dataset_idx.trace('w', self.dataset_idx_changed)
        self.dataset_dropdown = Dropdown(self.frame_1, column=1, row=0)

class Project:
    def __init__(self, name: str, dataset_name: str):
        self.name = name
        self.dataset_name = dataset_name

    def gui_update(self, gui: ProjectGui, library: Library):
        gui.project_heading.config(text=self.name)
        gui.dataset_dropdown.set_labels(library.options())
        gui.dataset_dropdown.selected.set(self.dataset_name)

    # Note that this does not update the gui.
    def set_dataset_name(self, name: str):
        self.dataset_name = name

    def __repr__(self):
        return f'Project({self.name}, {self.dataset_name})'