from __future__ import annotations
import tkinter as tk
from tkinter import ttk

from gui_helpers import Dropdown, frame, label, ButtonRow
from data import Library, Dataset
from layer import InputLayer, DenseLayer, SoftMaxLayer

class ProjectGui:
    def __init__(self, master, project: Project, library: Library, column: int, row: int):
        self.project = project
        self.library = library
        self.frame = frame(master, column=column, row=row)
        
        self.frame_0 = frame(self.frame, column=0, row=0)
        self.project_heading = tk.Label(self.frame_0, text=project.name)
        self.project_heading.grid(column=0, row=0)

        self.frame_1 = frame(self.frame, column=0, row=1)
        label(self.frame_1, text='Dataset:', column=0, row=0)
        self.dataset_dropdown = Dropdown(self.frame_1, column=1, row=0, labels=library.options())
        self.dataset_dropdown.selected.set(project.dataset.name)

        self.frame_2 = frame(self.frame, column=0, row=2)
        label(self.frame_2, text='Layer:', column=0, row=0)
        self.layer_selector = ButtonRow(self.frame_2, column=1, row=0, selection='index', labels=[str(layer) for layer in project.layers])
        self.layer_selector.selected.set(project.layer_index)

    def save(self):
        self.project.dataset = self.library.get_dataset_with_name(self.dataset_dropdown.selected.get())
        self.project.layer_index = self.layer_selector.selected.get()

    def destroy(self):
        self.frame.destroy()

class Project:
    def __init__(self, name: str, dataset: Dataset):
        self.name = name
        self.dataset = dataset
        self.layers = [
            InputLayer(dataset.input_shape()),
            DenseLayer(dataset.input_shape(), dataset.output_shape()),
            SoftMaxLayer(dataset.output_shape())
        ]
        self.layer_index = 0

    def __repr__(self):
        return f'Project({self.name})'