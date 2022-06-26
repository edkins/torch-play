from __future__ import annotations
import tkinter as tk
from tkinter import ttk
from gui_helpers import Dropdown, frame, label, ButtonRow
from data import Library, Dataset
from layer import InputLayer, DenseLayer, SoftMaxLayer

class ProjectGui:
    def __init__(self, master, on_dataset_change: callable, on_layer_change: callable):
        self.master = master
        
        self.frame_0 = frame(master, column=0, row=0)
        self.project_heading = tk.Label(self.frame_0, text='')
        self.project_heading.grid(column=0, row=0)

        self.frame_1 = frame(master, column=0, row=1)
        label(self.frame_1, text='Dataset:', column=0, row=0)
        self.dataset_dropdown = Dropdown(self.frame_1, column=1, row=0, onchange=on_dataset_change)

        self.frame_2 = frame(master, column=0, row=2)
        label(self.frame_2, text='Layer:', column=0, row=0)
        self.layer_selector = ButtonRow(self.frame_2, column=1, row=0, selection='index', onchange=on_layer_change)

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

    def gui_update(self, gui: ProjectGui, library: Library):
        gui.project_heading.config(text=self.name)
        gui.dataset_dropdown.selected.set(self.dataset.name)
        gui.dataset_dropdown.set_labels(library.options())
        gui.layer_selector.selected.set(self.layer_index)
        gui.layer_selector.set_labels([str(layer) for layer in self.layers])

    def __repr__(self):
        return f'Project({self.name})'