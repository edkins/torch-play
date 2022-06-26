from __future__ import annotations
import tkinter as tk
from tkinter import ttk

from matplotlib.style import available

from gui_helpers import Dropdown, frame, label, ButtonColumn, PrompterButton
from data import Library, Dataset
from layer import InputLayer, DenseLayer, SoftMaxLayer, available_layer_types, create_layer

class ProjectGui:
    def __init__(self, master, project: Project, library: Library, column: int, row: int):
        self.project = project
        self.library = library
        self.frame = frame(master, column=column, row=row)
        
        top_frame = frame(self.frame, column=0, row=0, columnspan=2)
        label(top_frame, text='Dataset:', column=0, row=0)
        self.dataset_dropdown = Dropdown(top_frame, column=1, row=0, selection='name', labels=library.options())
        self.dataset_dropdown.set(project.dataset.name)

        left_frame = frame(self.frame, column=0, row=1)
        label(left_frame, text='Layer:', column=0, row=0)
        self.new_layer_button = PrompterButton(left_frame, column=0, row=1, text='New', window_title='New layer', prompt=NewLayerPrompt, project_gui=self)
        self.layer_selector = ButtonColumn(left_frame, column=0, row=2, selection='index', labels=[str(layer) for layer in project.layers])
        self.layer_selector.set(project.layer_index)

    def save(self):
        self.project.dataset = self.library.get_dataset_with_name(self.dataset_dropdown.get())
        self.project.layer_index = self.layer_selector.get()

    def destroy(self):
        self.frame.destroy()

class NewLayerPrompt:
    def __init__(self, master: tk.Widget, column: int, row: int, project_gui: ProjectGui):
        self.project_gui = project_gui
        self.frame = frame(master, column=column, row=row)

        label(self.frame, "Layer type:", column=0, row=0)
        self.layer_type_dropdown = Dropdown(self.frame, column=1, row=0, selection='name', labels=available_layer_types)

        label(self.frame, "Insert after:", column=0, row=1)
        self.insert_after_dropdown = Dropdown(self.frame, column=1, row=1, selection='index', labels=[str(layer) for layer in project_gui.project.layers])

    def submit(self) -> bool:
        layer_type = self.layer_type_dropdown.get()
        insert_after = self.insert_after_dropdown.get()
        self.project_gui.project.add_layer(layer_type, insert_after)
        self.project_gui.layer_selector.set_labels([str(layer) for layer in self.project_gui.project.layers])
        self.project_gui.layer_selector.set(self.project_gui.project.layer_index)
        return True

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

    def add_layer(self, layer_type: str, insert_after: int):
        self.layers.insert(insert_after + 1, create_layer(layer_type, self.layers[insert_after].shape_out, self.layers[insert_after + 1].shape_in))
        self.layer_index = insert_after + 1