from __future__ import annotations
import tkinter as tk
import torch

from gui_helpers import Dropdown, frame, label, ButtonColumn, PrompterButton, Picture
from data import Library, Dataset
from layer import DenseLayer, SoftMaxLayer, available_layer_types, create_layer, default_layer_type
from experiment import Experiment

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
        self.layer_selector = ButtonColumn(left_frame, column=0, row=2, selection='index', labels=['input'] + [str(layer) for layer in project.layers])
        self.layer_selector.set(project.layer_index)

        self.run_button = tk.Button(left_frame, text='Run', command=self.run_experiment)
        self.run_button.grid(column=0, row=3)

        right_frame = frame(self.frame, column=1, row=1)
        self.picture = Picture(right_frame, column=0, row=0)
        self.picture.set(*project.dataset.get_train_image(0))

    def save(self):
        self.project.dataset = self.library.get_dataset_with_name(self.dataset_dropdown.get())
        self.project.layer_index = self.layer_selector.get()

    def destroy(self):
        self.frame.destroy()

    def run_experiment(self):
        self.project.run_experiment()
        self.run_button.config(text='Pause')
        self.run_button.config(command=self.pause_experiment)

    def pause_experiment(self):
        self.project.pause_experiment()
        self.run_button.config(text='Run')
        self.run_button.config(command=self.run_experiment)

class NewLayerPrompt:
    def __init__(self, master: tk.Widget, column: int, row: int, project_gui: ProjectGui):
        self.project_gui = project_gui
        self.frame = frame(master, column=column, row=row)

        label(self.frame, "Layer type:", column=0, row=0)
        self.layer_type_dropdown = Dropdown(self.frame, column=1, row=0, selection='name', labels=available_layer_types)
        self.layer_type_dropdown.set(default_layer_type)

        label(self.frame, "Insert after:", column=0, row=1)
        self.insert_after_dropdown = Dropdown(self.frame, column=1, row=1, selection='index', labels=['input'] + [str(layer) for layer in project_gui.project.layers])
        self.insert_after_dropdown.set(project_gui.layer_selector.get())

    def submit(self) -> bool:
        layer_type = self.layer_type_dropdown.get()
        insert_after = self.insert_after_dropdown.get() - 1
        self.project_gui.project.add_layer(layer_type, insert_after)
        self.project_gui.layer_selector.set_labels([str(layer) for layer in self.project_gui.project.layers])
        self.project_gui.layer_selector.set(self.project_gui.project.layer_index)
        return True

class Project:
    def __init__(self, name: str, dataset: Dataset):
        self.name = name
        self.dataset = dataset
        self.layers = [
            DenseLayer(dataset.input_shape(), dataset.output_shape()),
            SoftMaxLayer(dataset.output_shape())
        ]
        self.layer_index = 0
        self.experiment = None

    def __repr__(self):
        return f'Project({self.name})'

    def add_layer(self, layer_type: str, insert_after: int):
        if insert_after == -1:
            self.layers.insert(0, create_layer(layer_type, self.dataset.input_shape(), self.layers[0].shape_in()))
        else:
            self.layers.insert(insert_after + 1, create_layer(layer_type, self.layers[insert_after].shape_out(), self.layers[insert_after + 1].shape_in()))
        self.layer_index = insert_after + 1

    def run_experiment(self):
        if self.experiment is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            self.experiment = Experiment(self.layers, self.dataset, 10, 64, device)
        self.experiment.start()

    def pause_experiment(self):
        self.experiment.pause()
    