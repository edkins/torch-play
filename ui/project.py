from __future__ import annotations
import tkinter as tk
from tkinter import ttk
import torch
from typing import Callable, Optional

from gui_helpers import Dropdown, frame, label, ButtonColumn, PrompterButton, PictureColumn, ScrollableHGrid
from data import Library, Dataset
from layer import DenseLayer, SoftMaxLayer, available_layer_types, create_layer, default_layer_type
from experiment import Experiment, DummySnapshot
from tasks import TaskManager
from visualize import Visualizer, Artifact

class ProjectGui:
    def __init__(self, master, heading_master, project: Project, library: Library, column: int, row: int, heading_column: int, heading_row: int):
        self.project = project
        self.library = library
        
        heading_frame = ttk.Frame(heading_master)
        heading_frame.grid(column=heading_column, row=heading_row, columnspan=2)
        label(heading_frame, text='Dataset:', column=0, row=0)
        self.dataset_dropdown = Dropdown(heading_frame, column=1, row=0, selection='name', labels=library.options())
        self.dataset_dropdown.set(project.dataset.name)

        self.frame = ttk.Frame(master)
        self.frame.grid(column=column, row=row, sticky='ew')
        self.frame.columnconfigure(0, weight=1)

        top_frame = tk.Frame(self.frame)
        top_frame.grid(row=0, column=0, columnspan=2, sticky='ew')
        top_frame.columnconfigure(0, weight=1)
        self.hgrid = ScrollableHGrid(top_frame,
            lambda master,width,height:PictureColumn(master, constructor=Visualizer, count=self.project.picture_vcount(), width=width, height=height),
            column=0, row=0, child_width=100, child_height=100 * self.project.picture_vcount(),
            value_fetcher=project.fetch_artifacts,
            big_count=project.num_pictures())
        self.hgridscroll = ttk.Scrollbar(top_frame, orient='horizontal', command=self.hgrid.xview_scroll)
        self.hgridscroll.grid(column=0, row=1, sticky='ew')
        self.hgridvscroll = ttk.Scrollbar(top_frame, orient='vertical', command=self.hgrid.yview_scroll)
        self.hgridvscroll.grid(column=1, row=0, sticky='ns')
        self.hgrid.xscrollcommand = self.hgridscroll.set
        self.hgrid.yscrollcommand = self.hgridvscroll.set
        self.hgrid.send_xscroll_command()
        self.hgrid.send_yscroll_command()

        left_frame = frame(self.frame, column=0, row=1)
        label(left_frame, text='Layer:', column=0, row=0)
        self.new_layer_button = PrompterButton(left_frame, column=0, row=1, text='New', window_title='New layer', prompt=NewLayerPrompt, project_gui=self)
        self.layer_selector = ButtonColumn(left_frame, column=0, row=2, selection='index', labels=['input'] + [str(layer) for layer in project.layers])
        self.layer_selector.set(project.layer_index)

        self.run_button = tk.Button(left_frame, text='Run', command=self.run_experiment)
        self.run_button.grid(column=0, row=3)

        right_frame = frame(self.frame, column=1, row=1)

    def save(self):
        self.project.dataset = self.library.get_dataset_with_name(self.dataset_dropdown.get())
        self.project.layer_index = self.layer_selector.get()

    def destroy(self):
        self.frame.destroy()

    def run_experiment(self) -> None:
        self.project.run_experiment(self.update_epoch)
        self.run_button.config(text='Pause')
        self.run_button.config(command=self.pause_experiment)

    def pause_experiment(self):
        self.project.pause_experiment()
        self.run_button.config(text='Run')
        self.run_button.config(command=self.run_experiment)

    def update_epoch(self, epoch: int):
        self.project.update_epoch(epoch)
        self.hgrid.refresh()

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
    def __init__(self, name: str, dataset: Dataset, task_manager: TaskManager):
        self.name = name
        self.dataset = dataset
        self.layers = [
            DenseLayer(dataset.input_shape(), dataset.output_shape()),
            SoftMaxLayer(dataset.output_shape())
        ]
        self.layer_index = 0
        self.experiment = None
        self.task_manager = task_manager
        self.update_epoch(0)

    def __repr__(self):
        return f'Project({self.name})'

    def fetch_artifacts(self, index: int) -> list[Optional[Artifact]]:
        return self.snapshot.get_artifacts(index)

    def num_pictures(self):
        return self.dataset.train_n

    def add_layer(self, layer_type: str, insert_after: int):
        if insert_after == -1:
            self.layers.insert(0, create_layer(layer_type, self.dataset.input_shape(), self.layers[0].shape_in()))
        else:
            self.layers.insert(insert_after + 1, create_layer(layer_type, self.layers[insert_after].shape_out(), self.layers[insert_after + 1].shape_in()))
        self.layer_index = insert_after + 1

    def run_experiment(self, callback: Callable):
        if self.experiment is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            self.experiment = Experiment(self.layers, self.dataset, 10, 64, device, self.task_manager)
        self.experiment.start(callback)

    def pause_experiment(self):
        self.experiment.pause()
    
    def picture_vcount(self):
        return 1 + len(self.layers)

    def update_epoch(self, epoch: int):
        self.snapshot_epoch = epoch
        if self.experiment == None:
            self.snapshot = DummySnapshot(self.layers, self.dataset)
        else:
            self.snapshot = self.experiment.get_snapshot(epoch)
