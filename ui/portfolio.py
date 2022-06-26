import tkinter as tk
from typing import Optional
from gui_helpers import ButtonColumn, Hider, PrompterButton, tkdump
from project import Project, ProjectGui
from data import Library

class Portfolio:
    def __init__(self, library: Library):
        self.projects = []
        self.library = library

    def gui_create(self, master: tk.Widget):
        self.master = master
        self.left_frame = tk.Frame(master)
        self.left_frame.grid(column=0, row=0)
        self.main_hider = Hider(master, func=ProjectGui, column=1, row=0)
        self.new_button = PrompterButton(self.left_frame, text='New', window_title='New project', command=self.new_project, validator = self.validate_new_project_name, column=0, row=0)
        self.buttons = ButtonColumn(self.left_frame, column=0, row=1, selection='name', onchange=lambda value:self.gui_update())
        self.gui_update()
    
    def get_project_with_name(self, name: str) -> Optional[Project]:
        for project in self.projects:
            if project.name == name:
                return project
        return None

    def gui_update(self):
        self.buttons.set_labels([project.name for project in self.projects])
        proj = self.get_project_with_name(self.buttons.selected.get())
        self.main_hider.set_visibility(
            proj != None,
            on_dataset_change = self.set_dataset_name,
            on_layer_change = self.set_layer_index,
        )
        if proj != None and self.main_hider.gui != None:  # why is the main_hider.gui None sometimes?
            proj.gui_update(self.main_hider.gui, self.library)

    def set_dataset_name(self, name: str):
        self.get_project_with_name(self.buttons.selected.get()).dataset = self.library.get_dataset_with_name(name)

    def set_layer_index(self, index: int):
        self.get_project_with_name(self.buttons.selected.get()).layer_index = index

    def new_project(self, name: str):
        self.projects.append(Project(name=name, dataset=self.library.datasets[0]))
        self.buttons.selected.set(name)
        self.gui_update()

    def validate_new_project_name(self, name: str):
        return name != '' and self.get_project_with_name(name) == None