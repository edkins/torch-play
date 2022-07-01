import tkinter as tk
from tkinter import ttk
from typing import Optional
from gui_helpers import Dropdown, PrompterButton, TextPrompt, frame
from project import Project, ProjectGui
from data import Library
from tasks import TaskManager

class Placeholder:
    def __init__(self, master: tk.Widget, column: int, row: int):
        self.frame = frame(master, column=column, row=row)
        self.label = tk.Label(self.frame, text='Click "New" to start a project')
        self.label.grid(column=0, row=0)

    def save(self):
        pass

    def destroy(self):
        self.frame.destroy()

class PortfolioGui:
    def __init__(self, master: tk.Widget, library: Library, task_manager: TaskManager, column: int, row: int, projects:list[Project] = [], selected_project_name:str = ''):
        self.projects = projects
        self.selected_project_name = selected_project_name
        self.library = library
        self.task_manager = task_manager
        self.frame = ttk.Frame(master)
        self.frame.grid(column=column, row=row, sticky='ew')
        self.frame.columnconfigure(0, weight=1)

        self.top_frame = frame(self.frame, column=0, row=0)
        self.new_button = PrompterButton(self.top_frame, column=0, row=0, text='New project', window_title='New project', prompt=TextPrompt, command=self.new_project, validator = self.validate_new_project_name)
        self.buttons = Dropdown(
            self.top_frame,
            column=1, row=0,
            selection='name',
            labels=[project.name for project in self.projects],
            onchange=self.select_project)

        self.main_frame = Placeholder(self.frame, column=0, row=1)
        self.select_project(selected_project_name)
    
    def get_project_with_name(self, name: str) -> Optional[Project]:
        for project in self.projects:
            if project.name == name:
                return project
        return None

    def select_project(self, project_name: str):
        self.selected_project_name = project_name
        self.buttons.set(project_name)
        project = self.get_project_with_name(project_name)

        self.main_frame.save()
        self.main_frame.destroy()
        if project == None:
            self.main_frame = Placeholder(self.frame, column=0, row=1)
        else:
            self.main_frame = ProjectGui(self.frame, self.top_frame, project, self.library, column=0, row=1, heading_column=2, heading_row=0)

    def new_project(self, name: str):
        self.projects.append(Project(name=name, dataset=self.library.datasets[0], task_manager=self.task_manager))
        self.buttons.set_labels([project.name for project in self.projects])
        self.select_project(name)

    def validate_new_project_name(self, name: str):
        return name != '' and self.get_project_with_name(name) == None