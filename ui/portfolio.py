import tkinter as tk
from typing import Optional
from gui_helpers import Dropdown, PrompterButton, TextPrompt, frame
from project import Project, ProjectGui
from data import Library

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
    def __init__(self, master: tk.Widget, library: Library, column: int, row: int, projects:list[Project] = [], selected_project_name:str = ''):
        self.projects = projects
        self.selected_project_name = selected_project_name
        self.library = library
        self.frame = frame(master, column=column, row=row)

        top_frame = frame(self.frame, column=0, row=0)
        self.new_button = PrompterButton(top_frame, column=0, row=0, text='New project', window_title='New project', prompt=TextPrompt, command=self.new_project, validator = self.validate_new_project_name)
        self.buttons = Dropdown(
            top_frame,
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
            self.main_frame = ProjectGui(self.frame, project, self.library, column=0, row=1)

    def new_project(self, name: str):
        self.projects.append(Project(name=name, dataset=self.library.datasets[0]))
        self.buttons.set_labels([project.name for project in self.projects])
        self.select_project(name)

    def validate_new_project_name(self, name: str):
        return name != '' and self.get_project_with_name(name) == None