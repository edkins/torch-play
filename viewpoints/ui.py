from curses.panel import top_panel
import tkinter as tk
from tkinter import ttk
from typing import Optional

from portfolio import create_projects, create_viewpoints
from project import Project
from tasks import TaskManager

class MainWindow:
    def __init__(self):
        self.win = tk.Tk()
        self.win.geometry('1000x800')
        self.win.title('Machine learning visualizer - Viewpoints')
        self.task_manager = TaskManager(self.win)

        self.projects = create_projects()
        self.viewpoints = create_viewpoints()

        self.project_name = tk.StringVar(self.win, self.projects[0].name)
        self.viewpoint_name = tk.StringVar(self.win, self.viewpoints[0].name)

        top_panel = ttk.Frame(self.win)
        top_panel.grid(column=0, row=0)
        ttk.Label(top_panel, text='Project:').grid(column=0, row=0)
        ttk.Combobox(top_panel, values=list(sorted(p.name for p in self.projects)), state='readonly', textvariable=self.project_name).grid(column=1, row=0)
        ttk.Label(top_panel, text='Viewpoint:').grid(column=2, row=0)
        ttk.Combobox(top_panel, values=list(sorted(v.name for v in self.viewpoints)), state='readonly', textvariable=self.viewpoint_name).grid(column=3, row=0)

        train_panel = ttk.Frame(self.win)
        train_panel.grid(column=0, row=1)
        self.train_button = ttk.Button(train_panel, text='Train', command=self.start_or_stop_training)
        self.train_button.grid(column=0, row=0)
        self.progress_bar = ttk.Progressbar(train_panel, orient='horizontal', length=200, mode='determinate')
        self.progress_bar.grid(column=1, row=0)
        self.displayed_progress = 0

        main_panel = ttk.Frame(self.win)
        main_panel.grid(column=0, row=2)

    def mainloop(self) -> None:
        self.win.mainloop()

    def project(self) -> Optional[Project]:
        for project in self.projects:
            if project.name == self.project_name.get():
                return project
        return None

    def start_or_stop_training(self):
        project = self.project()
        if project == None:
            return
        if project.running:
            project.pause()
            self.train_button.configure(text='Train')
        else:
            self.train_button.configure(text='Pause')
            self.progress_bar.configure(maximum=project.max_epochs)
            project.start_training(self.task_manager, self.progress_callback)

    def progress_callback(self) -> None:
        project = self.project()
        if project == None:
            return
        self.progress_bar.step(project.num_epochs_completed - self.displayed_progress)
        self.displayed_progress = project.num_epochs_completed

if __name__ == '__main__':
    MainWindow().mainloop()
