from curses.panel import top_panel
import tkinter as tk
from tkinter import ttk
from typing import Optional
from PIL import ImageTk, Image

from portfolio import create_projects, CLASS_LABELS
from project import Project
from tasks import TaskManager
from image_dropdown import ImageDropdown
from to_canvas import onto_canvas

class MainWindow:
    def __init__(self):
        self.win = tk.Tk()
        self.win.geometry('2000x1600')
        self.win.title('Machine learning visualizer - Viewpoints')
        self.task_manager = TaskManager(self.win)

        self.projects = create_projects()

        self.project_name = tk.StringVar(self.win, self.projects[0].name)
        self.viewpoint_name = tk.StringVar(self.win, '')

        top_panel = ttk.Frame(self.win)
        top_panel.grid(column=0, row=0)
        ttk.Label(top_panel, text='Project:').grid(column=0, row=0)
        self.project_combo = ttk.Combobox(top_panel, values=list(sorted(p.name for p in self.projects)), state='readonly', textvariable=self.project_name)
        self.project_combo.grid(column=1, row=0)
        self.project_combo.bind('<<ComboboxSelected>>', lambda e: self.change_project())
        ttk.Label(top_panel, text='Viewpoint:').grid(column=2, row=0)
        self.viewpoint_combo = ttk.Combobox(top_panel, values=[], state='readonly', textvariable=self.viewpoint_name)
        self.viewpoint_combo.grid(column=3, row=0)
        self.viewpoint_combo.bind('<<ComboboxSelected>>', lambda e: self.populate_canvas())
        self.populate_viewpoints()

        train_panel = ttk.Frame(self.win)
        train_panel.grid(column=0, row=1)
        self.train_button = ttk.Button(train_panel, text='Train', command=self.start_or_stop_training)
        self.train_button.grid(column=0, row=0)
        self.progress_bar = ttk.Progressbar(train_panel, orient='horizontal', length=200, mode='determinate')
        self.progress_bar.grid(column=1, row=0)
        self.displayed_progress = 0

        self.inp_panel = None
        self.inp_preview = None
        self.inp_preview_img = None
        self.populate_inp_panel()

        main_panel = ttk.Frame(self.win)
        main_panel.grid(column=0, row=3)
        self.main_canvas = tk.Canvas(main_panel, width=2000, height=1000)
        self.main_canvas.grid(column=0, row=0)
        
        self.populate_progress_bar()

    def change_project(self) -> None:
        self.populate_viewpoints()
        self.populate_inp_panel()
        self.populate_progress_bar()

    def populate_inp_panel(self) -> None:
        if self.inp_panel != None:
            self.inp_panel.destroy()

        project = self.project()
        self.inp_panel = ttk.Frame(self.win)
        self.inp_panel.grid(column=0, row=2)
        self.inp_preview = ttk.Label(self.inp_panel)
        self.inp_preview.grid(column=0, row=0, rowspan=2)
        self.inp_clear = ttk.Button(self.inp_panel, text='X', width=1, command=lambda:self.select_test_image(None))
        self.inp_clear.grid(column=1, row=0, rowspan=2)
        self.inp_tensor = None
        self.inp_y = None
        ttk.Label(self.inp_panel, text='Training:').grid(column=2, row=0)
        ttk.Label(self.inp_panel, text='Test:').grid(column=2, row=1)
        dimension, k = project.out_size[0]

        train_indices = [[] for _ in range(k)]
        test_indices = [[] for _ in range(k)]
        total = 0
        for i, c in enumerate(project.get_all_training_y()):
            if len(train_indices[c]) < 100:
                train_indices[c].append(i)
                total += 1
                if total >= 100 * k:
                    break
        total = 0
        for i, c in enumerate(project.get_all_test_y()):
            if len(test_indices[c]) < 100:
                test_indices[c].append(i)
                total += 1
                if total >= 100 * k:
                    break

        for category in range(k):
            ImageDropdown(
                self.inp_panel,
                text=CLASS_LABELS[dimension][category],
                values = train_indices[category],
                renderer = project.get_training_image,
                onchange = self.select_training_image,
            ).grid(column=category+3, row=0)
            ImageDropdown(
                self.inp_panel,
                text=CLASS_LABELS[dimension][category],
                values = test_indices[category],
                renderer = project.get_test_image,
                onchange = self.select_test_image,
            ).grid(column=category+3, row=1)

    def populate_canvas(self) -> None:
        self.main_canvas.delete('all')
        project = self.project()
        if project == None:
            return
        viewpoint = project.get_viewpoint(self.viewpoint_name.get())
        if viewpoint == None:
            return
        if self.inp_tensor == None:
            return
        onto_canvas(project, viewpoint, self.main_canvas, self.inp_tensor, project.out_size[0][0], self.inp_y)

    def select_training_image(self, index: Optional[int]) -> None:
        project = self.project()
        if project == None or index == None:
            self.inp_preview_img = None
            self.inp_preview.configure(image=None)
            self.inp_tensor = None
            return
        self.inp_preview_img = ImageTk.PhotoImage(project.get_training_image(index).resize((150, 150), Image.NEAREST))
        self.inp_preview.configure(image=self.inp_preview_img)
        self.inp_tensor = project.get_training_x(index)
        self.inp_y = project.get_training_y(index)
        self.populate_canvas()

    def select_test_image(self, index: Optional[int]) -> None:
        project = self.project()
        if project == None or index == None:
            self.inp_preview_img = None
            self.inp_preview.configure(image=None)
            self.inp_tensor = None
            return
        self.inp_preview_img = ImageTk.PhotoImage(project.get_test_image(index).resize((150, 150), Image.NEAREST))
        self.inp_preview.configure(image=self.inp_preview_img)
        self.inp_tensor = project.get_test_x(index)
        self.inp_y = project.get_test_y(index)
        self.populate_canvas()
        

    def populate_viewpoints(self) -> None:
        project = self.project()
        if project == None:
            self.viewpoint_combo.configure(values=[])
            return
        self.viewpoint_combo.configure(values=list(sorted(v.name for v in project.viewpoints)))
        self.viewpoint_name.set(project.viewpoints[0].name)

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
            self.populate_canvas()
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

    def populate_progress_bar(self) -> None:
        project = self.project()
        if project == None:
            self.progress_bar.configure(maximum=0)
            return
        self.progress_bar.configure(maximum=project.max_epochs)
        self.progress_bar.step(project.num_epochs_completed - self.displayed_progress)
        self.displayed_progress = project.num_epochs_completed

if __name__ == '__main__':
    MainWindow().mainloop()
