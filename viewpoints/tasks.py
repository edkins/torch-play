import tkinter

class Task:
    def tick(self) -> bool:
        ...

class TaskManager:
    def __init__(self, win: tkinter.Tk):
        self.tasks = []
        self.stuff_to_do = False
        self.win = win
        self.stored_tick = False

    def include_task(self, task: Task) -> None:
        if task not in self.tasks:
            self.tasks.append(task)
        self.stuff_to_do = True
        if not self.stored_tick:
            self.stored_tick = True
            self.win.after(10, self.tick)

    def remove_task(self, task: Task) -> None:
        self.tasks.remove(task)

    def tick(self) -> bool:
        self.stored_tick = False
        if not self.stuff_to_do:
            return False
        stuff_to_do = False
        tasks_to_remove = []
        for task in self.tasks:
            if task.tick():
                stuff_to_do = True
            else:
                tasks_to_remove.append(task)
        self.stuff_to_do = stuff_to_do

        for task in tasks_to_remove:
            self.remove_task(task)

        if self.stuff_to_do:
            self.stored_tick = True
            self.win.after(10, self.tick)
        return stuff_to_do
