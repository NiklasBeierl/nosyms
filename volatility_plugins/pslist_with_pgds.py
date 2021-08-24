# This file is Copyright 2019 Volatility Foundation and licensed under the Volatility Software License 1.0
# which is available at https://www.volatilityfoundation.org/license/vsl-v1.0
#

from itertools import chain
from typing import Callable, Iterable, List, Any
from volatility3.framework import renderers, interfaces, contexts
from volatility3.framework.configuration import requirements
from volatility3.framework.objects import utility


class PsListWithPGDs(interfaces.plugins.PluginInterface):
    """Lists the processes present in a particular linux memory image with the the virtual addresses of their
    mm and active_mm as well as the corresponding values of the mm_struct->pgd"""

    _required_framework_version = (1, 0, 0)

    _version = (1, 0, 0)

    @classmethod
    def get_requirements(cls) -> List[interfaces.configuration.RequirementInterface]:
        return [
            requirements.TranslationLayerRequirement(
                name="primary",
                description="Memory layer for the kernel",
                architectures=["Intel32", "Intel64"],
            ),
            requirements.SymbolTableRequirement(name="vmlinux", description="Linux kernel symbols"),
            requirements.ListRequirement(
                name="pid",
                description="Filter on specific process IDs",
                element_type=int,
                optional=True,
            ),
        ]

    @classmethod
    def create_pid_filter(cls, pid_list: List[int] = None) -> Callable[[Any], bool]:
        """Constructs a filter function for process IDs.

        Args:
            pid_list: List of process IDs that are acceptable (or None if all are acceptable)

        Returns:
            Function which, when provided a process object, returns True if the process is to be filtered out of the list
        """
        pid_list = pid_list or []
        filter_list = [x for x in pid_list if x is not None]
        if filter_list:

            def filter_func(x):
                return x.pid not in filter_list

            return filter_func
        else:
            return lambda _: False

    def _generator(self):
        for task, mm, active_mm in self.list_tasks(
            self.context,
            self.config["primary"],
            self.config["vmlinux"],
            filter_func=self.create_pid_filter(self.config.get("pid", None)),
        ):
            pid = task.pid
            ppid = 0
            if task.parent:
                ppid = task.parent.pid

            pgd = mm.pgd if mm else -1
            active_pgd = active_mm.pgd if active_mm else -1

            name = utility.array_to_string(task.comm)
            yield 0, (pid, ppid, name, task.mm, pgd, task.active_mm, active_pgd, pgd == active_pgd)

    @classmethod
    def list_tasks(
        cls,
        context: interfaces.context.ContextInterface,
        layer_name: str,
        vmlinux_symbols: str,
        filter_func: Callable[[int], bool] = lambda _: False,
    ) -> Iterable[interfaces.objects.ObjectInterface]:
        """Lists all the tasks in the primary layer.

        Args:
            context: The context to retrieve required elements (layers, symbol tables) from
            layer_name: The name of the layer on which to operate
            vmlinux_symbols: The name of the table containing the kernel symbols
        Yields:
            3-tuples of task, mm, active_mm, the later two may be None.
        """
        vmlinux = contexts.Module(context, vmlinux_symbols, layer_name, 0)
        init_task = vmlinux.object_from_symbol(symbol_name="init_task")

        for task in chain([init_task], init_task.tasks):
            if not filter_func(task):
                mm = vmlinux.object("mm_struct", task.mm) if task.mm else None
                active_mm = vmlinux.object("mm_struct", task.active_mm) if task.active_mm else None
                yield task, mm, active_mm

    def run(self):
        return renderers.TreeGrid(
            [
                ("PID", int),
                ("PPID", int),
                ("COMM", str),
                ("mm", int),
                ("mm->pgd", int),
                ("active_mm", int),
                ("active_mm->pgd", int),
                ("mm->pgd == active_mm->pgd", bool),
            ],
            self._generator(),
        )
