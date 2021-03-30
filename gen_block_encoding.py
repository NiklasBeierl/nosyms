import json
import enum
import pandas as pd
from collections import defaultdict
from typing import List, Dict, Tuple
from volatility.framework import contexts
from volatility.framework.symbols import linux

sym_path = "//home/niklas/code/nosyms/symbols/all_syms/2.6.15-1.2054_FC5.x86_64.json"
ctx = contexts.Context()
table_name = ctx.symbol_space.free_table_name("BlockTypeEncoder")
table = linux.LinuxKernelIntermedSymbols(ctx,
                                         table_name,
                                         name=table_name,
                                         isf_url="file://" + sym_path)
# ctx.symbol_space.append(table) # TODO: Necessary?

task_struct_sym = table.get_type("task_struct")

with open(sym_path) as f:
    syms_json = json.load(f)

task_struct_json = syms_json["user_types"]["task_struct"]
task_struct_fields = pd.DataFrame(task_struct_json["fields"]).T.sort_values("offset")


class BlockType(enum.Enum):
    Zero = 0
    Pointer = 1
    String = 2
    Data = 3


class AtomicType:
    Padding = 0
    Pointer = 1
    String = 2
    Data = 3


# Every base type has to have one of these "kinds".
# Reference: volatility/schemas/schema-6.2.0.json:definitions.element_base_type
# "void" is not covered here because it can only be pointed to, but can not be a field in a struct. sizeof(void) == 0
BASE_KIND_TO_ATOMIC = {
    "char": AtomicType.String,
    "int": AtomicType.Data,
    "float": AtomicType.Data,
    "bool": AtomicType.Data
}


def _prevent_recursion_loop(recur_arg, func):
    def _func(*args, **kwargs):
        bread_crumbs = kwargs.get("bread_crumbs", [])
        arg = args[recur_arg] if type(recur_arg) is int else kwargs.get(recur_arg)
        call_id = func.__name__ + arg
        if call_id in bread_crumbs:
            raise Exception("Infinite recursion detected.", bread_crumbs)
        bread_crumbs.append(call_id)
        return func(*args, **kwargs)

    return _func


def prevent_recursion_loop(func_or_arg_name):
    if callable(func_or_arg_name):
        return _prevent_recursion_loop(1, func_or_arg_name)
    else:
        return lambda func: _prevent_recursion_loop(func_or_arg_name, func)


class AtomicEncoder:
    def __init__(self, syms):
        self.syms = syms

    def base_type_to_atomic(self, base_type: str) -> List[AtomicType]:
        """
        Reference: volatility/schemas/schema-6.2.0.json:definitions.element_base_type
        """
        base_type = self.syms["base_types"][base_type]
        if base_type["kind"] == "void":
            raise Exception("Attempted to interpret void as a struct member, how did you get here?")
        return [BASE_KIND_TO_ATOMIC[base_type["kind"]]] * base_type["size"]

    def enum_to_atomic(self, enum: str) -> List[AtomicType]:
        # Enums can hardly be discovered in the raw data, they are treated as data.
        return self.syms["enums"][enum]["size"] * [AtomicType.Data]

    @staticmethod
    def _assign_final_type(atoms: List[Tuple[AtomicType, str]]) -> AtomicType:
        """
        Chooses the final atomic type from a list of possible ones for a given offset within a type.
        """
        if not atoms:  # No field covered this offset, has to be padding.
            return (AtomicType.Padding, None)
        elif len(set(atoms)) == 1:  # Clear case.
            return atoms[0]
        else:  # Conflict, default to Data.
            return (AtomicType.Data, None)

    def user_type_to_atomic(self, user_type: "element_user_type", bread_crumbs=[]):
        """
        Reference: volatility/schemas/schema-6.2.0.json:definitions.element_user_type
        """
        user_type = self.syms["user_types"][user_type]
        # TODO: user_type has a "kind" as well, do I care?
        atomic_fields = [
            (name, description["offset"]) + self.type_descriptor_to_atomic(description["type"],
                                                                           bread_crumbs=bread_crumbs)
            for name, description in user_type["fields"].items()]
        offset_map = defaultdict(lambda: list())
        for name, offset, atoms, target in atomic_fields:
            for i, atom in enumerate(atoms):
                offset_map[offset + i].append((atom, target))

        result = [self._assign_final_type(offset_map[i]) for i in range(user_type["size"])]
        return result

    @staticmethod
    def _get_pointer_name(subtype: dict) -> str:
        # So in theory "subtype" is another type descriptor. Which means this can recurse.
        # But the pointers are only useful when they point to something we can recognize.
        if subtype["kind"] in ["base", "struct", "union", "class", "enum"]:
            return subtype["name"]
        else:  # ["array", "bitfield", "pointer", "function"]
            return subtype["kind"]

    def type_descriptor_to_atomic(self, type_descriptor: "type_descriptor", bread_crumbs=[]) -> List[Tuple]:
        """
        Reference: volatility/schemas/schema-6.2.0.json:definitions.type_descriptor
        :param type_descriptor:
        :param bread_crumbs:
        :return:
        """
        kind = type_descriptor["kind"]
        if kind in ["function", "pointer"]:
            return (self.base_type_to_atomic("pointer"), self._get_pointer_name(type_descriptor["subtype"]))
        elif kind == "enum":
            return (self.enum_to_atomic(type_descriptor["name"]), None)
        elif kind == "base":
            return (self.base_type_to_atomic(type_descriptor["name"]), None)
        elif kind == "bitfield":  # Bitfield is just a wrapper around an enum or base_type.
            return self.type_descriptor_to_atomic(type_descriptor["type"], bread_crumbs=bread_crumbs)
        elif kind == "array":
            return (type_descriptor["count"] * self.type_descriptor_to_atomic(type_descriptor["subtype"]), None)
        elif kind in["struct", "class", "union"]:  # Reference to a user type
            return (self.user_type_to_atomic(type_descriptor["name"], bread_crumbs=bread_crumbs), None)


encoder = AtomicEncoder(syms_json)
atomic_task_struct = encoder.user_type_to_atomic("task_struct")

print("Done")
