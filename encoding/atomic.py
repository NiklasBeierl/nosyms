from encoding.types import AtomicType
from collections import defaultdict
from typing import List, Union, Tuple

# Every base type has to have one of these "kinds".
# Reference: volatility/schemas/schema-6.2.0.json:definitions.element_base_type
# "void" is not covered here because it can only be pointed to, but can not be a field in a struct. sizeof(void) == 0
_BASE_KIND_TO_ATOMIC = {
    "char": AtomicType.String,
    "int": AtomicType.Data,
    "float": AtomicType.Data,
    "bool": AtomicType.Data
}


class AtomicEncoder:
    def __init__(self, syms):
        self.syms = syms

    def base_type_to_atomic(self, base_type: str, pointer_name=None) -> List[AtomicType]:
        """
        Reference: volatility/schemas/schema-6.2.0.json:definitions.element_base_type
        """
        base_type_description = self.syms["base_types"][base_type]
        if base_type_description["kind"] == "void":
            raise Exception("Attempted to interpret void as a struct member, how did you get here?")
        if base_type == "pointer":
            return [AtomicType.Pointer(pointer_name)] * base_type_description["size"]
        return [_BASE_KIND_TO_ATOMIC[base_type_description["kind"]]] * base_type_description["size"]

    def enum_to_atomic(self, enum: str) -> List[AtomicType]:
        # Enums can hardly be discovered in the raw data, they are treated as data.
        return self.syms["enums"][enum]["size"] * [AtomicType.Data]

    @staticmethod
    def _assign_final_type(atoms: List[Tuple[AtomicType, str]]) -> AtomicType:
        """
        Chooses the final atomic type from a list of possible ones for a given offset within a type.
        """
        if not atoms:  # No field covered this offset, has to be padding.
            return AtomicType.Padding
        elif len(set(atoms)) == 1:  # Clear case.
            return atoms[0]
        else:  # Conflict, default to Data.
            return AtomicType.Data

    def user_type_to_atomic(self, user_type: "element_user_type", bread_crumbs=[]) -> List[AtomicType]:
        """
        Reference: volatility/schemas/schema-6.2.0.json:definitions.element_user_type
        """
        user_type = self.syms["user_types"][user_type]
        # TODO: user_type has a "kind" as well, do I care?
        atomic_fields = [
            (name, description["offset"], self.type_descriptor_to_atomic(description["type"],
                                                                         bread_crumbs=bread_crumbs))
            for name, description in user_type["fields"].items()]
        offset_map = defaultdict(lambda: list())
        for name, offset, atoms in atomic_fields:
            atoms = atoms if type(atoms) is list else [atoms]
            for i, atom in enumerate(atoms):
                offset_map[offset + i].append(atom)

        result = [self._assign_final_type(offset_map[i]) for i in range(user_type["size"])]
        return result

    @staticmethod
    def _get_pointer_name(subtype: dict) -> str:
        # According to the schema "subtype" is another type descriptor. Which means this can recurse.
        # But the pointers are only useful when they point to something we can recognize. Thus if we get a pointer
        # to something that is not a struct, we do not follow it for embedding.
        if subtype["kind"] in ["base", "struct", "union", "class"]:
            return subtype["name"]
        else:  # ["array", "bitfield", "pointer", "function", "enum"]
            return subtype["kind"]

    def type_descriptor_to_atomic(self,
                                  type_descriptor: "type_descriptor",
                                  bread_crumbs=[]
                                  ) -> Union[AtomicType, List[AtomicType]]:
        """
        Reference: volatility/schemas/schema-6.2.0.json:definitions.type_descriptor
        :param type_descriptor:
        :param bread_crumbs:
        :return:
        """
        kind = type_descriptor["kind"]
        if kind in ["function", "pointer"]:
            return self.base_type_to_atomic("pointer", pointer_name=self._get_pointer_name(type_descriptor["subtype"]))
        elif kind == "enum":
            return self.enum_to_atomic(type_descriptor["name"])
        elif kind == "base":
            return self.base_type_to_atomic(type_descriptor["name"])
        elif kind == "bitfield":  # Bitfield is just a wrapper around an enum or base_type_description.
            return self.type_descriptor_to_atomic(type_descriptor["type"], bread_crumbs=bread_crumbs)
        elif kind == "array":
            member = self.type_descriptor_to_atomic(type_descriptor["subtype"])
            member = member if type(member) is list else [member]
            return type_descriptor["count"] * member
        elif kind in ["struct", "class", "union"]:  # Reference to a user type
            return self.user_type_to_atomic(type_descriptor["name"], bread_crumbs=bread_crumbs)
