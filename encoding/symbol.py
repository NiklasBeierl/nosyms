import json
from collections import defaultdict
from functools import cached_property, lru_cache
from typing import List, Union, Tuple, Callable, Dict
from encoding.types import AtomicType

# Every base type has to have one of these "kinds".
# Reference: volatility/schemas/schema-6.2.0.json:definitions.element_base_type
# "void" is not covered here because it can only be pointed to, but can not be a field in a struct: sizeof(void) == 0
_BASE_KIND_TO_ATOMIC = {
    "char": AtomicType.String,
    "int": AtomicType.Data,
    "float": AtomicType.Data,
    "bool": AtomicType.Data,
}


def _add_pointer_dict_logic(
    encode_func: Callable[..., List[AtomicType]]
) -> Callable[..., Tuple[List[AtomicType], Dict[int, "type_descriptor"]]]:
    """
    Wrap an encode function such that it returns a list of purely blocks and additionally returns a dict with the
    pointer offsets and corresponding type_descriptors.
    :param encode_func: Encode function to wrap.
    :return: Wrapped encode function.
    """

    def new_encode_func(self, *args, **kwargs) -> Tuple[List[AtomicType], Dict[int, str]]:
        blocks: List[AtomicType] = encode_func(self, *args, **kwargs)

        pointer_dict: Dict[int, str] = {}
        offset = 0
        while offset < len(blocks):
            block = blocks[offset]
            if isinstance(block, str):
                assert offset % self._pointer_size == 0
                # All bytes hold the same type_descriptor
                if len(set(blocks[offset : offset + self._pointer_size])) == 1:
                    pointer_dict[offset] = json.loads(block)
                offset += self._pointer_size
            else:
                offset += 1

        real_blocks = [b if isinstance(b, AtomicType) else AtomicType.Pointer for b in blocks]
        return real_blocks, pointer_dict

    new_encode_func.__doc__ = encode_func.__doc__

    return new_encode_func


class SymbolEncoder:
    def __init__(self, syms):
        self.syms = syms

    @cached_property
    def _pointer_size(self):
        return self.syms["base_types"]["pointer"]["size"]

    @staticmethod
    def _select_final_type(atoms: List[Tuple[AtomicType, str]]) -> AtomicType:
        """
        Chooses the final atomic type from a list of possible ones for a given offset within a type.
        """
        if not atoms:  # No field covered this offset, has to be padding.
            return AtomicType.Padding
        if len(set(atoms)) == 1:  # Clear case.
            return atoms[0]
        # Conflict, default to Data.
        return AtomicType.Data

    @lru_cache
    def base_type_to_atomic(self, base_type_name: str) -> List[AtomicType]:
        """
        Reference: volatility/schemas/schema-6.2.0.json:definitions.element_base_type
        """
        base_type_description = self.syms["base_types"][base_type_name]
        if base_type_description["kind"] == "void":
            raise Exception("Attempted to interpret void as a struct member, how did you get here?")
        return [_BASE_KIND_TO_ATOMIC[base_type_description["kind"]]] * base_type_description["size"]

    @lru_cache
    def enum_to_atomic(self, enum_name: str) -> List[AtomicType]:
        """
        Reference: volatility/schemas/schema-6.2.0.json:definitions.element_enum
        """
        return self.syms["enums"][enum_name]["size"] * [AtomicType.Data]

    @lru_cache
    def _user_type_to_atomic(self, user_type: str) -> List[Union[AtomicType, str]]:
        """
        Reference: volatility/schemas/schema-6.2.0.json:definitions.element_user_type
        """
        user_type = self.syms["user_types"][user_type]
        atomic_fields = [
            (description["offset"], self._type_descriptor_to_atomic(description["type"]))
            for description in user_type["fields"].values()
        ]
        offset_map = defaultdict(lambda: list())
        for offset, atoms in atomic_fields:
            atoms = atoms if type(atoms) is list else [atoms]
            for i, atom in enumerate(atoms):
                offset_map[offset + i].append(atom)

        result = [self._select_final_type(offset_map[i]) for i in range(user_type["size"])]
        return result

    def _type_descriptor_to_atomic(self, type_descriptor: "type_descriptor") -> List[Union[AtomicType, str]]:
        """
        Reference: volatility/schemas/schema-6.2.0.json:definitions.type_descriptor
        """
        kind = type_descriptor["kind"]
        if kind == "pointer":  # TODO: pointers can have a "base" attribute, what does it mean?
            subtype_descriptor_str = json.dumps(type_descriptor["subtype"])
            return [subtype_descriptor_str] * self._pointer_size
        elif kind == "function":  # We would need the ability to identify functions to further leverage that.
            return [AtomicType.Pointer] * self._pointer_size
        elif kind == "enum":
            return self.enum_to_atomic(type_descriptor["name"])
        elif kind == "base":
            return self.base_type_to_atomic(type_descriptor["name"])
        elif kind == "bitfield":  # Bitfield is just a wrapper around an enum or base_type
            return self._type_descriptor_to_atomic(type_descriptor["type"])
        elif kind == "array":
            member = self._type_descriptor_to_atomic(type_descriptor["subtype"])
            member = member if isinstance(member, list) else [member]
            return member * type_descriptor["count"]
        elif kind in ["struct", "class", "union"]:  # Reference to a user type
            return self._user_type_to_atomic(type_descriptor["name"])
        else:
            raise ValueError(f"Unknown 'kind' encountered in type_descriptor: {kind}")

    user_type_to_atomic = _add_pointer_dict_logic(_user_type_to_atomic)
    type_descriptor_to_atomic = _add_pointer_dict_logic(_type_descriptor_to_atomic)
