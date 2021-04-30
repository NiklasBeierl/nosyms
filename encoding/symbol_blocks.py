import json
from collections import defaultdict
from functools import cached_property, lru_cache
from typing import List, Union, Tuple, Callable, Dict
from encoding import BlockType

# Every base type has to have one of these "kinds".
# Reference: volatility/schemas/schema-6.2.0.json:definitions.element_base_type
# "void" is not covered here because it can only be pointed to, but can not be a field in a struct: sizeof(void) == 0
_BASE_KIND_TO_BLOCK = {
    "char": BlockType.String,
    "int": BlockType.Data,
    "float": BlockType.Data,
    "bool": BlockType.Data,
}


def _add_pointer_dict_logic(
    encode_func: Callable[..., List[BlockType]]
) -> Callable[..., Tuple[List[BlockType], Dict[int, "type_descriptor"]]]:
    """
    Wrap an encode function such that it returns a list of purely blocks and additionally returns a dict with the
    pointer offsets and corresponding type_descriptors. This is necessary because encoding user types and type
    descriptors requires recursion, but its easier to figure out the pointers and their offsets as the last step.
    When :func:`_encode_user_type` and :func:`_encode_type_descriptor` hit a pointer they add the pointers
    `subtype` to their result lists as json strings. (Hashable type needed to keep :func:`_select_final_types` simple.)
    Here we json.loads these subtypes if the offsets corresponding to the pointer are uncontested
    (Might be in case of Unions!) and add them to a offset type descriptor dict which will be returned along the list of
    BlockTypes.
    :param encode_func: Encode function to wrap.
    :return: Wrapped encode function.
    """

    def new_encode_func(self, *args, **kwargs) -> Tuple[List[BlockType], Dict[int, str]]:
        blocks: List[BlockType] = encode_func(self, *args, **kwargs)

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

        real_blocks = [b if isinstance(b, BlockType) else BlockType.Pointer for b in blocks]
        return real_blocks, pointer_dict

    new_encode_func.__doc__ = encode_func.__doc__

    return new_encode_func


class VolatilitySymbolsEncoder:
    def __init__(self, syms):
        self.syms = syms

    @cached_property
    def _pointer_size(self):
        return self.syms["base_types"]["pointer"]["size"]

    @staticmethod
    def _select_final_type(blocks: List[Tuple[BlockType, str]]) -> BlockType:
        """
        Chooses the final block type from a list of possible ones for a given offset within a type.
        """
        if not blocks:  # No field covered this offset, has to be padding.
            return BlockType.Zero
        if len(set(blocks)) == 1:  # Clear case.
            return blocks[0]
        # Conflict, default to Data.
        return BlockType.Data

    @lru_cache
    def encode_base_type(self, base_type_name: str) -> List[BlockType]:
        """
        Encode a base type into a list of BlockTypes.
        :param base_type_name: Name of a base type: volatility/schemas/schema-6.2.0.json:definitions.element_base_type
        :return: List of BlockTypes representing the base type.
        """
        base_type_description = self.syms["base_types"][base_type_name]
        if base_type_description["kind"] == "void":
            raise Exception("Attempted to interpret void as a struct member, how did you get here?")
        return [_BASE_KIND_TO_BLOCK[base_type_description["kind"]]] * base_type_description["size"]

    @lru_cache
    def encode_enum(self, enum_name: str) -> List[BlockType]:
        """
        Encode an enum into a list of BlockTypes.
        :param enum_name: Name of an enum: volatility/schemas/schema-6.2.0.json:definitions.element_enum
        :return: List of BlockTypes representing the enum.
        """
        return self.syms["enums"][enum_name]["size"] * [BlockType.Data]

    @lru_cache
    def _encode_user_type(self, user_type_name: str) -> List[Union[BlockType, str]]:
        """
        Encode a user type into a list of BlockTypes and a dict with its points-to relationships.
        :param user_type_name: Name of a user type: volatility/schemas/schema-6.2.0.json:definitions.element_user_type
        :return: List of BlockTypes representing the user type, a dict with points to relationships.
        """
        user_type = self.syms["user_types"][user_type_name]
        field_blocks = [
            (description["offset"], self._encode_type_descriptor(description["type"]))
            for description in user_type["fields"].values()
        ]
        offset_map = defaultdict(list)
        for offset, blocks in field_blocks:
            blocks = blocks if isinstance(blocks, list) else [blocks]
            for i, block in enumerate(blocks):
                offset_map[offset + i].append(block)

        result = [self._select_final_type(offset_map[i]) for i in range(user_type["size"])]
        return result

    def _encode_type_descriptor(self, type_descriptor: "type_descriptor") -> List[Union[BlockType, str]]:
        """
        Encode a type_descriptor into a list of BlockTypes and a dict with its points-to relationships.
        :param type_descriptor: Json dict: volatility/schemas/schema-6.2.0.json:definitions.type_descriptor
        :return: List of BlockTypes representing the type descriptor, a dict with points-to relationships.
        """
        kind = type_descriptor["kind"]
        if kind == "pointer":  # TODO: pointers can have a "base" attribute, what does it mean?
            subtype_descriptor_str = json.dumps(type_descriptor["subtype"])
            return [subtype_descriptor_str] * self._pointer_size
        elif kind == "function":  # We would need the ability to identify functions to further leverage that.
            return [BlockType.Pointer] * self._pointer_size
        elif kind == "enum":
            return self.encode_enum(type_descriptor["name"])
        elif kind == "base":
            return self.encode_base_type(type_descriptor["name"])
        elif kind == "bitfield":  # Bitfield is just a wrapper around an enum or base_type
            return self._encode_type_descriptor(type_descriptor["type"])
        elif kind == "array":
            member = self._encode_type_descriptor(type_descriptor["subtype"])
            member = member if isinstance(member, list) else [member]
            return member * type_descriptor["count"]
        elif kind in ["struct", "class", "union"]:  # Reference to a user type
            return self._encode_user_type(type_descriptor["name"])
        else:
            raise ValueError(f"Unknown 'kind' encountered in type_descriptor: {kind}")

    encode_user_type = _add_pointer_dict_logic(_encode_user_type)
    encode_type_descriptor = _add_pointer_dict_logic(_encode_type_descriptor)
