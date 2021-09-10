from __future__ import annotations
from typing import Dict, Optional, Union, Literal
from pydantic import BaseModel, validator
from frozendict import frozendict

FrozenDict = Dict

Endian = Literal["big", "little"]

BaseTypeKind = Literal["void", "int", "float", "char", "bool"]

UserTypeKind = Literal["struct", "union", "class"]


class TypePointer(BaseModel, frozen=True):
    kind: Literal["pointer"]
    subtype: TypeDescriptor
    base: Optional[str] = None


class TypeBase(BaseModel, frozen=True):
    kind: Literal["base"]
    name: str


class TypeArray(BaseModel, frozen=True):
    kind: Literal["array"]
    subtype: TypeDescriptor
    count: int


class TypeStruct(BaseModel, frozen=True):
    kind: UserTypeKind
    name: str


class TypeEnum(BaseModel, frozen=True):
    kind: Literal["enum"]
    name: str


class TypeFunction(BaseModel, frozen=True):
    kind: Literal["function"]


class TypeBitField(BaseModel, frozen=True):
    kind: Literal["bitfield"]
    type: Union[TypeBase, TypeEnum]
    bit_position: int
    bit_length: int


TypeDescriptor = Union[TypePointer, TypeBase, TypeArray, TypeStruct, TypeEnum, TypeFunction, TypeBitField]
TypeArray.update_forward_refs()
TypePointer.update_forward_refs()


class Field(BaseModel, frozen=True):
    type: TypeDescriptor
    offset: int
    anonymous: Optional[bool] = None


class ElementEnum(BaseModel, frozen=True):
    size: int
    base: str
    constants: FrozenDict[str, int]

    @validator("constants")
    def validate_dicts(cls, value):
        return frozendict(value)


class ElementSymbol(BaseModel, frozen=True):
    address: int
    type: Optional[TypeDescriptor] = None
    linkage_name: Optional[str] = None
    constant_data: Optional[str] = None


class ElementBaseType(BaseModel, frozen=True):
    size: int
    signed: bool
    kind: BaseTypeKind
    endian: Endian


class ElementUserType(BaseModel, frozen=True):
    size: int
    kind: UserTypeKind
    fields: FrozenDict[str, Field]

    @validator("fields")
    def validate_dicts(cls, value):
        return frozendict(value)


class SymbolContainer(BaseModel, frozen=True):
    # metadata # Don't care about the metadata at the moment
    base_types: FrozenDict[str, ElementBaseType]
    user_types: FrozenDict[str, ElementUserType]
    enums: FrozenDict[str, ElementEnum]
    symbols: FrozenDict[str, ElementSymbol]

    @validator("base_types", "user_types", "enums", "symbols")
    def validate_dicts(cls, value):
        return frozendict(value)
