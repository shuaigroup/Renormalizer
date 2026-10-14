# -*- coding: utf-8 -*-
# Author: Jiajun Ren <jiajunren0522@gmail.com>
#         Weitang Li <liwt31@163.com>

from __future__ import division

import math
import logging
import numpy as np

from renormalizer.utils import constant

logger = logging.getLogger(__name__)

au_ratio_dict = {
    "meV": constant.au2ev * 1e3,
    "eV": constant.au2ev,
    "cm^{-1}": 1 / constant.cm2au,
    "cm-1": 1 / constant.cm2au,
    "K": constant.au2K,
    "a.u.": 1,
    "au": 1,
    "fs": constant.au2fs,
    "ps": constant.au2ps
}

au_ratio_dict.update({k.lower(): v for k, v in au_ratio_dict.items()})

allowed_units = set(au_ratio_dict.keys())


def convert_to_au(num, unit):
    """Convert numeric value(s) to atomic units, handling both scalars and arrays."""
    if isinstance(num, (list, tuple, np.ndarray)):
        return np.array(num) / au_ratio_dict[unit]
    else:
        return num / au_ratio_dict[unit]


class Quantity:
    """
    Unified Quantity class using duck typing and property delegation.
    Maintains full backward compatibility while supporting both scalar and array values.
    """

    def __init__(self, value, unit="a.u."):
        if unit not in allowed_units:
            raise ValueError(f"Unit not in {allowed_units}, got {unit}.")

        # Store as numpy array for unified handling
        self._data = np.array(value, dtype=float, ndmin=1)  # Ensure at least 1D
        self.unit = unit

        # Temperature warning (only for scalar temperatures)
        if (
            self._data.size == 1
            and self._data[0] < 0.1
            and self._data[0] != 0
            and unit.lower() in ["k", "kelvin"]
        ):
            logger.warning("Temperature too low and might cause numerical errors")

    @property
    def value(self):
        """
        Scalar access - returns single value for 1-element arrays,
        raises error for multi-element arrays.
        """
        if self._data.size == 1:
            return float(self._data[0])
        else:
            raise AttributeError(
                f"Cannot access .value on multi-element Quantity. "
                f"Use .values or indexing. Shape: {self._data.shape}"
            )

    @property
    def values(self):
        """Array access - always returns numpy array."""
        return self._data

    @property
    def shape(self):
        """Shape property for numpy compatibility."""
        return self._data.shape

    @property
    def size(self):
        """Size property for numpy compatibility."""
        return self._data.size

    def as_au(self):
        """Convert to atomic units."""
        if self._data.size == 1:
            return convert_to_au(float(self._data[0]), self.unit)
        else:
            return convert_to_au(self._data, self.unit)

    def as_unit(self, unit):
        """Convert to specified unit."""
        au_value = self.as_au()
        new_value = au_value * au_ratio_dict[unit]
        return Quantity(new_value, unit)

    def to_beta(self):
        """Convert temperature to beta (1/kT)."""
        au_value = self.as_au()

        if self._data.size == 1:
            if au_value == 0:
                return math.inf
            return 1.0 / au_value
        else:
            # For arrays, return list of beta values
            result = []
            for val in au_value if isinstance(au_value, np.ndarray) else [au_value]:
                if val == 0:
                    result.append(math.inf)
                else:
                    result.append(1.0 / val)
            return result

    # Compatiblity with numpy methods
    __array_priority__ = 1000  # Use higher priority to override numpy
    
    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        if method != '__call__':
            return NotImplemented
            
        new_inputs = []
        for inp in inputs:
            if isinstance(inp, Quantity):
                new_inputs.append(inp.as_au())
            else:
                new_inputs.append(inp)
                
        result = ufunc(*new_inputs, **kwargs)
        
        return Quantity(result)
    
    def __array_function__(self, func, types, args, kwargs):
        new_args = []
        for arg in args:
            if isinstance(arg, Quantity):
                new_args.append(arg.as_au())
            else:
                new_args.append(arg)
                
        result = func(*new_args, **kwargs)
        
        return Quantity(result)
    
    # Arithmetic operations
    def __neg__(self):
        return Quantity(-self._data, self.unit)

    def __add__(self, other):
        if not isinstance(other, Quantity):
            raise TypeError("Can only add Quantity objects")

        return Quantity(self.as_au() + other.as_au())

    def __sub__(self, other):
        if not isinstance(other, Quantity):
            raise TypeError("Can only subtract Quantity objects")
        
        return Quantity(self.as_au() - other.as_au())

    def __mul__(self, other):
        if isinstance(other, Quantity):
            raise TypeError("Multiplication between Quantity objects is not supported")

        return Quantity(self.as_au() * other)

    def __rmul__(self, other):
        if isinstance(other, Quantity):
            raise TypeError("Multiplication between Quantity objects is not supported")

        return Quantity(other * self.as_au())

    def __truediv__(self, other):
        if isinstance(other, Quantity):
            raise TypeError("Division between Quantity objects is not supported")

        return Quantity(self.as_au() / other)

    def __eq__(self, other):
        if hasattr(other, "as_au"):
            return np.array_equal(self.as_au(), other.as_au())
        elif other == 0:
            return np.all(self._data == 0)
        else:
            raise TypeError(f"Quantity can only compare with Quantity or 0, not {type(other)}")

    def __ne__(self, other):
        return not self == other

    # Array-like operations
    def __len__(self):
        return len(self._data)

    def __getitem__(self, index):
        result = self._data[index]
        # Return scalar Quantity for single elements, array Quantity for slices
        if isinstance(result, np.ndarray) and result.ndim > 0:
            return Quantity(result, self.unit)
        else:
            return Quantity(float(result), self.unit)

    def __iter__(self):
        for value in self._data:
            yield Quantity(float(value), self.unit)

    # Numpy compatibility
    def __array__(self):
        """Allow numpy to convert this to array."""
        return self._data

    def __str__(self):
        if self._data.size == 1:
            return f"{float(self._data[0])} {self.unit}"
        else:
            # NumPy 2: list(...) would print np.float64(...); keep plain floats.
            return f"{[float(x) for x in self._data]} {self.unit}"

    def __repr__(self):
        if self._data.size == 1:
            return f"Quantity({float(self._data[0])}, '{self.unit}')"
        else:
            # NumPy 2: same as __str__.
            return f"Quantity({[float(x) for x in self._data]}, '{self.unit}')"

    # TODO: magic methods such as `__lt__` and so on


def parse_quantity_str(quantity_str: str):
    """
    Parse a string containing a quantity or list of quantities with units.
    
    Args:
        quantity_str: String in format "value unit" or "value1,value2,value3 unit"
    
    Returns:
        Quantity object
    
    Raises:
        ValueError: If input is invalid or cannot be parsed
    """
    if not quantity_str or not isinstance(quantity_str, str):
        raise ValueError("Input must be a non-empty string")
    
    cleaned_str = quantity_str.strip()
    
    if not cleaned_str:
        raise ValueError("Input string cannot be empty")
    
    # Try to find a known unit at the end of the string
    found_unit = None
    for unit in sorted(allowed_units, key=len, reverse=True):
        if cleaned_str.endswith(unit):
            unit_start = len(cleaned_str) - len(unit)
            if unit_start == 0:
                continue
            
            if unit_start > 0 and cleaned_str[unit_start-1].isspace():
                found_unit = unit
                value_str = cleaned_str[:unit_start].strip()
                break
            elif unit_start > 0 and not cleaned_str[unit_start-1].isspace():
                continue
    
    if found_unit:
        unit_str = found_unit
    else:
        # Fallback to last space method
        last_space_index = cleaned_str.rfind(' ')
        
        if last_space_index == -1:
            value_str = cleaned_str
            unit_str = ""
        else:
            value_str = cleaned_str[:last_space_index].strip()
            unit_str = cleaned_str[last_space_index:].strip()
            
            if unit_str and unit_str not in allowed_units:
                value_str = cleaned_str
                unit_str = ""
    
    # Parse values
    if ',' in value_str:
        try:
            values = [float(val.strip()) for val in value_str.split(',') if val.strip()]
            return Quantity(values, unit_str)
        except ValueError as e:
            raise ValueError(f"Invalid numeric values '{value_str}': {e}")
    else:
        try:
            value = float(value_str)
            return Quantity(value, unit_str)
        except ValueError as e:
            raise ValueError(f"Invalid numeric value '{value_str}': {e}")
        