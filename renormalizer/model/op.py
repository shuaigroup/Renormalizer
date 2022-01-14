# -*- coding: utf-8 -*-
from collections import defaultdict
from itertools import chain
from typing import List, Union, Tuple, Dict

import numpy as np

from renormalizer.utils import Quantity


class Op:
    r"""
    The operator class. The class can be considered as a symbolic way to express
    quantum operator such as :math:`a^\dagger_i a_j`, :math:`b^\dagger_i + b_i`
    or :math:`g \omega a^\dagger_i a_i (b^\dagger_{ij} + b_{ij})`.
    Multiplications between operators are supported.

    Parameters
    ----------
    symbol : str
        The string of the operator, such as ``"a"``, ``"a^\dagger a"`` and ``r"b^\dagger + b"``.
        The supported operators are defined in :class:`~renormalizer.model.basis.BasisSet`.
        For complex symbols consisting of multiplication of simple symbols,
        separate the simple symbols with a single space.
    dof : hashable object or :class:`list` of hashable objects.
        The name of the DoF related to the operator.
        For simple symbol such as ``"a"``, the type could be any hashable object.
        :class:`int`, :class:`str` and :class:`tuple` of them are recommended types for ``dof``.
        For complex symbol such as ``"a^\dagger a"``, the type should be a list of hashable objects,
        with each element representing one of the simple symbol contained in the complex symbol.
        Using a single hashable object is also supported for complex symbol,
        in which case every symbol is assumed to share the same DoF name.
    factor : :class:`float` or :class:`complex` or :class:`Quantity`
        The prefactor of the operator.
    qn : :class:`int` or :class:`list` of :class:`int`.
        The quantum number of the symbol. For simple symbol the ``qn`` should be an :class:`int`.
        For complex symbol quantum number of each simple symbol contained in the complex symbol
        should be provided. If ``qn`` is set to `None`, then the quantum number for every symbol is set to 0
        except for``"a^\dagger"`` and ``"a"``, whose quantum number are set to 1 and -1 respectively.

    Notes
    =====
    Symbols connected by plus ``"+"`` such as ``r"b^\dagger + b"`` is considered as a simple symbol,
    because this class is designed to deal with operator multiplication rather than addition.

    Warnings
    ========
    If you wish to specify the DoF of each symbol in a complex symbol,
    you should use a :class:`list` instead of :class:`tuple`.
    Because in the latter case the :class:`tuple` is recognized as a single DoF
    and the DoFs of all simple symbols are set to the :class:`tuple`.

    """

    @classmethod
    def product(cls, op_list: List["Op"]):
        """
        Construct a new operator as a multiplication product of operators.

        Parameters
        ----------
        op_list : :class:`list` of :class:`Op`
            The operators to be multiplied.
        Returns
        -------
        op : :class:`Op`
            The product operator.
        """
        symbol = " ".join(op.symbol for op in op_list)
        dof_name = list(chain.from_iterable(op.dofs for op in op_list))
        factor = np.product([op.factor for op in op_list])
        qn = list(chain.from_iterable(op.qn_list for op in op_list))
        return Op(symbol, dof_name, factor, qn)

    @classmethod
    def identity(cls, dof):
        """
        Construct identity operator.
        """
        if isinstance(dof, list):
            return cls(" ".join(["I"] * len(dof)), dof)
        else:
            return cls("I", dof)

    def __init__(self, symbol: str, dof, factor: Union[float, Quantity] = 1.0, qn: Union[List, int] = None):
        # This is one of the most external user interface, so detailed argument checking is necessary
        if not isinstance(symbol, str):
            raise TypeError(f"symbol should be a str. Got {symbol} as {type(symbol)}")
        self.symbol: str = symbol
        # replace " + " so that " " can be used to split the symbol ("b^\dagger + b")
        # the logic of Op is based on multiplication of symbols. So special treatment on addition is inevitable
        self.split_symbol : List[str] = [s.replace("plus", " + ") for s in symbol.replace(" + ", "plus").split(" ")]
        num_simple_symbol = len(self.split_symbol)
        # simple symbol
        if num_simple_symbol == 1:
            # dof_name is a list. Check it's valid
            if isinstance(dof, list):
                assert len(dof) == 1
                dofs = dof
            # convert to list for unified treatment
            else:
                dofs = [dof]
            # same for qn
            if isinstance(qn, list):
                assert len(qn) == 1
                qn_list = qn
            else:
                qn_list = [qn]
        # complex symbol
        else:
            # check dof_name length
            if isinstance(dof, list):
                if num_simple_symbol != len(dof):
                    raise ValueError("symbol and DoF name not match")
                dofs = dof
            # other types. Assuming sharing the same dof name
            else:
                dofs = [dof] * num_simple_symbol
            # check qn
            if isinstance(qn, list):
                if num_simple_symbol != len(qn):
                    raise ValueError("symbol and qn length not match")
                qn_list = qn
            # the default value
            elif qn is None:
                qn_list = [None] * num_simple_symbol
            # Can't assume the same qn as Dof name above
            else:
                raise ValueError("qn should be a list.")

        for dof in dofs:
            if dof.__hash__ is None:
                raise ValueError(f"dof name should be hashable. Got {dof}.")
        for qn in qn_list:
            if qn is not None and not isinstance(qn, int):
                raise TypeError(f"qn for each symbol should be an integer. Got {qn_list}.")

        # argument checking done.
        assert len(dofs) == len(self.split_symbol)

        self.dofs: List = dofs
        if isinstance(factor, Quantity):
            factor = factor.as_au()
        self._factor: float = factor + 0.0 # convert to float
        self.qn_list: List[int] = qn_list

    def split_elementary(self, dof_to_siteidx) -> Tuple[List["Op"], Union[float, complex]]:
        """
        Construct elementary operators according to site index.
        "elementary operator" means that in the operator every DoF is on the same MPS site.

        Parameters
        ----------
        dof_to_siteidx : dict
            Mapping from DoF name to MPS site index.

        Returns
        -------
        elementary_operators : :class:`list` of :class:`Op`
            A list of elementary operators. Factors are set to 1.
        factor : :class:`float` or class:`complex`
            Factor of the operator.
        """
        # simplest case
        if len(self.dofs) == 1:
            return [Op(self.symbol, self.dofs, qn=self.qn_list)], self.factor
        # group operators according to site index.
        # The same site idx (the same basis) in one group.
        grouped_op_info: Dict[int, List[Op]] = defaultdict(list)
        for elem_symbol, elem_name, qn in zip(self.split_symbol, self.dofs, self.qn_list):
            site_idx = dof_to_siteidx.get(elem_name)
            if site_idx is None:
                raise ValueError(f"Unknown DoF name {elem_name} in {self}.")
            # Note that the order of operators on each site is not changed
            grouped_op_info[site_idx].append(Op(elem_symbol, elem_name, qn=qn))
        # Construct elementary operators. Small site index first.
        ops = []
        for site_idx in sorted(grouped_op_info.keys()):
            elem_ops: List[Op] = grouped_op_info[site_idx]
            ops.append(Op.product(elem_ops))
        return ops, self.factor

    @property
    def factor(self):
        """
        The factor of the operator.
        """
        # forbid rewriting factor since Op should be immutable
        return self._factor

    @property
    def qn(self) -> int:
        r"""
        Total quantum number of the operator. Sum of ``self.qn_list``.
        Quantum number of ``"a^\dagger"`` and ``"a"`` is taken to be 1 and -1
        if not specified.
        """
        # should be the most common case
        if set(self.qn_list) == {None}:
            return self.split_symbol.count(r"a^\dagger") - self.split_symbol.count("a")
        # None and integer both in self._qn_list. Not good
        elif None in set(self.qn_list):
            raise ValueError(f"qn ill-defined in {self}")
        # most general case
        else:
            return sum(self.qn_list)

    def to_tuple(self):
        """
        Convert the operator into a tuple. The fields are the symbol, the DoFs
        the factor and the quantum number list. The converted tuple can be hashed.

        Returns
        -------
        op_tuple : tuple
            The converted tuple.
        """
        return self.symbol, tuple(self.dofs), self.factor, tuple(self.qn_list)

    def __hash__(self):
        return hash(self.to_tuple())

    def __eq__(self, other):
        # compare float point directly because generally a small epsilon
        # between floats can not be defined for all possible cases.
        return self.to_tuple() == other.to_tuple()

    def __str__(self):
        return str(self.to_tuple())

    def __mul__(self, other) -> "Op":
        # multiplication with another Op or with scalar
        # convert numpy scalar to python scalar
        if isinstance(other, np.generic):
            other = other.item()
        if isinstance(other, Op):
            return Op.product([self, other])
        elif isinstance(other, (int, float, complex)):
            new_symbol = self.symbol
            new_dof_name = self.dofs
            new_factor = self.factor * other
            new_qn = self.qn_list
        else:
            raise ValueError(f"Unsupported type: {type(other)}")
        return Op(new_symbol, new_dof_name, new_factor, new_qn)

    def __rmul__(self, other) -> "Op":
        # Operators not allowed due to commutation property.
        # In principle Op * Op should use `__mul__` method.
        assert isinstance(other, (int, float, complex, np.generic))
        return self * other
