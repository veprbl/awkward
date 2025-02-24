# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import pytest  # noqa: F401
import numpy as np  # noqa: F401
import awkward as ak  # noqa: F401

to_list = ak._v2.operations.to_list


def test():
    assert to_list(
        ak._v2.operations.prod(
            ak._v2.highlevel.Array([[[2, 3, 5]], [[7], [11]], [[]]]), axis=-1
        )
    ) == [
        [30],
        [7, 11],
        [1],
    ]

    assert to_list(
        ak._v2.operations.prod(
            ak._v2.highlevel.Array([[[2, 3, 5]], [[7], [11]], []]), axis=-1
        )
    ) == [
        [30],
        [7, 11],
        [],
    ]
