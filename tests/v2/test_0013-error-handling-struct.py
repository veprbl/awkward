# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import pytest  # noqa: F401
import numpy as np  # noqa: F401
import awkward as ak  # noqa: F401


def test_numpyarray():
    array = ak._v2.contents.NumpyArray(np.arange(10) * 1.1)

    with pytest.raises(IndexError):
        array[20]

    with pytest.raises(IndexError):
        array[-20]

    array[-20:20]

    with pytest.raises(IndexError):
        array[
            20,
        ]

    with pytest.raises(IndexError):
        array[
            -20,
        ]

    array[
        -20:20,
    ]

    with pytest.raises(IndexError):
        array[2, 3]

    with pytest.raises(IndexError):
        array[[5, 3, 20, 8]]

    with pytest.raises(IndexError):
        array[[5, 3, -20, 8]]

    with pytest.raises(IndexError):
        array[20]

    with pytest.raises(IndexError):
        array[-20]

    array[-20:20]

    with pytest.raises(IndexError):
        array[
            20,
        ]

    with pytest.raises(IndexError):
        array[
            -20,
        ]

    array[
        -20:20,
    ]

    with pytest.raises(IndexError):
        array[2, 3]

    with pytest.raises(IndexError):
        array[[5, 3, 20, 8]]

    with pytest.raises(IndexError):
        array[[5, 3, -20, 8]]


def test_listarray_numpyarray():
    starts = ak._v2.index.Index64(np.array([0, 3, 3, 5, 6]))
    stops = ak._v2.index.Index64(np.array([3, 3, 5, 6]))
    content = ak._v2.contents.NumpyArray(np.arange(10) * 1.1)

    with pytest.raises(ValueError):
        array = ak._v2.contents.listarray.ListArray(starts, stops, content)

    starts = ak._v2.index.Index64(np.array([0, 3, 3, 5, 6]))
    stops = ak._v2.index.Index64(np.array([3, 3, 5, 6, 10]))
    content = ak._v2.contents.NumpyArray(np.arange(10) * 1.1)
    array = ak._v2.contents.ListArray(starts, stops, content)

    with pytest.raises(IndexError):
        array[20]

    with pytest.raises(IndexError):
        array[-20]

    array[-20:20]

    with pytest.raises(IndexError):
        array[
            20,
        ]

    with pytest.raises(IndexError):
        array[
            -20,
        ]

    array[
        -20:20,
    ]

    with pytest.raises(IndexError):
        array[2, 1, 0]

    with pytest.raises(IndexError):
        array[[2, 0, 0, 20, 3]]

    with pytest.raises(IndexError):
        array[[2, 0, 0, -20, 3]]

    starts = ak._v2.index.Index64(np.array([0, 3, 3, 5, 6]))
    stops = ak._v2.index.Index64(np.array([3, 3, 5, 6, 10]))
    content = ak._v2.contents.NumpyArray(np.arange(10) * 1.1)
    array = ak._v2.contents.ListArray(starts, stops, content)

    with pytest.raises(IndexError):
        array[2, 20]

    with pytest.raises(IndexError):
        array[2, -20]

    with pytest.raises(IndexError):
        array[1:][2, 20]

    with pytest.raises(IndexError):
        array[1:][2, -20]

    with pytest.raises(IndexError):
        array[2, [1, 0, 0, 20]]

    with pytest.raises(IndexError):
        array[2, [1, 0, 0, -20]]

    with pytest.raises(IndexError):
        array[1:][2, [0, 20]]

    with pytest.raises(IndexError):
        array[1:][2, [0, -20]]


def test_listarray_listarray_numpyarray():
    content = ak._v2.contents.NumpyArray(
        np.array([1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9])
    )
    starts1 = ak._v2.index.Index64(np.array([0, 3, 3, 5, 6]))
    stops1 = ak._v2.index.Index64(np.array([3, 3, 5, 6, 9]))
    starts2 = ak._v2.index.Index64(np.array([0, 2, 3, 3]))
    stops2 = ak._v2.index.Index64(np.array([2, 3, 3, 5]))

    array1 = ak._v2.contents.ListArray(starts1, stops1, content)
    array2 = ak._v2.contents.ListArray(starts2, stops2, array1)

    with pytest.raises(IndexError):
        array2[20]

    with pytest.raises(IndexError):
        array2[
            20,
        ]

    with pytest.raises(IndexError):
        array2[2, 20]

    with pytest.raises(IndexError):
        array2[-20]

    with pytest.raises(IndexError):
        array2[
            -20,
        ]

    with pytest.raises(IndexError):
        array2[2, -20]

    with pytest.raises(IndexError):
        array2[1, 0, 20]

    with pytest.raises(IndexError):
        array2[20]

    with pytest.raises(IndexError):
        array2[
            20,
        ]

    with pytest.raises(IndexError):
        array2[2, 20]

    with pytest.raises(IndexError):
        array2[1:][2, 20]

    with pytest.raises(IndexError):
        array2[-20]

    with pytest.raises(IndexError):
        array2[
            -20,
        ]

    with pytest.raises(IndexError):
        array2[2, -20]

    with pytest.raises(IndexError):
        array2[1:][2, -20]

    with pytest.raises(IndexError):
        array2[1, 0, 20]

    with pytest.raises(IndexError):
        array2[1:][2, 0, 20]

    with pytest.raises(IndexError):
        array2[:, 1:][3, 0, 20]
