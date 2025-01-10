"""Test the Routing."""

from __future__ import annotations

import sys

import pytest

#import mqt.qecc.co3 as co
sys.path.append("../../../src/mqt/qecc/co3/utils")
import lattice_router as co


@pytest.mark.parametrize(
    ("pos_1", "pos_2", "expected_dist"),
    [
        ((0, 0), (2, 4), 6),
        ((0, 1), (2, 7), 8),
        ((0, 5), (4, 5), 8),
        ((1, 7), (2, 7), 1),
        ((0, 0), (4, 9), 13)
    ]
)
def test_distance_triangular(pos_1, pos_2, expected_dist):
    """Test Distance."""
    # Setup
    m = 4
    n = 4
    lat = co.HexagonalLattice(m, n)
    dist = lat.distance_triangular(pos_1, pos_2)

    error_message = (
        f"Expected distance between {pos_1} and {pos_2} "
        f"to be {expected_dist}, but got {dist}"
    )

    assert dist == expected_dist, error_message


def test_shortest_first_router_1():
    """Test routing and check final layers."""
    terminal_pairs = [
        ((1, 0), (1, 5)),
        ((4, 11), (4, 9)),
        ((4, 7), (2, 7)),
        ((3, 10), (1, 10))
    ]
    m, n = 5, 5
    lat = co.ShortestFirstRouter(m, n, terminal_pairs)

    expected_vdp_layers = [
        {
            ((4, 11), (4, 9)): [(4, 11), (4, 10), (4, 9)],
            ((4, 7), (2, 7)): [(4, 7), (3, 7), (3, 6), (2, 6), (2, 7)],
            ((3, 10), (1, 10)): [(3, 10), (2, 10), (2, 9), (1, 9), (1, 10)],
            ((1, 0), (1, 5)): [(1, 0), (1, 1), (1, 2), (1, 3), (1, 4), (1, 5)],
        }
    ]

    error_message = f"Expected vdp_layers to be {expected_vdp_layers}, but got {lat.vdp_layers}"
    assert lat.vdp_layers == expected_vdp_layers, error_message


def test_shortest_first_router_2():
    """Test number of resulting layers of routing example."""
    terminal_pairs = [
        ((1, 9), (4, 7)),
        ((4, 11), (2, 7)),
        ((0, 5), (0, 2)),
        ((1, 6), (3, 6)),
        ((2, 6), (3, 5))
    ]
    m, n = 5, 5
    lat = co.ShortestFirstRouter(m, n, terminal_pairs)
    error_message = f"Expected 2 layers, but got {len(lat.vdp_layers)}"
    assert len(lat.vdp_layers) == 2, error_message

def test_shortest_first_router_3():
    """Test routing and check final layers for another example."""
    terminal_pairs = [
        ((3, 4), (2, 7)), 
        ((3, 2), (3, 5)), 
        ((0, 2), (1, 3)), 
        ((2, 1), (1, 5)), 
        ((0, 5), (0, 6))]
    m, n = 3, 3
    lat = co.ShortestFirstRouter(m, n, terminal_pairs)

    expected_vdp_layers = [
        {((0, 5), (0, 6)): [(0, 5), (0, 6)],
        ((0, 2), (1, 3)): [(0, 2), (1, 2), (1, 3)],
        ((3, 4), (2, 7)): [(3, 4), (2, 4), (2, 5), (2, 6), (2, 7)]},
        {((2, 1), (1, 5)): [(2, 1), (2, 2), (2, 3), (2, 4), (2, 5), (1, 5)]},
        {((3, 2), (3, 5)): [(3, 2),
        (2, 2),
        (2, 3),
        (2, 4),
        (2, 5),
        (2, 6),
        (3, 6),
        (3, 5)]}
   ]

    error_message = f"Expected vdp_layers to be {expected_vdp_layers}, but got {lat.vdp_layers}"
    assert lat.vdp_layers == expected_vdp_layers, error_message    

def test_standard_qubit_locs():
    """Test allocation of qubits (no logical labels) for the standard layout."""
    data_qubit_locs_expected = [(1, 2), (1, 8), (2, 5), (2, 11), (3, 2), (3, 8), (4, 5), (4, 11), (5, 2), (5, 8)]
    m, n = 6, 6
    lat = co.HexagonalLattice(m, n)
    data_qubit_locs = lat.gen_layout_sparse()
    error_message = "Generation of Standard Layout in HexagonalLattice is faulty."
    assert data_qubit_locs == data_qubit_locs_expected, error_message

def test_pair_qubit_locs():
    """Test allocation of qubits (no logical labels) for the pair layout."""
    data_qubit_locs_expected = [(1, 2), (1, 3), (1, 6), (1, 7), (1, 10), (1, 11), (3, 2), (3, 3), (3, 6), (3, 7), (3, 10), (3, 11), (5, 2), (5, 3), (5, 6), (5, 7), (5, 10), (5, 11)]
    m, n = 6, 6
    lat = co.HexagonalLattice(m, n)
    data_qubit_locs = lat.gen_layout_pair()
    error_message = "Generation of Pair Layout in HexagonalLattice is faulty."
    assert data_qubit_locs == data_qubit_locs_expected, error_message

def test_row_qubit_locs():
    """Test allocation of qubits (no logical labels) for the row layout."""
    data_qubit_locs_expected = [(1, 2), (1, 3), (2, 2), (2, 3), (3, 2), (3, 3), (4, 2), (4, 3), (5, 2), (5, 3), (1, 6), (1, 7), (2, 6), (2, 7), (3, 6), (3, 7), (4, 6), (4, 7), (5, 6), (5, 7), (1, 10), (1, 11), (2, 10), (2, 11), (3, 10), (3, 11), (4, 10), (4, 11), (5, 10), (5, 11)]
    m, n = 6, 6
    lat = co.HexagonalLattice(m, n)
    data_qubit_locs = lat.gen_layout_row()
    error_message = "Generation of Row Layout in HexagonalLattice is faulty."
    assert data_qubit_locs == data_qubit_locs_expected, error_message