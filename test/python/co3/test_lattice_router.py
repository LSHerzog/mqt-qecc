import pytest
import sys
# from mqt.qecc.co3 import lattice_router as lr
sys.path.append("../../../src/mqt/qecc/co3/utils")
import lattice_router as lr

@pytest.mark.parametrize(
    "pos_1, pos_2, expected_dist",
    [
        ((0, 0), (2, 4), 6),
        ((0, 1), (2, 7), 8),
        ((0, 5), (4, 5), 8),
        ((1, 7), (2, 7), 1),
        ((0, 0), (4, 9), 13)
    ]
)
def test_distance_triangular(pos_1, pos_2, expected_dist):
    # Setup
    m = 4
    n = 4
    lat = lr.hexagonal_lattice(m, n)
    dist = lat.distance_triangular(pos_1, pos_2)

    error_message = (
        f"Expected distance between {pos_1} and {pos_2} "
        f"to be {expected_dist}, but got {dist}"
    )

    assert dist == expected_dist, error_message


def test_shortest_first_router():
    terminal_pairs = [
        ((1, 0), (1, 5)),
        ((4, 11), (4, 9)),
        ((4, 7), (2, 7)),
        ((3, 10), (1, 10))
    ]
    m, n = 5, 5
    lat = lr.shortest_first_router(m, n, terminal_pairs)

    expected_VDP_layers = [
        {
            ((4, 11), (4, 9)): [(4, 11), (4, 10), (4, 9)],
            ((4, 7), (2, 7)): [(4, 7), (3, 7), (3, 6), (2, 6), (2, 7)],
            ((3, 10), (1, 10)): [(3, 10), (2, 10), (2, 9), (1, 9), (1, 10)],
            ((1, 0), (1, 5)): [(1, 0), (1, 1), (1, 2), (1, 3), (1, 4), (1, 5)],
        }
    ]

    error_message = f"Expected VDP_layers to be {expected_VDP_layers}, but got {lat.VDP_layers}"
    assert lat.VDP_layers == expected_VDP_layers, error_message


def test_shortest_first_router_2():
    terminal_pairs = [
        ((1, 9), (4, 7)),
        ((4, 11), (2, 7)),
        ((0, 5), (0, 2)),
        ((1, 6), (3, 6)),
        ((2, 6), (3, 5))
    ]
    m, n = 5, 5
    lat = lr.shortest_first_router(m, n, terminal_pairs)
    error_message = f"Expected 2 layers, but got {len(lat.VDP_layers)}"
    assert len(lat.VDP_layers) == 2, error_message
