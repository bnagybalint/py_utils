import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple

from numpy.core.fromnumeric import clip
from geom.clip import clip_line

VMIN = np.array([0,0,0])
VMAX = np.array([4,6,8])

def _is_outside(line: np.ndarray) -> bool:
    return np.allclose(line[0], line[1])

def test_3d_single_line_trivially_inside():
    assert np.allclose(clip_line(np.array([[2,2,2], [4,2,2]]), VMIN, VMAX), np.array([[2,2,2], [4,2,2]])), "Trivially inside (x-aligned)"
    assert np.allclose(clip_line(np.array([[2,2,2], [2,4,2]]), VMIN, VMAX), np.array([[2,2,2], [2,4,2]])), "Trivially inside (y-aligned)"
    assert np.allclose(clip_line(np.array([[2,2,2], [2,2,4]]), VMIN, VMAX), np.array([[2,2,2], [2,2,4]])), "Trivially inside (z-aligned)"
    assert np.allclose(clip_line(np.array([[1,5,4], [2,4,6]]), VMIN, VMAX), np.array([[1,5,4], [2,4,6]])), "Trivially inside"
    assert np.allclose(clip_line(np.array([[2,4,1], [3,5,2]]), VMIN, VMAX), np.array([[2,4,1], [3,5,2]])), "Trivially inside"

def test_3d_single_line_trivially_outside():
    assert _is_outside(clip_line(np.array([[-2,3,4], [-3,5,6]]), VMIN, VMAX)), "Trivially outside because of x"
    assert _is_outside(clip_line(np.array([[12,3,4], [13,5,6]]), VMIN, VMAX)), "Trivially outside because of x"
    assert _is_outside(clip_line(np.array([[2,-3,4], [3,-5,6]]), VMIN, VMAX)), "Trivially outside because of y"
    assert _is_outside(clip_line(np.array([[2,12,4], [3,15,6]]), VMIN, VMAX)), "Trivially outside because of y"
    assert _is_outside(clip_line(np.array([[2,3,-4], [3,5,-6]]), VMIN, VMAX)), "Trivially outside because of z"
    assert _is_outside(clip_line(np.array([[2,3,14], [3,5,16]]), VMIN, VMAX)), "Trivially outside because of z"

def test_3d_single_crossing_opposite_faces():
    assert np.allclose(clip_line(np.array([[-2,3,4], [13,3,4]]), VMIN, VMAX), np.array([[0,3,4], [4,3,4]])), "opposite faces, x-aligned"
    assert np.allclose(clip_line(np.array([[2,-3,4], [2,13,4]]), VMIN, VMAX), np.array([[2,0,4], [2,6,4]])), "opposite faces, y-aligned"
    assert np.allclose(clip_line(np.array([[2,3,-4], [2,3,14]]), VMIN, VMAX), np.array([[2,3,0], [2,3,8]])), "opposite faces, z-aligned"

    assert np.allclose(clip_line(np.array([[-2,1,1], [6,3,5]]), VMIN, VMAX), np.array([[0.0,1.5,2.0], [4.0,2.5,4.0]])), "opposite faces"

def test_3d_single_crossing_neighbouring_faces():
    assert np.allclose(clip_line(np.array([[2,-1,4], [6,3,0]]), VMIN, VMAX), np.array([[3.0,0.0,3.0], [4.0,1.0,2.0]])), "neighbouring faces (-y -> +x)"

def test_3d_single_line_special_cases():
    # TODO add special cases
    # - zero-length line, inside
    # - zero-length line, outside
    # - zero-length line, on boundary
    # - touching corner
    # - touching edge
    # - touching face
    # - contained in edge
    # - contained in face
    pass

def test_3d_multiline():
    lines = np.array([
        [[2,2,2], [4,2,2]],
        [[2,2,2], [2,4,2]],
        [[2,2,2], [2,2,4]],
        [[1,5,4], [2,4,6]],
        [[2,4,1], [3,5,2]],
        [[-2,3,4], [-3,5,6]],
        [[12,3,4], [13,5,6]],
        [[2,-3,4], [3,-5,6]],
        [[2,12,4], [3,15,6]],
        [[2,3,-4], [3,5,-6]],
        [[2,3,14], [3,5,16]],
        [[-2,3,4], [13,3,4]],
        [[2,-3,4], [2,13,4]],
        [[2,3,-4], [2,3,14]],
        [[-2,1,1], [6,3,5]],
        [[2,-1,4], [6,3,0]]
    ])

    result = clip_line(lines, VMIN, VMAX)

    # all values copied from above tests
    assert np.allclose(result[0], np.array([[2,2,2], [4,2,2]]))
    assert np.allclose(result[1], np.array([[2,2,2], [2,4,2]]))
    assert np.allclose(result[2], np.array([[2,2,2], [2,2,4]]))
    assert np.allclose(result[3], np.array([[1,5,4], [2,4,6]]))
    assert np.allclose(result[4], np.array([[2,4,1], [3,5,2]]))
    assert _is_outside(result[5])
    assert _is_outside(result[6])
    assert _is_outside(result[7])
    assert _is_outside(result[8])
    assert _is_outside(result[9])
    assert _is_outside(result[10])
    assert np.allclose(result[11], np.array([[0,3,4], [4,3,4]]))
    assert np.allclose(result[12], np.array([[2,0,4], [2,6,4]]))
    assert np.allclose(result[13], np.array([[2,3,0], [2,3,8]]))
    assert np.allclose(result[14], np.array([[0.0,1.5,2.0], [4.0,2.5,4.0]]))
    assert np.allclose(result[15], np.array([[3.0,0.0,3.0], [4.0,1.0,2.0]]))
