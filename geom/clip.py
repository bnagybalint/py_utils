import numpy as np


def clip_line(
    lines: np.ndarray,
    clip_min: np.ndarray,
    clip_max: np.ndarray,
) -> np.ndarray:
    """Clip line(s) to clip volume.

    Parameters
    ----------
    lines : np.ndarray
        Array of shape (2,D) or (N,2,D) representing the lines to clip, where N is the number of lines, D is the dimensionality of the space used.
        NOTE: currently only 3D clipping is supported (D=3)
    clip_min : np.ndarray
        Array of shape (D,) describing the minimum planes along each axis, where D is the dimensionality of the space used
    clip_max : np.ndarray
        Array of shape (D,) describing the maximum planes along each axis, where D is the dimensionality of the space used

    Returns
    -------
    np.ndarray
        Clipped lines. Lines completely outside the clip volume are set to zero-length lines.
    """
    if not isinstance(lines, np.ndarray):
        raise TypeError("Argument 'lines' should be a numpy array")

    orig_shape = lines.shape
    if len(lines.shape) == 3:
        num_lines, points_per_line, dim = orig_shape
    elif len(lines.shape) == 2:
        points_per_line, dim = orig_shape
        num_lines = 1
        used_shape = (num_lines, points_per_line, dim)
        lines = lines.reshape(used_shape)
    else:
        raise ValueError("Array shape must be either (2,D) or (N,2,D).")

    if points_per_line != 2:
        raise ValueError("Array shape must be either (2,D) or (N,2,D).")
    if dim != 3:
        raise ValueError("Only clipping of 3D lines is supported at the moment")
    if clip_min.shape != (dim,):
        raise ValueError("Clipping boundary min should be of shape (3,)")
    if clip_max.shape != (dim,):
        raise ValueError("Clipping boundary max should be of shape (3,)")
    
    result = _clip_lines_unchecked(lines, clip_min, clip_max)
    return result.reshape(orig_shape)

def _clip_lines_unchecked(
    lines: np.ndarray,
    clip_min: np.ndarray,
    clip_max: np.ndarray,
) -> np.ndarray:
    return np.array([_clip_line_unchecked(l, clip_min, clip_max) for l in lines])

def _clip_line_unchecked(
    line: np.ndarray,
    clip_min: np.ndarray,
    clip_max: np.ndarray,
) -> np.ndarray:
    # based on: https://en.wikipedia.org/wiki/Liang%E2%80%93Barsky_algorithm

    a, b = line

    # point-to-point relationship values
    p1 = -(b[0] - a[0])
    p2 = -p1
    p3 = -(b[1] - a[1])
    p4 = -p3
    p5 = -(b[2] - a[2])
    p6 = -p5

    # point-to-min/max relationship values
    q1 = a[0] - clip_min[0]
    q2 = clip_max[0] - a[0]
    q3 = a[1] - clip_min[1]
    q4 = clip_max[1] - a[1]
    q5 = a[2] - clip_min[2]
    q6 = clip_max[2] - a[2]

    posarr = [1]
    negarr = [0]

    # check if line is parallel to clipping window
    if (
        p1 == 0 and q1 < 0 or
        p2 == 0 and q2 < 0 or
        p3 == 0 and q3 < 0 or
        p4 == 0 and q4 < 0 or
        p5 == 0 and q5 < 0 or
        p6 == 0 and q6 < 0
    ):
        return np.array([clip_min, clip_min])

    # check along edge x
    if p1 != 0:
        r1 = q1 / p1
        r2 = q2 / p2
        if p1 < 0:
            negarr.append(r1) # for negative p1, add it to negative array
            posarr.append(r2) # and add p2 to positive array
        else:
            negarr.append(r2)
            posarr.append(r1)
            
    # check along edge y
    if p3 != 0:
        r3 = q3 / p3
        r4 = q4 / p4
        if p3 < 0:
            negarr.append(r3)
            posarr.append(r4)
        else:
            negarr.append(r4)
            posarr.append(r3)

    # check along edge z
    if p5 != 0:
        r5 = q5 / p5
        r6 = q6 / p6
        if p5 < 0:
            negarr.append(r5)
            posarr.append(r6)
        else:
            negarr.append(r6)
            posarr.append(r5)

    # find tightest clipping parameters
    rn1 = max(negarr) # maximum of negative array
    rn2 = min(posarr) # minimum of positive array

    if rn1 > rn2:
        # line is outside the clipping window
        return np.array([clip_min, clip_min])

    xn1 = a[0] + p2 * rn1
    yn1 = a[1] + p4 * rn1
    zn1 = a[2] + p6 * rn1

    xn2 = a[0] + p2 * rn2
    yn2 = a[1] + p4 * rn2
    zn2 = a[2] + p6 * rn2
    
    return np.array([[xn1,yn1,zn1], [xn2,yn2,zn2]])