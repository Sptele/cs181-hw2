# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "marimo>=0.23.3",
#     "matplotlib>=3.11.2",
#     "numpy>=2.5.3",
# ]
# ///

import marimo

__generated_with = "0.24.2"
app = marimo.App(width="medium", auto_download=["html"])


@app.cell(hide_code=True)
def _():
    import math
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np

    return math, mo, np, plt


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Homework 2: Corner and Blob Detection

    In this assignment you generate simple images of black squares on a white background and detect their corners and blobs.

    **Submission:** submit `hw2.zip` containing `hw1.py` and `hw2.py`. Keep both files in the same folder, because this homework imports your Homework 1 functions.

    **Rules:** use your Homework 1 functions `laplacian`, `convolve2d_full`, and `harris_corners`, plus the provided `harris_response(image)`. Do not use `np.convolve`, `scipy`, or `cv2`.
    """)
    return


@app.cell
def _():
    # Load hw1.py and take laplacian, convolve2d_full, and harris_corners.
    from hw1 import app as hw1_app

    _, hw1_definitions = hw1_app.run()
    laplacian = hw1_definitions["laplacian"]
    convolve2d_full = hw1_definitions["convolve2d_full"]
    harris_corners = hw1_definitions["harris_corners"]
    return convolve2d_full, harris_corners


@app.cell
def _(convolve2d_full, np):
    def harris_response(image, k=0.04):
        """Return the Harris corner response function (CRF) map of image."""
        # Provided: the CRF computation from Homework 1, Problem 3.
        image = np.asarray(image, dtype=float)
        horizontal_kernel = np.array([[0.5, 0.0, -0.5]])
        gradient_x = convolve2d_full(image, horizontal_kernel)[:, 1 : 1 + image.shape[1]]
        gradient_y = convolve2d_full(image, horizontal_kernel.T)[1 : 1 + image.shape[0], :]
        box = np.ones((3, 3), dtype=float) / 9

        def average_3_by_3(values):
            full = convolve2d_full(values, box)
            return full[1 : 1 + image.shape[0], 1 : 1 + image.shape[1]]

        a_xx = average_3_by_3(gradient_x * gradient_x)
        a_xy = average_3_by_3(gradient_x * gradient_y)
        a_yy = average_3_by_3(gradient_y * gradient_y)
        response = a_xx * a_yy - a_xy * a_xy - k * (a_xx + a_yy) ** 2

        return response

    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Problem 1: Eigenvalues of the Structure Matrix

    In Homework 1, the Harris detector used $R = \det(A) - k\,\bigl(\operatorname{trace}(A)\bigr)^2$ without computing the eigenvalues of the structure matrix $A$. Here you compute them directly.

    $$
    A_1 = \begin{bmatrix} 3 & 3 \\ 3 & 3 \end{bmatrix}
    \qquad
    A_2 = \begin{bmatrix} 8 & 0 \\ 0 & 7 \end{bmatrix}
    \qquad
    A_3 = \frac{1}{1000}\begin{bmatrix} 1 & 0 \\ 0 & 1 \end{bmatrix}
    \qquad
    A_4 = \begin{bmatrix} 1 & 3 \\ 3 & 1 \end{bmatrix}
    $$

    1. Enter the four matrices in the cell below.
    2. Complete `classify_feature(structure)`. Find the two eigenvalues by solving $\det(A - \lambda I) = 0$ yourself (no `np.linalg`), then return `"corner"`, `"edge"`, `"flat"`, or `"invalid"`.
    3. In the answer cell, report the eigenvalues and feature type of each matrix, and state any assumptions.
    """)
    return


@app.cell
def _():
    # EDIT BELOW: transcribe the matrices displayed above into nested lists.
    # Replace the zero defaults with the entries of A_1, A_2, A_3, and A_4.
    structure_1 = [[3, 3], [3, 3]]
    structure_2 = [[8, 0], [0, 7]]
    structure_3 = [[1/1000, 0], [0, 1/1000]]
    structure_4 = [[1, 3], [3, 1]]
    return structure_1, structure_2, structure_3, structure_4


@app.cell
def classify_feature(math):
    def classify_feature(structure, threshold=1.0):
        """Return 'corner', 'edge', 'flat' or 'invalid' for a 2x2 structure matrix."""

        # EDIT BELOW: compute both eigenvalues of [[a, b], [b, c]], then use
        # them to decide and return the feature type or invalid.

        # (a-L)(b-L)-b*c = 0
        # ab -al -bl +l^2 - bc  =0

        # l^2 - l(tr(A)) + det(A)

        tr = structure[0][0] + structure[1][1]
        det = structure[0][0] * structure[1][1] - structure[1][0] * structure[0][1]
    
        e1 = 0.5 * (tr + math.sqrt(tr**2 - 4*det))
        e2 = 0.5 * (tr - math.sqrt(tr**2 - 4*det))

        if e1 < 0 or e2 < 0:
            return "invalid"

        if e1 >= threshold and e2 >= threshold:
            return "corner"
        elif e1 >= threshold or e2 >= threshold:
            return "edge"
        elif e1 < threshold and e2 < threshold:
            return "flat"

        return "invalid"

    return (classify_feature,)


@app.cell
def _(classify_feature, structure_1, structure_2, structure_3, structure_4):
    try:
        print("A1:", classify_feature(structure_1))
        print("A2:", classify_feature(structure_2))
        print("A3:", classify_feature(structure_3))
        print("A4:", classify_feature(structure_4))
    except Exception as error:
        print(f"Problem 1 test is not ready: {error}")
    return


@app.cell
def _(mo):
    mo.md(r"""
    A1:
    - eigenvalues: 6, 0
    - Edge

    A2:
    - eigenvalues: 8, 7
    - Corner

    A3:
    - eigenvalues: 0.001 (multiplicity 2)
    - Flat

    A4:
    - eigenvalues: 4, -2
    - Invalid
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Problem 2: Single-Square Image (Image 1)

    ## 2(a) Image generation

    Generate two grayscale images: a white background (255) with black squares (0). Coordinates are \((x, y) = (\text{column}, \text{row})\).

    - **Image 1:** 100×100, one 32×32 square with top-left corner \((30, 40)\).
    - **Image 2:** 512×512, four squares:

    | Square | Top-left \((x, y)\) | Side |
    |---|---|---|
    | 1 | \((60, 60)\) | 40 |
    | 2 | \((180, 300)\) | 80 |
    | 3 | \((300, 100)\) | 120 |
    | 4 | \((340, 340)\) | 140 |

    Complete `make_square_image` below.
    """)
    return


@app.cell
def _():
    def make_square_image(size, squares):
        """Return a size-by-size white image with black squares.

        squares is a list of (x, y, side) tuples, where (x, y) = (column, row)
        is the top-left pixel of a square and side is its side length.
        """
        # EDIT BELOW: set the pixels of image.
        image = ...

        return image

    # EDIT BELOW: list the squares of each image as (x, y, side) tuples.
    image_1 = make_square_image(100, [])
    image_2 = make_square_image(512, [])
    return image_1, image_2


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Visualize the generated images:
    """)
    return


@app.cell(hide_code=True)
def _(image_1, image_2, mo, plt):
    try:
        _figure, _axes = plt.subplots(1, 2, figsize=(8, 4))
        for _axis, _image, _title in (
            (_axes[0], image_1, "Image 1: single square"),
            (_axes[1], image_2, "Image 2: four squares"),
        ):
            _axis.imshow(_image, cmap="gray", vmin=0, vmax=255)
            _axis.set_title(_title)
            _axis.set_xticks([])
            _axis.set_yticks([])
        plt.tight_layout()
        _display = plt.gcf()
    except Exception as error:
        _display = mo.md(f"2(a) display is not ready: {error}")
    _display
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2(b) Gaussian smoothing

    Smooth Image 1 with \(\sigma \in \{1, 2, 4, 8\}\). Complete:

    - `gaussian_kernel_1d(sigma)`: a 1D Gaussian sampled at offsets \(-r, \dots, r\), where \(r = \lceil 3\sigma \rceil\), normalized to sum to 1.
    - `convolve_same(image, kernel)`: use `convolve2d_full`, then crop so the output has the same size as `image`, with the kernel centred on each pixel. The border must not create fake edges: smoothing a constant image should return the same image.
    - `gaussian_smooth(image, sigma)`: 2D Gaussian smoothing. Tip: apply the 1D kernel along the rows, then along the columns. This is much faster than a full 2D kernel.
    """)
    return


@app.cell
def _():
    def gaussian_kernel_1d(sigma):
        """Return the normalized 1D Gaussian kernel of radius ceil(3 * sigma)."""
        # EDIT BELOW: compute the radius, sample g at the integer offsets, normalize.
        kernel = ...

        return kernel

    def convolve_same(image, kernel):
        """Return the same-size convolution of image with an odd-sized 2D kernel."""
        # EDIT BELOW: use convolve2d_full, then crop to image.shape.
        same = ...

        return same

    def gaussian_smooth(image, sigma):
        """Return image smoothed by a 2D Gaussian of standard deviation sigma."""
        # EDIT BELOW: smooth with gaussian_kernel_1d and convolve_same.
        smoothed = ...

        return smoothed

    return (gaussian_smooth,)


@app.cell(hide_code=True)
def _(gaussian_smooth, image_1, mo, plt):
    sigmas = (1, 2, 4, 8)
    try:
        smoothed_by_sigma = {sigma: gaussian_smooth(image_1, sigma) for sigma in sigmas}
        smoothing_error = None
        _figure, _axes = plt.subplots(1, 4, figsize=(13, 3.6))
        for _axis, _sigma in zip(_axes, sigmas):
            _axis.imshow(smoothed_by_sigma[_sigma], cmap="gray", vmin=0, vmax=255)
            _axis.set_title(f"Smoothed, σ = {_sigma}")
            _axis.axis("off")
        plt.tight_layout()
        _display = plt.gcf()
    except Exception as error:
        smoothed_by_sigma = None
        smoothing_error = str(error)
        _display = mo.md(f"Gaussian smoothing is not ready: {error}")
    _display
    return sigmas, smoothed_by_sigma, smoothing_error


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2(c) Harris corners

    Complete `corners_at_scale`: run `harris_corners` on each smoothed image. Choose `threshold` and `max_corners` so that **exactly the 4 corners** of the square are found at every \(\sigma\).

    *Hint: first check what `harris_corners` detects on a black square on a white background.*
    """)
    return


@app.cell
def _(harris_corners):
    def corners_at_scale(smoothed):
        """Return the Harris corners of a smoothed image as (row, column)."""
        # EDIT BELOW: call harris_corners with your choice of parameters.
        corners = harris_corners(smoothed, threshold=..., max_corners=...)

        return corners

    return (corners_at_scale,)


@app.cell(hide_code=True)
def _(corners_at_scale, mo, plt, sigmas, smoothed_by_sigma, smoothing_error):
    try:
        if smoothing_error:
            raise RuntimeError(smoothing_error)
        corners_by_sigma = {
            sigma: corners_at_scale(smoothed_by_sigma[sigma]) for sigma in sigmas
        }
        corners_error = None
        _figure, _axes = plt.subplots(1, 4, figsize=(13, 3.6))
        for _axis, _sigma in zip(_axes, sigmas):
            _axis.imshow(smoothed_by_sigma[_sigma], cmap="gray", vmin=0, vmax=255)
            _corners = corners_by_sigma[_sigma].reshape(-1, 2)
            _axis.scatter(_corners[:, 1], _corners[:, 0], c="r", s=18)
            _axis.set_title(f"Harris, σ = {_sigma} ({len(_corners)} found)")
            _axis.axis("off")
        plt.tight_layout()
        _display = plt.gcf()
    except Exception as error:
        corners_by_sigma = None
        corners_error = str(error)
        _display = mo.md(f"Problem 2(c) is not ready: {error}")
    _display
    return corners_by_sigma, corners_error


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2(d) Laplacian of Gaussian (LoG)

    - `log_response(image, sigma)`: smooth the image, then apply `laplacian` with `convolve_same`.
    - `local_extrema(response, fraction)`: return the `(row, column)` of every pixel (skip the 1-pixel border) where
        - \(|\text{response}|\) is at least that of all 8 neighbours (ties count, tolerance \(10^{-9} \max|\text{response}|\)), and
        - \(|\text{response}| \ge\) `fraction` × \(\max|\text{response}|\).

    Choose a default value for `fraction`.
    """)
    return


@app.cell
def _(np):
    def log_response(image, sigma):
        """Return the Laplacian of Gaussian response of image at scale sigma."""
        # EDIT BELOW
        response = ...

        return response

    # EDIT BELOW: choose a default fraction between 0 and 1.
    def local_extrema(response, fraction=...):
        """Return (row, column) locations of the strong local extrema of response."""
        magnitude = np.abs(np.asarray(response, dtype=float))
        extrema = []

        # EDIT BELOW: append (row, column) for every pixel meeting the conditions above.

        return np.asarray(extrema, dtype=int).reshape(-1, 2)

    return local_extrema, log_response


@app.cell(hide_code=True)
def _(
    image_1,
    local_extrema,
    log_response,
    mo,
    np,
    plt,
    sigmas,
    smoothing_error,
):
    try:
        if smoothing_error:
            raise RuntimeError(smoothing_error)
        log_by_sigma = {sigma: log_response(image_1, sigma) for sigma in sigmas}
        _figure, _axes = plt.subplots(1, 4, figsize=(13, 3.6))
        for _axis, _sigma in zip(_axes, sigmas):
            _response = log_by_sigma[_sigma]
            _extrema = local_extrema(_response)
            _limit = float(np.abs(_response).max()) or 1.0
            _axis.imshow(_response, cmap="RdBu", vmin=-_limit, vmax=_limit)
            _axis.scatter(_extrema[:, 1], _extrema[:, 0], c="k", marker="x", s=30)
            _axis.set_title(f"LoG, σ = {_sigma}")
            _axis.axis("off")
        plt.tight_layout()
        _display = plt.gcf()
    except Exception as error:
        log_by_sigma = None
        _display = mo.md(f"Problem 2(d) is not ready: {error}")
    _display
    return (log_by_sigma,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2(e) Difference of Gaussian (DoG)

    - `dog_response(image, sigma, k=1.4)`: \(G_{k\sigma} * I - G_\sigma * I\).
    - `dog_log_scale(dog, log)`: the number \(c\) that best fits DoG ≈ \(c\) × LoG, i.e. minimizes \(\sum (D - c\,L)^2\).

    **In the answer cell:**

    1. Compare the DoG and LoG extrema.
    2. The table shows \(c \approx (k-1)\sigma^2\). Explain why, using \(\partial G_\sigma / \partial \sigma = \sigma \nabla^2 G_\sigma\).
    """)
    return


@app.cell
def _(np):
    def dog_response(image, sigma, k=1.4):
        """Return the Difference of Gaussian response of image at scale sigma."""
        # EDIT BELOW
        response = ...

        return response

    def dog_log_scale(dog, log):
        """Return the least-squares factor c such that c * log best fits dog."""
        dog = np.asarray(dog, dtype=float)
        log = np.asarray(log, dtype=float)

        # EDIT BELOW: the least-squares factor c.
        scale = ...

        return scale

    return dog_log_scale, dog_response


@app.cell(hide_code=True)
def _(
    dog_log_scale,
    dog_response,
    image_1,
    local_extrema,
    log_by_sigma,
    mo,
    np,
    plt,
    sigmas,
    smoothing_error,
):
    try:
        if smoothing_error:
            raise RuntimeError(smoothing_error)
        _dog_by_sigma = {sigma: dog_response(image_1, sigma) for sigma in sigmas}
        _figure, _axes = plt.subplots(1, 4, figsize=(13, 3.6))
        for _axis, _sigma in zip(_axes, sigmas):
            _response = _dog_by_sigma[_sigma]
            _extrema = local_extrema(_response)
            _limit = float(np.abs(_response).max()) or 1.0
            _axis.imshow(_response, cmap="RdBu", vmin=-_limit, vmax=_limit)
            _axis.scatter(_extrema[:, 1], _extrema[:, 0], c="k", marker="x", s=30)
            _axis.set_title(f"DoG, σ = {_sigma}")
            _axis.axis("off")
        plt.tight_layout()
        _lines = [
            "| σ | Fitted c | (k − 1)σ² | Ratio |",
            "|---|---|---|---|",
        ]
        if log_by_sigma is not None:
            for _sigma in sigmas:
                _scale = dog_log_scale(_dog_by_sigma[_sigma], log_by_sigma[_sigma])
                _predicted = 0.4 * _sigma**2
                _lines.append(
                    f"| {_sigma} | {_scale:.3f} | {_predicted:.2f} | {_scale / _predicted:.3f} |"
                )
        _display = mo.vstack([plt.gcf(), mo.md("\n".join(_lines))])
    except Exception as error:
        _display = mo.md(f"Problem 2(e) is not ready: {error}")
    _display
    return


@app.cell
def _(mo):
    mo.md(r"""
    **Your answer to 2(e).**

    *Double-click this cell to write your answer. You may replace everything here.*
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2(f) Corner drift

    Set `true_corners_image_1` to the 4 corner pixels of the square in Image 1, as `(row, column)`. The table shows how far the strongest Harris corner from 2(c) is from the nearest true corner.

    **In the answer cell:**

    1. Give your `threshold`, `max_corners`, and `fraction`, with a one-sentence reason for each.
    2. Explain how you handled the image border in 2(b) and 2(c).
    3. How does the distance change with \(\sigma\)? In which direction does the corner move, and why?
    """)
    return


@app.cell
def _(np):
    # EDIT BELOW: the four corner pixels of the square in Image 1, as (row, column).
    true_corners_image_1 = ...

    def nearest_corner_distance(points, true_corners):
        """Return the distance from each point to its nearest true corner."""
        # Provided.
        points = np.asarray(points, dtype=float).reshape(-1, 2)
        true_corners = np.asarray(true_corners, dtype=float).reshape(-1, 2)
        differences = points[:, None, :] - true_corners[None, :, :]
        distances = np.hypot(differences[..., 0], differences[..., 1]).min(axis=1)

        return distances

    return nearest_corner_distance, true_corners_image_1


@app.cell(hide_code=True)
def _(
    corners_by_sigma,
    corners_error,
    mo,
    nearest_corner_distance,
    plt,
    sigmas,
    true_corners_image_1,
):
    try:
        if corners_error:
            raise RuntimeError(corners_error)
        _lines = [
            "| σ | Strongest corner (row, column) | Distance to nearest true corner (pixels) |",
            "|---|---|---|",
        ]
        _distances = []
        for _sigma in sigmas:
            _corners = corners_by_sigma[_sigma].reshape(-1, 2)
            if len(_corners) == 0:
                _lines.append(f"| {_sigma} | none detected | |")
                _distances.append(float("nan"))
                continue
            _distance = float(nearest_corner_distance(_corners[:1], true_corners_image_1)[0])
            _distances.append(_distance)
            _lines.append(
                f"| {_sigma} | ({_corners[0, 0]}, {_corners[0, 1]}) | {_distance:.2f} |"
            )
        _figure, _axis = plt.subplots(figsize=(4.5, 3))
        _axis.plot(sigmas, _distances, "o-")
        _axis.set_xlabel("σ")
        _axis.set_ylabel("Corner drift (pixels)")
        _axis.set_xticks(sigmas)
        plt.tight_layout()
        _display = mo.hstack([mo.md("\n".join(_lines)), plt.gcf()], align="center")
    except Exception as error:
        _display = mo.md(f"Problem 2(f) is not ready: {error}")
    _display
    return


@app.cell
def _(mo):
    mo.md(r"""
    **Your answer to 2(f).**

    *Double-click this cell to write your answer. You may replace everything here.*
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Problem 3: Multi-Scale Blob and Corner Detection (Image 2)

    Reuse your Problem 2 functions on Image 2.

    ## 3(a) Gaussian pyramid

    For each \(\sigma \in \{1, 2, 4, 8\}\), smooth Image 2, then downsample the result by a factor of \(\sigma\) (i.e. 1, 2, 4, 8).

    - `downsample(image, factor)`: keep every `factor`-th row and column, starting at `[0, 0]`.
    - `build_scale_pyramid(image, sigmas)`: return `(smoothed, levels)`, the full-size smoothed images and their downsampled versions.
    """)
    return


@app.cell
def _():
    def downsample(image, factor):
        """Keep every factor-th row and column of image, starting at [0, 0]."""
        # EDIT BELOW
        sampled = ...

        return sampled

    def build_scale_pyramid(image, sigmas):
        """Return (smoothed, levels): full-resolution and downsampled images per sigma."""
        # EDIT BELOW
        smoothed = ...
        levels = ...

        return smoothed, levels

    return (build_scale_pyramid,)


@app.cell(hide_code=True)
def _(build_scale_pyramid, image_2, mo, plt, sigmas):
    try:
        image_2_smoothed, image_2_levels = build_scale_pyramid(image_2, sigmas)
        pyramid_error = None
        _figure, _axes = plt.subplots(
            2, 4, figsize=(13, 6.2), gridspec_kw={"width_ratios": [1, 1, 1, 1]}
        )
        for _column, _sigma in enumerate(sigmas):
            _axes[0, _column].imshow(image_2_smoothed[_column], cmap="gray", vmin=0, vmax=255)
            _axes[0, _column].set_title(f"Smoothed, σ = {_sigma} (512×512)")
            _level = image_2_levels[_column]
            _axes[1, _column].imshow(
                _level, cmap="gray", vmin=0, vmax=255, interpolation="nearest",
                extent=(0, _level.shape[1] / 512, _level.shape[0] / 512, 0),
            )
            _axes[1, _column].set_xlim(0, 1)
            _axes[1, _column].set_ylim(1, 0)
            _axes[1, _column].set_title(f"Level {_column}: ÷{_sigma} ({_level.shape[0]}×{_level.shape[1]})")
        for _axis in _axes.ravel():
            _axis.axis("off")
        plt.tight_layout()
        _display = plt.gcf()
    except Exception as error:
        image_2_smoothed = image_2_levels = None
        pyramid_error = str(error)
        _display = mo.md(f"Problem 3(a) is not ready: {error}")
    _display
    return image_2_levels, image_2_smoothed, pyramid_error


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **In the answer cell:** briefly describe how the levels differ (e.g., how edges and corners blur or disappear).
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    **Your answer to 3(a).**

    *Double-click this cell to write your answer. You may replace everything here.*
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3(b) Harris corners at multiple scales

    Compare Harris on (i) the full-size smoothed images and (ii) the pyramid levels.

    - `to_original(points, factor)`: map `(row, column)` points in a level back to the original image.
    - `image_2_corners(image)`: call `harris_corners` with one `threshold` and `max_corners` that works for every \(\sigma\) and level. Return only corners of the squares (Image 2 has 16).
    - `corner_strengths(image, points)`: the CRF value at each point, using `harris_response`.
    """)
    return


@app.cell
def _(harris_corners):
    def to_original(points, factor):
        """Map (row, column) points in a level downsampled by factor to the original image."""
        # EDIT BELOW
        mapped = ...

        return mapped

    def image_2_corners(image):
        """Return the Harris corners of a smoothed Image 2 or pyramid level."""
        # EDIT BELOW: call harris_corners with your choice of parameters.
        corners = harris_corners(image, threshold=..., max_corners=...)

        return corners

    def corner_strengths(image, points):
        """Return the CRF of image at each (row, column) point."""
        # EDIT BELOW
        strengths = ...

        return strengths

    return corner_strengths, image_2_corners, to_original


@app.cell(hide_code=True)
def _(
    corner_strengths,
    image_2_corners,
    image_2_levels,
    image_2_smoothed,
    mo,
    nearest_corner_distance,
    np,
    plt,
    pyramid_error,
    sigmas,
    to_original,
):
    try:
        if pyramid_error:
            raise RuntimeError(pyramid_error)
        _true_corners = np.array(
            [
                (_y + _dy, _x + _dx)
                for _x, _y, _side in ((60, 60, 40), (180, 300, 80), (300, 100, 120), (340, 340, 140))
                for _dy in (0, _side - 1)
                for _dx in (0, _side - 1)
            ]
        )
        image_2_corners_full = {}
        image_2_corners_pyramid = {}
        _lines = [
            "| σ | Corners (i) | Corners (ii) | Median R (i) | Median R (ii) "
            "| Median distance to true corner (i) | (ii) |",
            "|---|---|---|---|---|---|---|",
        ]
        _figure, _axes = plt.subplots(1, 4, figsize=(14, 3.9))
        for _axis, _index, _sigma in zip(_axes, range(len(sigmas)), sigmas):
            _full = image_2_corners(image_2_smoothed[_index]).reshape(-1, 2)
            _level_corners = image_2_corners(image_2_levels[_index]).reshape(-1, 2)
            _pyramid = to_original(_level_corners, _sigma)
            image_2_corners_full[_sigma] = _full
            image_2_corners_pyramid[_sigma] = _pyramid
            _strength_full = corner_strengths(image_2_smoothed[_index], _full)
            _strength_pyramid = corner_strengths(image_2_levels[_index], _level_corners)
            _distance_full = np.median(nearest_corner_distance(_full, _true_corners))
            _distance_pyramid = np.median(nearest_corner_distance(_pyramid, _true_corners))
            _lines.append(
                f"| {_sigma} | {len(_full)} | {len(_pyramid)} | {np.median(_strength_full):.3g} "
                f"| {np.median(_strength_pyramid):.3g} | {_distance_full:.1f} | {_distance_pyramid:.1f} |"
            )
            _axis.imshow(image_2_smoothed[_index], cmap="gray", vmin=0, vmax=255)
            _axis.scatter(_full[:, 1], _full[:, 0], s=22, facecolors="none", edgecolors="r", label="(i) scale space")
            _axis.scatter(_pyramid[:, 1], _pyramid[:, 0], s=22, c="c", marker="x", label="(ii) pyramid")
            _axis.set_title(f"σ = {_sigma}")
            _axis.axis("off")
        _axes[0].legend(loc="lower left", fontsize=8)
        plt.tight_layout()
        _display = mo.vstack([plt.gcf(), mo.md("\n".join(_lines))])
    except Exception as error:
        image_2_corners_full = image_2_corners_pyramid = None
        _display = mo.md(f"Problem 3(b) is not ready: {error}")
    _display
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **In the answer cell:**

    1. How do corner locations, counts, and strengths differ between (i) and (ii)?
    2. Why are the pyramid CRF values larger?
    3. Why do the two approaches disagree at \(\sigma = 8\)?
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    **Your answer to 3(b).**

    *Double-click this cell to write your answer. You may replace everything here.*
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3(c) LoG and DoG on the Gaussian pyramid

    Apply LoG and DoG to each pyramid level with \(\sigma \in \{1, 2, 4, 8\}\) (in level pixels) to find blob centres. First decide: is the centre of a dark square a **maximum** or a **minimum** of the response?

    - `effective_sigma(level_sigma, factor, sigma)`: the total scale in original pixels, \(\sigma_\text{eff} = \sqrt{\sigma_\ell^2 + (f\sigma)^2}\).
    - `normalized_log(level, level_sigma, factor, sigma)`: \(s^2 \times\) `log_response(level, sigma)` with \(s = \sigma_\text{eff} / f\), so that different scales can be compared.
    - `dark_blob_candidates(response)`: pixels (skip the border) that are local extrema of the sign you chose (compared with all 8 neighbours, ties allowed), with magnitude at least \(10^{-6}\) × the largest magnitude of that sign.
    - `pyramid_blobs(levels, factors, sigmas, method="log", k=1.4)`: for every level and \(\sigma\), return rows `(row, column, sigma_eff, strength)` in original coordinates.
        - Response: `normalized_log` for LoG, `dog_response / (k - 1)` for DoG.
        - Strength: the magnitude of the response.
        - The level with factor \(f\) has \(\sigma_\ell = f\).
    - `suppress_overlaps(blobs, fraction)`: go from strongest to weakest blob, and keep a blob only if its centre is farther than \(\sqrt{2}\max(\sigma_\text{eff}, \sigma_\text{eff}')\) from every blob already kept. Return the kept blobs with strength at least `fraction` × the strongest.
    """)
    return


@app.cell
def _(np):
    def effective_sigma(level_sigma, factor, sigma):
        """Return the scale, in original pixels, of sigma applied to a pyramid level."""
        # EDIT BELOW
        sigma_eff = ...

        return sigma_eff

    def normalized_log(level, level_sigma, factor, sigma):
        """Return the scale-normalized LoG of a pyramid level."""
        # EDIT BELOW
        response = ...

        return response

    def dark_blob_candidates(response):
        """Return (row, column) of the local extrema of response at dark blob centres."""
        # EDIT BELOW
        candidates = ...

        return candidates

    def pyramid_blobs(levels, factors, sigmas, method="log", k=1.4):
        """Return (row, column, sigma_eff, strength) for every blob candidate."""
        # EDIT BELOW
        blobs = []
        # Append (row, column, sigma_eff, strength) for every candidate.

        return np.asarray(blobs, dtype=float).reshape(-1, 4)

    # EDIT BELOW: choose a default fraction between 0 and 1.
    def suppress_overlaps(blobs, fraction=...):
        """Keep the strongest non-overlapping blobs above fraction of the strongest."""
        blobs = np.asarray(blobs, dtype=float).reshape(-1, 4)

        # EDIT BELOW
        kept = ...

        return kept

    return effective_sigma, pyramid_blobs, suppress_overlaps


@app.cell(hide_code=True)
def _(effective_sigma, mo, sigmas):
    try:
        _lines = [
            "| Level (f) | σ = 1 | σ = 2 | σ = 4 | σ = 8 |",
            "|---|---|---|---|---|",
        ]
        for _factor in sigmas:
            _row = [f"{effective_sigma(_factor, _factor, _sigma):.1f}" for _sigma in sigmas]
            _lines.append(f"| {_factor} | " + " | ".join(_row) + " |")
        _display = mo.md("**Effective scale σ_eff (original pixels)**\n\n" + "\n".join(_lines))
    except Exception as error:
        _display = mo.md(f"Effective-scale table is not ready: {error}")
    _display
    return


@app.cell(hide_code=True)
def _(
    image_2_levels,
    mo,
    np,
    pyramid_blobs,
    pyramid_error,
    sigmas,
    suppress_overlaps,
):
    _centres = {
        "square 1 (side 40)": (79.5, 79.5),
        "square 2 (side 80)": (339.5, 219.5),
        "square 3 (side 120)": (159.5, 359.5),
        "square 4 (side 140)": (409.5, 409.5),
    }

    def _nearest_square(row, column):
        return min(_centres, key=lambda name: np.hypot(row - _centres[name][0], column - _centres[name][1]))

    try:
        if pyramid_error:
            raise RuntimeError(pyramid_error)
        image_2_blobs = {}
        _tables = []
        for _method in ("log", "dog"):
            _candidates = pyramid_blobs(image_2_levels, sigmas, sigmas, method=_method)
            image_2_blobs[_method] = suppress_overlaps(_candidates)
            _lines = [
                f"**{_method.upper()}: {len(_candidates)} candidates, {len(image_2_blobs[_method])} kept**",
                "",
                "| (row, column) | σ_eff | Strength | Nearest square |",
                "|---|---|---|---|",
            ]
            for _row, _column, _sigma_eff, _strength in image_2_blobs[_method]:
                _lines.append(
                    f"| ({_row:.0f}, {_column:.0f}) | {_sigma_eff:.1f} | {_strength:.1f} "
                    f"| {_nearest_square(_row, _column)} |"
                )
            _tables.append(mo.md("\n".join(_lines)))
        _display = mo.hstack(_tables, align="start")
    except Exception as error:
        image_2_blobs = None
        _display = mo.md(f"Problem 3(c) is not ready: {error}")
    _display
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **In the answer cell:**

    1. Which \(\sigma_\text{eff}\) is selected for each square, and how does it relate to the side length?
    2. Do LoG and DoG select the same blobs?
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    **Your answer to 3(c).**

    *Double-click this cell to write your answer. You may replace everything here.*
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3(d) Mapping and visualization

    In the cell below, draw all detections on Image 2 in original coordinates:

    - **Left:** Harris corners from (i) and (ii), with a different colour or marker for each \(\sigma\).
    - **Right:** LoG and DoG blobs as circles of radius \(r = c\,\sigma_\text{eff}\), where \(c\) is a constant you choose for visibility.
    """)
    return


@app.cell
def _(image_2, plt):
    # EDIT BELOW: overlay every detection on Image 2.
    _figure, (_corner_axis, _blob_axis) = plt.subplots(1, 2, figsize=(13, 6.5))
    for _axis in (_corner_axis, _blob_axis):
        _axis.imshow(image_2, cmap="gray", vmin=0, vmax=255)
        _axis.set_xticks([])
        _axis.set_yticks([])

    # Left axis: Harris corners from image_2_corners_full and image_2_corners_pyramid.

    # Right axis: LoG and DoG blobs from image_2_blobs, drawn as circles.

    plt.tight_layout()
    plt.gcf()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **In the answer cell:** what does the overlay show about detecting features at a single scale versus across scales?
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    **Your answer to 3(d).**

    *Double-click this cell to write your answer. You may replace everything here.*
    """)
    return


if __name__ == "__main__":
    app.run()
