# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "marimo>=0.23.3",
#     "matplotlib>=3.10.9",
#     "numpy>=2.2.6",
#     "pillow>=12.3.0",
# ]
# ///

import marimo

__generated_with = "0.24.2"
app = marimo.App(width="medium", auto_download=["html"])


@app.cell(hide_code=True)
def _():
    from io import BytesIO
    from urllib.request import Request, urlopen

    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    from PIL import Image

    return BytesIO, Image, Request, mo, np, plt, urlopen


@app.cell(hide_code=True)
def _(BytesIO, Image, Request, np, urlopen):
    def image_from_url(image_url):
        """Download image_url and return it as a grayscale NumPy array."""
        request = Request(image_url, headers={"User-Agent": "Mozilla/5.0"})
        with urlopen(request, timeout=30) as response:
            return np.asarray(Image.open(BytesIO(response.read())).convert("L"), dtype=float)

    return (image_from_url,)


@app.cell(hide_code=True)
def _(image_from_url):
    mare_image = image_from_url(
        "https://bigdatahealth.ucsb.edu/sites/default/files/styles/logo/"
        "public/logo/img_2289.jpeg"
    )
    mare_side = min(mare_image.shape)
    mare_top = (mare_image.shape[0] - mare_side) // 2
    mare_left = (mare_image.shape[1] - mare_side) // 2
    mare_image = mare_image[
        mare_top : mare_top + mare_side, mare_left : mare_left + mare_side
    ]
    storke_tower_image = image_from_url(
        "https://thumb.wikimedia.org/wikipedia/commons/thumb/4/48/"
        "Ucsb-storke-tower01.jpg/960px-Ucsb-storke-tower01.jpg"
    )
    kavli_image = image_from_url(
        "https://www.kitp.ucsb.edu/sites/default/files/kitp/kavlisb.jpeg"
    )
    return kavli_image, mare_image, storke_tower_image


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Homework 1

    ## Editing

    Marimo notebooks are just Python files, where each "cell" is a function. You can either edit here, in [Molab](https://molab.marimo.io/), or directly in Python.

    ### Option 1 - Molab [Recommended]

    This is the most recommended option. Take this notebook (.py file), upload it to Molab, and start working. Once you are finished making your edits, click "Export" and select "Python" to export it as a notebook source.

    ### Option 2 - Local Marimo

    To work locally, first create a python project and place this file `hw1.py` within it. Then, install marimo using your favorite package manager (`uv add marimo`). Finally, launch this notebook by typing in your terminal `uv run marimo edit` and a new tab should open in your browser.

    ### Option 3 - Regular Python [Not Recommended]

    It is theoretically possible to edit the `.py` file directly (although we don't recommend it). If you choose this route, make sure that your Python/Markdown renders properly in Marimo.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Problem 1: Image Convolution and Edge Detection

    You are given the following two images $I_1$ and $I_2$, with three convolution kernels $S_x$, $S_y$ and $L$. [.] indicates the origin of each image.

    $$
    I_1 = \begin{bmatrix}
    [1] & 1 & 1 \\
    3 & 3 & 3 \\
    6 & 6 & 6
    \end{bmatrix}
    \qquad
    I_2 = \begin{bmatrix}
    [10] & 10 & 10 \\
    10 & 100 & 100 \\
    10 & 100 & 100
    \end{bmatrix}
    $$

    $$
    S_x = \begin{bmatrix}-1 & 0 & 1 \\ -2 & [0] & 2 \\ -1 & 0 & 1\end{bmatrix}
    \qquad
    S_y = \begin{bmatrix}-1 & -2 & -1 \\ 0 & [0] & 0 \\ 1 & 2 & 1\end{bmatrix}
    \qquad
    L = \begin{bmatrix}0 & 1 & 0 \\ 1 & [-4] & 1 \\ 0 & 1 & 0\end{bmatrix}
    $$
    """)
    return


@app.cell
def _(np):
    # EDIT BELOW: transcribe the matrices displayed above into NumPy arrays.
    # Replace the zero defaults with the instructor-provided entries for I_1 and I_2.
    image_1 = np.array([[1, 1, 1], [3, 3, 3], [6, 6, 6]])
    image_2 = np.array([[10, 10, 10], [10, 100, 100], [10, 100, 100]])

    sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
    laplacian = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])
    return image_1, image_2, laplacian, sobel_x, sobel_y


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1(a) Sobel X and Sobel Y By Hand

    Compute the full linear convolutions $S_x * I_1$ and $S_y * I_1$ **by hand**, where $*$ denotes linear convolution. Mark the origin in each output matrix. Use the two responses to locate the edges, and explain your reasoning.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    Your response to 1(a)

    $$S_x * I_1 = \begin{bmatrix}
        -1 & -1 & 0 & 1 & 1 \\
        -5 & [-5] & 0 & 5 & 5 \\
        -13 & -13 & 0 & 13 & 13 \\
        -15 & -15 & 0 & 15 & 15 \\
        -6 & -6 & 0 & 6 & 6
        \end{bmatrix}

        \\
        \\

        S_y * I_1 = \begin{bmatrix}
        -1 & -3 & -4 & -3 & -1 \\
        -3 & [-9] & -12 & -9 & -3 \\
        -5 & -15 & -20 & -15 & -5\\
        3 & 9 & 12 & 9 & 3 \\
        6 & 18 & 24 & 18 & 6
        \end{bmatrix}
        $$

        In the X-convolution output, there is a vertical edge centered around the 3rd column, because the intensity changes dramatically from the 2nd column to the 3rd column to the 4th column; this change indicates a large derivative that is indicative of an edge.

        In the Y-convolution output, there is a horizontal edge between the third and fourth rows, because the intensity changes dramatically from the 3rd to the 4th row. This change indicates a large derivative that is indicative of an edge.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1(b) Laplacian By Hand

    Compute the full linear convolution $L * I_2$ **by hand**. Mark the origin in the output matrix. Use the response to locate any corner(s), and explain your reasoning.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    $$
    L * I_2 = \begin{bmatrix}
        0 & 10 & 10 & 10 & 0 \\
        10 & [-20] & 80 & 70 & 10 \\
        10 & 80 & -180 & -190 & 100 \\
        10 & 70 & -190 & -200 & 100 \\
        0 & 10 & 100 & 100 & 0
        \end{bmatrix}
    $$

    There is a corner around the origin, specifically in the middle of the 2x2 grid formed with the origin in the top left. There, multiple edges (indicated by zero-crossings) intersect to form a corner; there are zero-crossings in multiple directions there.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1(c) Verify in Python

    Complete the two code cells below. The function must perform full 2D linear convolution without calling a library convolution routine (you may not use any numpy functions).
    """)
    return


@app.cell
def _(np):
    def convolve2d_full(image, kernel):
        """Return the full 2D linear convolution of image and kernel."""
        output_shape = (
            image.shape[0] + kernel.shape[0] - 1,
            image.shape[1] + kernel.shape[1] - 1,
        )

        output = np.empty(output_shape, dtype=float)

        offsets = range(-(len(kernel)//2), len(kernel)//2+1)
        offset = offsets[-1]

        for y in range(output_shape[0]):
            for x in range(output_shape[1]):
                output[y,x] = 0

                for ky in offsets:
                    for kx in offsets:
                        img_y = y-ky-offset
                        img_x = x-kx-offset

                        if img_y < 0 or img_x < 0 or img_y >= image.shape[0] or img_x >= image.shape[1]:
                            pass
                        else: 
                            output[y, x] += kernel[ky+offset, kx+offset] * image[img_y, img_x]
        

        return output

    return (convolve2d_full,)


@app.cell(hide_code=True)
def _(convolve2d_full, image_1, image_2, laplacian, sobel_x, sobel_y):
    try:
        convolve2d_full(image_1, sobel_x)
        convolve2d_full(image_1, sobel_y)
        convolve2d_full(image_2, laplacian)
    except Exception as error:
        print(f"Problem 1(c) test is not ready: {error}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Test Your Convolution on a Real Image

    The next cell downloads a photograph, converts it to grayscale, and runs your convolution function with both Sobel kernels. It displays the input, the horizontal and vertical responses, and their combined edge magnitude. You can use this as a quick check after implementing `convolve2d_full`.
    """)
    return


@app.cell(hide_code=True)
def _(convolve2d_full, mare_image, np, sobel_x, sobel_y):
    try:
        mare_sobel_x = convolve2d_full(mare_image, sobel_x)
        mare_sobel_y = convolve2d_full(mare_image, sobel_y)
        mare_edges = np.hypot(mare_sobel_x, mare_sobel_y)
        mare_error = None
    except Exception as error:
        mare_sobel_x = mare_sobel_y = mare_edges = None
        mare_error = str(error)
    return mare_edges, mare_error, mare_sobel_x, mare_sobel_y


@app.cell(hide_code=True)
def _(mare_edges, mare_error, mare_image, mare_sobel_x, mare_sobel_y, mo):
    try:
        if mare_error:
            mo.output.replace(mo.md(f"Sobel test on mare is not ready: {mare_error}"))
        else:
            mo.output.replace(mo.vstack(
                [
                    mo.hstack(
                        [mo.image(mare_image, width=220), mo.image(mare_sobel_x, width=220)]
                    ),
                    mo.hstack(
                        [mo.image(mare_sobel_y, width=220), mo.image(mare_edges, width=220)]
                    ),
                ]
            ))
    except Exception as error:
        mo.output.replace(mo.md(f"Could not display the Sobel test: {error}"))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Problem 2: Gaussian and Laplacian Pyramids

    You are given a color image. Convert it to grayscale, then crop the 512×512 region beginning at the upper-left corner. Use this crop as the input image for both parts.

    You will build pyramid levels at 512×512, 256×256, 128×128, and 32×32. To reach 32×32 from 128×128, blur and downsample twice. Do not use a prebuilt pyramid implementation.

    A normalized 5×5 Gaussian kernel is provided in the starter code. In your report, state your boundary convention, show the requested pyramid images at their corresponding resolutions, and include your code. If you use any non-standard-library helper, include it with a brief explanation.
    """)
    return


@app.cell
def _(storke_tower_image):
    # EDIT BELOW: take the upper-left crop from storke_tower_image.
    input_image = storke_tower_image[:512, :512]
    return (input_image,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2(a) Gaussian Pyramid

    Create a Gaussian pyramid from the 512×512 grayscale crop using Gaussian convolution and the provided normalized 5×5 kernel. Apply blur and downsampling to produce four levels, in this order: 512×512, 256×256, 128×128, and 32×32. Document your boundary convention, and include the four images and your code in the report. Do not use a prebuilt pyramid implementation.
    """)
    return


@app.cell
def blur_and_downsample(convolve2d_full):
    def blur_and_downsample(image, kernel):
        """Blur image with kernel, then reduce each spatial dimension by two."""
        # EDIT BELOW: call convolve2d_full, crop to image.shape, then subsample.
        r = convolve2d_full(image, kernel)

        return r[kernel.shape[0]//2:(r.shape[0]-(kernel.shape[0]//2)):2, kernel.shape[1]//2:(r.shape[1]-(kernel.shape[1]//2)):2]

        # Document your boundary convention:
        # Zero-padding; when we get out of bounds, we just respond with a zero
    return (blur_and_downsample,)


@app.cell
def _(blur_and_downsample, np):
    def build_gaussian_pyramid(
        image, level_shapes=((512, 512), (256, 256), (128, 128), (32, 32))
    ):
        """Return Gaussian-pyramid levels with exactly the requested shapes."""
        kernel = np.array(
            [
                [1, 4, 6, 4, 1],
                [4, 16, 24, 16, 4],
                [6, 24, 36, 24, 6],
                [4, 16, 24, 16, 4],
                [1, 4, 6, 4, 1],
            ],
            dtype=float,
        ) / 256
        pyramid = [image]
        queue = [image]
        # EDIT BELOW: call blur_and_downsample until each requested level is reached.
        # Keep levels in the same order as level_shapes.
        for i in range(len(level_shapes)+1):
            o = blur_and_downsample(queue[i], kernel)

            queue.append(o)

            if o.shape in level_shapes:
                pyramid.append(o)

        return pyramid

    return (build_gaussian_pyramid,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2(b) Laplacian Pyramid

    Using your Gaussian pyramid, construct the corresponding Laplacian pyramid. Construct the Laplacian pyramid for each of the four resolutions given above. For example, to construct the Laplacian pyramid at resolution 512x512, you take the Gaussian smoothed image at resolution 512x512 and subtract it from the original 512x512 image.
    """)
    return


@app.cell
def build_laplacian_pyramid(np):
    def build_laplacian_pyramid(gaussian_pyramid):
        """Return Laplacian levels followed by the coarsest Gaussian residual."""
        pyramid = []
        rvr = gaussian_pyramid[::-1]

        # EDIT BELOW: construct the Laplacian level at each requested resolution,
        # then append the final coarsest Gaussian level.

        # start at the smallest and upscale by inserting 0s. then subtract the next layer from the upscaled (maybe reverse)

        for idx in range(len(rvr)-1):
            scale = rvr[idx+1].shape[0] // rvr[idx].shape[0]

            gp = rvr[idx]
            new_size = gp.shape[0]*scale
            o = np.zeros((new_size, new_size), dtype=float)        

            for i in range(0, new_size, scale):
                for j in range(0, new_size, scale):
                    o[i, j] = gp[i//scale, j//scale]

            for i in range(new_size):
                for j in range(new_size):
                    o[i, j] = rvr[idx+1][i, j] - o[i, j]
            pyramid.append(o)

        rtnr = pyramid[::-1]
        rtnr.append(rvr[0])
    
        return rtnr

    return (build_laplacian_pyramid,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Pyramid Visualizations

    After completing your functions, the hidden cells below display the Gaussian pyramid in the first row and the Laplacian pyramid in the second row.
    """)
    return


@app.cell(hide_code=True)
def _(build_gaussian_pyramid, build_laplacian_pyramid, input_image):
    try:
        gaussian_pyramid = build_gaussian_pyramid(input_image)
        laplacian_pyramid = build_laplacian_pyramid(gaussian_pyramid)
        pyramid_error = None
    except Exception as error:
        gaussian_pyramid = laplacian_pyramid = None
        pyramid_error = str(error)
    return gaussian_pyramid, laplacian_pyramid, pyramid_error


@app.cell(hide_code=True)
def _(gaussian_pyramid, laplacian_pyramid, mo, pyramid_error):
    try:
        if pyramid_error:
            mo.output.replace(mo.md(f"Pyramid visualization is not ready: {pyramid_error}"))
        else:
            mo.output.replace(mo.vstack(
                [
                    mo.hstack([mo.image(level, width=150) for level in gaussian_pyramid]),
                    mo.hstack([mo.image(level, width=150) for level in laplacian_pyramid]),
                ]
            ))
    except Exception as error:
        mo.output.replace(mo.md(f"Could not display the pyramids: {error}"))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Problem 3: Harris Corner Detector

    For this part, implement a Harris corner detector. At each image location, compute the corner response function (CRF)

    $$
    R(x, y) = \det(A) - k\bigl(\operatorname{trace}(A)\bigr)^2,
    $$

    where $A$ is the local structure matrix

    $$
    A =
    \begin{bmatrix}
    \langle I_x^2 \rangle & \langle I_x I_y \rangle \\
    \langle I_x I_y \rangle & \langle I_y^2 \rangle
    \end{bmatrix}.
    $$

    Use a simple constant 3×3 averaging window for the local averages. You may use any discrete derivative approximation, but write down the equations you use. You may ignore image-border effects by zero-padding or by limiting calculations to the image interior.

    Use the CRF to identify the top \(N\) candidate corners, where \(N\) is around 20–50. Threshold and sort the candidates before selecting the final locations. Your function should return `(row, column)` locations.

    Put your complete implementation, including any helper functions, in the cell below. You may use functions defined elsewhere in this notebook. Use the KAVLI image to demonstrate your detector: include your code in the report, overlay the detected corners on the image, list their coordinates, state your parameter values, and comment on the results.
    """)
    return


@app.cell
def harris_corners(convolve2d_full, np, sobel_x, sobel_y):
    def harris_corners(image, threshold, max_corners=40, k=0.04):
        """Return up to max_corners Harris corners as (row, column) locations."""
        # EDIT BELOW: write your complete detector and any helpers here.

        def crop_convolve(image, kernel):
            r = convolve2d_full(image, kernel)

            return r[kernel.shape[0]//2:(r.shape[0]-(kernel.shape[0]//2)), kernel.shape[1]//2:(r.shape[1]-(kernel.shape[1]//2))]


        # We will use Sobel X and Sobel Y

        s_x = crop_convolve(image, sobel_x)
        s_y = crop_convolve(image, sobel_y)

        def smooth(img):
            def averaged(i, j):
                avg = 0

                for io in [-1, 0, 1]:
                    for jo in [-1, 0, 1]:
                        if i+io < 0 or j+jo < 0 or i+io >= img.shape[0] or j+jo >= img.shape[1]:
                            continue

                        else:
                            avg += img[i+io, j+jo]


                return avg / 9

            for i in range(img.shape[0]):
                for j in range(img.shape[1]):
                    img[i, j] = averaged(i, j)

            return img


        Sx2 = smooth(s_x ** 2)
        Sy2 = smooth(s_y ** 2)
        Sxy = smooth(s_x*s_y)
    

        rtnr = []

        for i in range(s_x.shape[0]):
            for j in range(s_x.shape[1]):
                R = (Sx2[i,j]*Sy2[i,j]-Sxy[i,j]**2) - k*((Sx2[i,j]+Sy2[i,j])**2)
               
                if R >= threshold:
                    rtnr.append((R, i, j))

        rtnr.sort(reverse=True)

        return np.array(rtnr[:max_corners])[:,1:]  

    return (harris_corners,)


@app.cell
def _(harris_corners, kavli_image, plt):
    # EDIT BELOW: choose a threshold, then plot the detected corners.
    corners = harris_corners(kavli_image, threshold=100000)

    # It's interesting how the algorithm weighs what the 'strongest' corners are (different than what I would say):
    # Seems to pick up the brightest, closest, and most well defined corners
    # also picks up false corners given by the intersection of circular objects & poles
    # Not a perfect method but fairly good!

    plt.imshow(kavli_image, cmap="gray")
    plt.axis("off")
    plt.scatter(corners[:, 1], corners[:, 0], c="r", s=10)
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Submitting Your Work

    If you used Molab, click **Export** and choose **Python** to download your completed notebook as a `.py` file. Submit that `.py` notebook file to Gradescope.
    """)
    return


if __name__ == "__main__":
    app.run()
