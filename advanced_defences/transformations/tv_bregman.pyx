#cython: cdivision=True
#cython: boundscheck=False
#cython: nonecheck=False
#cython: wraparound=False

cimport numpy as cnp
import numpy as np
from libc.math cimport exp, fabs, sqrt
from libc.float cimport DBL_MAX


def _denoise_tv_bregman(image, mask, double weight, int max_iter, int gs_iter,
                        double eps, char isotropic):
    image = np.atleast_3d(image)

    cdef:
        Py_ssize_t rows = image.shape[0]
        Py_ssize_t cols = image.shape[1]
        Py_ssize_t dims = image.shape[2]
        Py_ssize_t r, c, k

        Py_ssize_t total = rows * cols * dims

    u = np.zeros(image.shape, dtype=np.double)
    u[:, :, :] = image

    cdef:
        double[:, :, ::1] cimage = np.ascontiguousarray(image)
        char[:, :, ::1] cmask = mask
        double[:, :, ::1] cu = u

        double[:, :, ::1] dx = np.zeros(image.shape, dtype=np.double)
        double[:, :, ::1] dy = np.zeros(image.shape, dtype=np.double)
        double[:, :, ::1] bx = np.zeros(image.shape, dtype=np.double)
        double[:, :, ::1] by = np.zeros(image.shape, dtype=np.double)
        double[:, :, ::1] z = np.zeros(image.shape, dtype=np.double)
        double[:, :, ::1] uprev = np.ascontiguousarray(image)

        double ux, uy, unew, bxx, byy, dxx, dyy, s
        int i = 0
        double lam = 2 * weight
        double rmse = DBL_MAX
        double neighbors = 0
        double inner = 0

    while i < max_iter and rmse > eps:

        for _ in range(gs_iter):
            for k in range(dims):
                for r in range(rows):
                    for c in range(cols):
                    # Gauss-Seidel method
                        inner = z[r, c, k]
                        neighbors = 0
                        if r > 0:
                            inner += cu[r - 1, c, k]
                            neighbors += 1
                        if r < rows - 1:
                            inner += cu[r + 1, c, k]
                            neighbors += 1
                        if c > 0:
                            inner += cu[r, c - 1, k]
                            neighbors += 1
                        if c < cols - 1:
                            inner += cu[r, c + 1, k]
                            neighbors += 1
                        if cmask[r, c, k] == 1:
                            unew = (lam * inner + weight * cimage[r, c, k]) / (weight + neighbors * lam)
                        else:
                            unew = inner / 4
                        cu[r, c, k] = unew

        rmse = 0
        for k in range(dims):
            for r in range(rows):
                for c in range(cols):
                    # forward derivatives
                    if c == cols - 1:
                        ux = 0
                    else:
                        ux = cu[r, c + 1, k] - cu[r, c, k]
                    if r == rows - 1:
                        uy = 0
                    else:
                        uy = cu[r + 1, c, k] - cu[r, c, k]

                    bxx = bx[r, c, k]
                    byy = by[r, c, k]

                    # d_subproblem after reference [4]
                    if isotropic:
                        s = sqrt((ux + bxx)**2 + (uy + byy)**2)
                        dxx = s * lam * (ux + bxx) / (s * lam + 1)
                        dyy = s * lam * (uy + byy) / (s * lam + 1)

                    else:
                        s = ux + bxx
                        if s > 1 / lam:
                            dxx = s - 1/lam
                        elif s < -1 / lam:
                            dxx = s + 1 / lam
                        else:
                            dxx = 0
                        s = uy + byy
                        if s > 1 / lam:
                            dyy = s - 1 / lam
                        elif s < -1 / lam:
                            dyy = s + 1 / lam
                        else:
                            dyy = 0

                    dx[r, c, k] = dxx
                    dy[r, c, k] = dyy

                    bx[r, c, k] += ux - dxx
                    by[r, c, k] += uy - dyy

                    z[r, c, k] = -dx[r, c, k] - dy[r, c, k] + bx[r, c, k] + by[r, c, k]
                    if r > 0:
                        z[r, c, k] += dy[r - 1, c, k] - by[r - 1, c, k]
                    if c > 0:
                        z[r, c, k] += dx[r, c - 1, k] - bx[r, c - 1, k]

                    # update rmse
                    rmse += (cu[r, c, k] - uprev[r, c, k])**2

        rmse = sqrt(rmse / total)
        uprev = np.copy(cu)
        i += 1

    return np.squeeze(np.asarray(u))
