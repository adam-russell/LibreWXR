/* SPDX-License-Identifier: AGPL-3.0-or-later */
#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <stdint.h>
#include <string.h>

/* Match NumPy float32's evaluation order. Build without contraction/fast math. */
static uint8_t pixel(const uint8_t *data, Py_ssize_t sy, Py_ssize_t sx,
                     int32_t r0, int32_t r1, int32_t c0, int32_t c1,
                     float dr, float dc) {
    float a = data[r0 * sy + c0 * sx], b = data[r0 * sy + c1 * sx];
    float c = data[r1 * sy + c0 * sx], d = data[r1 * sy + c1 * sx];
    if (a == 0 || b == 0 || c == 0 || d == 0) return (uint8_t)a;
    float t0 = (a * (1.0f - dr)) * (1.0f - dc);
    float t1 = (b * (1.0f - dr)) * dc;
    float t2 = (c * dr) * (1.0f - dc);
    float t3 = (d * dr) * dc;
    float value = ((t0 + t1) + t2) + t3;
    value = value + 0.5f;
    if (value < 0) value = 0;
    if (value > 255) value = 255;
    return (uint8_t)value;
}

static int format_is(const Py_buffer *view, const char *format, Py_ssize_t size) {
    return view->format && strcmp(view->format, format) == 0 && view->itemsize == size;
}

static PyObject *sample(PyObject *self, PyObject *args) {
    PyObject *objects[10];
    Py_buffer b[10] = {{0}};
    int separable, acquired = 0, bad_plan = 0;
    if (!PyArg_ParseTuple(args, "OOOOOOOOOOp:sample", &objects[0], &objects[1],
                          &objects[2], &objects[3], &objects[4], &objects[5],
                          &objects[6], &objects[7], &objects[8], &objects[9],
                          &separable)) return NULL;
    for (int i = 0; i < 10; ++i) {
        int flags = PyBUF_STRIDES | PyBUF_FORMAT;
        if (i == 9) flags |= PyBUF_WRITABLE;
        if (PyObject_GetBuffer(objects[i], &b[i], flags) < 0) goto fail;
        ++acquired;
    }
    if (!format_is(&b[0], "B", 1) || b[0].ndim != 2 ||
        b[0].shape[0] <= 0 || b[0].shape[1] <= 0 || b[0].suboffsets ||
        !format_is(&b[9], "B", 1) || b[9].ndim != 2 ||
        !PyBuffer_IsContiguous(&b[9], 'C')) {
        PyErr_SetString(PyExc_ValueError, "Expected a uint8 frame and contiguous uint8 tile");
        goto fail;
    }
    Py_ssize_t h = b[9].shape[0], w = b[9].shape[1];
    if (h <= 0 || w <= 0 || h > PY_SSIZE_T_MAX / w) {
        PyErr_SetString(PyExc_ValueError, "Invalid output dimensions");
        goto fail;
    }
    for (int i = 1; i <= 8; ++i) {
        const char *format = i <= 4 ? "i" : i <= 6 ? "f" : "?";
        Py_ssize_t itemsize = i <= 6 ? 4 : 1;
        int row_axis = i == 1 || i == 2 || i == 5 || i == 7;
        if (!format_is(&b[i], format, itemsize) ||
            !PyBuffer_IsContiguous(&b[i], 'C') || b[i].suboffsets ||
            (i <= 6 && (uintptr_t)b[i].buf % 4 != 0) ||
            b[i].ndim != (separable ? 1 : 2) ||
            b[i].shape[0] != (separable && !row_axis ? w : h) ||
            (!separable && b[i].shape[1] != w)) {
            PyErr_SetString(PyExc_ValueError, "Sampling plan buffer format or shape mismatch");
            goto fail;
        }
    }
    const uint8_t *data = b[0].buf, *rv = b[7].buf, *cv = b[8].buf;
    const int32_t *r0 = b[1].buf, *r1 = b[2].buf, *c0 = b[3].buf, *c1 = b[4].buf;
    const float *dr = b[5].buf, *dc = b[6].buf;
    uint8_t *out = b[9].buf;
    Py_ssize_t height = b[0].shape[0], width = b[0].shape[1];
    Py_ssize_t sy = b[0].strides[0], sx = b[0].strides[1];
    /* Buffer exports keep every source/plan/output alive while the GIL is free. */
    Py_BEGIN_ALLOW_THREADS
    for (Py_ssize_t y = 0; y < h && !bad_plan; ++y) {
        for (Py_ssize_t x = 0; x < w; ++x) {
            Py_ssize_t i = y * w + x, r = separable ? y : i, c = separable ? x : i;
            if (!rv[r] || !cv[c]) { out[i] = 0; continue; }
            int32_t a = r0[r], b1 = r1[r], c2 = c0[c], d = c1[c];
            float fr = dr[r], fc = dc[c];
            /* Reject malformed plans before pointer arithmetic or float casts. */
            if (a < 0 || a >= height || b1 < 0 || b1 >= height ||
                c2 < 0 || c2 >= width || d < 0 || d >= width ||
                !(fr >= 0 && fr <= 1) || !(fc >= 0 && fc <= 1)) {
                bad_plan = 1;
                break;
            }
            out[i] = pixel(data, sy, sx, a, b1, c2, d, fr, fc);
        }
    }
    Py_END_ALLOW_THREADS
    if (bad_plan) {
        PyErr_SetString(PyExc_ValueError, "Sampling plan indices or weights are out of range");
        goto fail;
    }
    for (int i = acquired - 1; i >= 0; --i) PyBuffer_Release(&b[i]);
    Py_RETURN_NONE;
fail:
    for (int i = acquired - 1; i >= 0; --i) PyBuffer_Release(&b[i]);
    return NULL;
}

static PyMethodDef methods[] = {
    {"sample", sample, METH_VARARGS, "Sample a read-only interpolation plan into a uint8 tile."},
    {NULL, NULL, 0, NULL}
};
static struct PyModuleDef module = {
    PyModuleDef_HEAD_INIT, "_sampling_native", "Exact float32 radar sampling.", -1, methods
};
PyMODINIT_FUNC PyInit__sampling_native(void) { return PyModule_Create(&module); }
