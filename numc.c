#include "numc.h"
#include <structmember.h>

PyTypeObject Matrix61cType;

/* Helper functions for initalization of matrices and vectors */

//Make a copy of mat1 and store it into mat 2
void copy(matrix *mat1, matrix *mat2) {
    if (mat1->rows != mat2->rows || mat1->cols != mat2->cols) {
        PyErr_SetString(PyExc_RuntimeError, "Sizes of both matrices are not the same");
        return;
    }

    for (int i = 0; i < mat1->rows; i++) {
        for (int j = 0; j < mat1->cols; j++) {
            mat2->data[i][j] = mat1->data[i][j];
        }
    }
}

/*
 * Return a tuple given rows and cols
 */
PyObject *get_shape(int rows, int cols) {
  if (rows == 1 || cols == 1) {
    return PyTuple_Pack(1, PyLong_FromLong(rows * cols));
  } else {
    return PyTuple_Pack(2, PyLong_FromLong(rows), PyLong_FromLong(cols));
  }
}
/*
 * Matrix(rows, cols, low, high). Fill a matrix random double values
 */
int init_rand(PyObject *self, int rows, int cols, unsigned int seed, double low,
              double high) {
    matrix *new_mat;
    int alloc_failed = allocate_matrix(&new_mat, rows, cols);
    if (alloc_failed) return alloc_failed;
    rand_matrix(new_mat, seed, low, high);
    ((Matrix61c *)self)->mat = new_mat;
    ((Matrix61c *)self)->shape = get_shape(new_mat->rows, new_mat->cols);
    return 0;
}

/*
 * Matrix(rows, cols, val). Fill a matrix of dimension rows * cols with val
 */
int init_fill(PyObject *self, int rows, int cols, double val) {
    matrix *new_mat;
    int alloc_failed = allocate_matrix(&new_mat, rows, cols);
    if (alloc_failed)
        return alloc_failed;
    else {
        fill_matrix(new_mat, val);
        ((Matrix61c *)self)->mat = new_mat;
        ((Matrix61c *)self)->shape = get_shape(new_mat->rows, new_mat->cols);
    }
    return 0;
}

/*
 * Matrix(rows, cols, 1d_list). Fill a matrix with dimension rows * cols with 1d_list values
 */
int init_1d(PyObject *self, int rows, int cols, PyObject *lst) {
    if (rows * cols != PyList_Size(lst)) {
        PyErr_SetString(PyExc_ValueError, "Incorrect number of elements in list");
        return -1;
    }
    matrix *new_mat;
    int alloc_failed = allocate_matrix(&new_mat, rows, cols);
    if (alloc_failed) return alloc_failed;
    int count = 0;
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            set(new_mat, i, j, PyFloat_AsDouble(PyList_GetItem(lst, count)));
            count++;
        }
    }
    ((Matrix61c *)self)->mat = new_mat;
    ((Matrix61c *)self)->shape = get_shape(new_mat->rows, new_mat->cols);
    return 0;
}

/*
 * Matrix(2d_list). Fill a matrix with dimension len(2d_list) * len(2d_list[0])
 */
int init_2d(PyObject *self, PyObject *lst) {
    int rows = PyList_Size(lst);
    if (rows == 0) {
        PyErr_SetString(PyExc_ValueError,
                        "Cannot initialize numc.Matrix with an empty list");
        return -1;
    }
    int cols;
    if (!PyList_Check(PyList_GetItem(lst, 0))) {
        PyErr_SetString(PyExc_ValueError, "List values not valid");
        return -1;
    } else {
        cols = PyList_Size(PyList_GetItem(lst, 0));
    }
    for (int i = 0; i < rows; i++) {
        if (!PyList_Check(PyList_GetItem(lst, i)) ||
                PyList_Size(PyList_GetItem(lst, i)) != cols) {
            PyErr_SetString(PyExc_ValueError, "List values not valid");
            return -1;
        }
    }
    matrix *new_mat;
    int alloc_failed = allocate_matrix(&new_mat, rows, cols);
    if (alloc_failed) return alloc_failed;
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            set(new_mat, i, j,
                PyFloat_AsDouble(PyList_GetItem(PyList_GetItem(lst, i), j)));
        }
    }
    ((Matrix61c *)self)->mat = new_mat;
    ((Matrix61c *)self)->shape = get_shape(new_mat->rows, new_mat->cols);
    return 0;
}

/*
 * This deallocation function is called when reference count is 0
 */
void Matrix61c_dealloc(Matrix61c *self) {
    deallocate_matrix(self->mat);
    Py_TYPE(self)->tp_free(self);
}

/* For immutable types all initializations should take place in tp_new */
PyObject *Matrix61c_new(PyTypeObject *type, PyObject *args,
                        PyObject *kwds) {
    /* size of allocated memory is tp_basicsize + nitems*tp_itemsize*/
    Matrix61c *self = (Matrix61c *)type->tp_alloc(type, 0);
    return (PyObject *)self;
}

/*
 * This matrix61c type is mutable, so needs init function. Return 0 on success otherwise -1
 */
int Matrix61c_init(PyObject *self, PyObject *args, PyObject *kwds) {
    /* Generate random matrices */
    if (kwds != NULL) {
        PyObject *rand = PyDict_GetItemString(kwds, "rand");
        if (!rand) {
            PyErr_SetString(PyExc_TypeError, "Invalid arguments");
            return -1;
        }
        if (!PyBool_Check(rand)) {
            PyErr_SetString(PyExc_TypeError, "Invalid arguments");
            return -1;
        }
        if (rand != Py_True) {
            PyErr_SetString(PyExc_TypeError, "Invalid arguments");
            return -1;
        }

        PyObject *low = PyDict_GetItemString(kwds, "low");
        PyObject *high = PyDict_GetItemString(kwds, "high");
        PyObject *seed = PyDict_GetItemString(kwds, "seed");
        double double_low = 0;
        double double_high = 1;
        unsigned int unsigned_seed = 0;

        if (low) {
            if (PyFloat_Check(low)) {
                double_low = PyFloat_AsDouble(low);
            } else if (PyLong_Check(low)) {
                double_low = PyLong_AsLong(low);
            }
        }

        if (high) {
            if (PyFloat_Check(high)) {
                double_high = PyFloat_AsDouble(high);
            } else if (PyLong_Check(high)) {
                double_high = PyLong_AsLong(high);
            }
        }

        if (double_low >= double_high) {
            PyErr_SetString(PyExc_TypeError, "Invalid arguments");
            return -1;
        }

        // Set seed if argument exists
        if (seed) {
            if (PyLong_Check(seed)) {
                unsigned_seed = PyLong_AsUnsignedLong(seed);
            }
        }

        PyObject *rows = NULL;
        PyObject *cols = NULL;
        if (PyArg_UnpackTuple(args, "args", 2, 2, &rows, &cols)) {
            if (rows && cols && PyLong_Check(rows) && PyLong_Check(cols)) {
                return init_rand(self, PyLong_AsLong(rows), PyLong_AsLong(cols), unsigned_seed, double_low,
                                 double_high);
            }
        } else {
            PyErr_SetString(PyExc_TypeError, "Invalid arguments");
            return -1;
        }
    }
    PyObject *arg1 = NULL;
    PyObject *arg2 = NULL;
    PyObject *arg3 = NULL;
    if (PyArg_UnpackTuple(args, "args", 1, 3, &arg1, &arg2, &arg3)) {
        /* arguments are (rows, cols, val) */
        if (arg1 && arg2 && arg3 && PyLong_Check(arg1) && PyLong_Check(arg2) && (PyLong_Check(arg3)
                || PyFloat_Check(arg3))) {
            if (PyLong_Check(arg3)) {
                return init_fill(self, PyLong_AsLong(arg1), PyLong_AsLong(arg2), PyLong_AsLong(arg3));
            } else
                return init_fill(self, PyLong_AsLong(arg1), PyLong_AsLong(arg2), PyFloat_AsDouble(arg3));
        } else if (arg1 && arg2 && arg3 && PyLong_Check(arg1) && PyLong_Check(arg2) && PyList_Check(arg3)) {
            /* Matrix(rows, cols, 1D list) */
            return init_1d(self, PyLong_AsLong(arg1), PyLong_AsLong(arg2), arg3);
        } else if (arg1 && PyList_Check(arg1) && arg2 == NULL && arg3 == NULL) {
            /* Matrix(rows, cols, 1D list) */
            return init_2d(self, arg1);
        } else if (arg1 && arg2 && PyLong_Check(arg1) && PyLong_Check(arg2) && arg3 == NULL) {
            /* Matrix(rows, cols, 1D list) */
            return init_fill(self, PyLong_AsLong(arg1), PyLong_AsLong(arg2), 0);
        } else {
            PyErr_SetString(PyExc_TypeError, "Invalid arguments");
            return -1;
        }
    } else {
        PyErr_SetString(PyExc_TypeError, "Invalid arguments");
        return -1;
    }
}

/*
 * List of lists representations for matrices
 */
PyObject *Matrix61c_to_list(Matrix61c *self) {
    int rows = self->mat->rows;
    int cols = self->mat->cols;
    PyObject *py_lst = NULL;
    if (self->mat->is_1d) {  // If 1D matrix, print as a single list
        py_lst = PyList_New(rows * cols);
        int count = 0;
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                PyList_SetItem(py_lst, count, PyFloat_FromDouble(get(self->mat, i, j)));
                count++;
            }
        }
    } else {  // if 2D, print as nested list
        py_lst = PyList_New(rows);
        for (int i = 0; i < rows; i++) {
            PyList_SetItem(py_lst, i, PyList_New(cols));
            PyObject *curr_row = PyList_GetItem(py_lst, i);
            for (int j = 0; j < cols; j++) {
                PyList_SetItem(curr_row, j, PyFloat_FromDouble(get(self->mat, i, j)));
            }
        }
    }
    return py_lst;
}

PyObject *Matrix61c_class_to_list(Matrix61c *self, PyObject *args) {
    PyObject *mat = NULL;
    if (PyArg_UnpackTuple(args, "args", 1, 1, &mat)) {
        if (!PyObject_TypeCheck(mat, &Matrix61cType)) {
            PyErr_SetString(PyExc_TypeError, "Argument must of type numc.Matrix!");
            return NULL;
        }
        Matrix61c* mat61c = (Matrix61c*)mat;
        return Matrix61c_to_list(mat61c);
    } else {
        PyErr_SetString(PyExc_TypeError, "Invalid arguments");
        return NULL;
    }
}

/*
 * Add class methods
 */
PyMethodDef Matrix61c_class_methods[] = {
    {"to_list", (PyCFunction)Matrix61c_class_to_list, METH_VARARGS, "Returns a list representation of numc.Matrix"},
    {NULL, NULL, 0, NULL}
};

/*
 * Matrix61c string representation. For printing purposes.
 */
PyObject *Matrix61c_repr(PyObject *self) {
    PyObject *py_lst = Matrix61c_to_list((Matrix61c *)self);
    return PyObject_Repr(py_lst);
}

/* NUMBER METHODS */

/*
 * Add the second numc.Matrix (Matrix61c) object to the first one. The first operand is
 * self, and the second operand can be obtained by casting `args`.
 */
PyObject *Matrix61c_add(Matrix61c* self, PyObject* args) {
    /* TODO: YOUR CODE HERE */
    Matrix61c* second_mat;
    if (!PyObject_TypeCheck(args, &Matrix61cType)) {
        if (!PyArg_ParseTuple(args, "O", &second_mat)) {
            PyErr_SetString(PyExc_TypeError, "Numc cannot add ('+') other than a numc matrix");
            return NULL;
        }
        if (!PyObject_TypeCheck(second_mat, &Matrix61cType)) {
            PyErr_SetString(PyExc_TypeError, "Numc cannot add ('+') other than a numc matrix");
            return NULL;
        }
    } else {
        second_mat = (Matrix61c*)args;
    }
    if (self->mat->rows <= 0 || self->mat->cols <= 0 || second_mat->mat->rows <= 0 || second_mat->mat->cols <= 0 || self->mat->rows != second_mat->mat->rows || self->mat->cols != second_mat->mat->cols) {
        PyErr_SetString(PyExc_ValueError, "Both your matrices do not have the same dimensions");
        return NULL;
    }
    Matrix61c* result_mat = (Matrix61c*) Matrix61c_new(&Matrix61cType, NULL, NULL);
    int val = allocate_matrix(&result_mat->mat, self->mat->rows, self->mat->cols);
    if (val != 0) {
        PyErr_SetString(PyExc_RuntimeError, "Failed to allocate memory");
        return NULL;
    }
    add_matrix(result_mat->mat, self->mat, second_mat->mat);
    result_mat->shape = get_shape(second_mat->mat->rows, second_mat->mat->cols);
    return (PyObject*)result_mat;
}

/*
 * Substract the second numc.Matrix (Matrix61c) object from the first one. The first operand is
 * self, and the second operand can be obtained by casting `args`.
 */
PyObject *Matrix61c_sub(Matrix61c* self, PyObject* args) {
    /* TODO: YOUR CODE HERE */
    Matrix61c* second_mat;
    if (!PyObject_TypeCheck(args, &Matrix61cType)) {
        if (! PyArg_ParseTuple(args, "O", &second_mat)) {
            PyErr_SetString(PyExc_TypeError, "Numc cannot sub ('-') other than a numc matrix");
            return NULL;
        }
        if (!PyObject_TypeCheck(second_mat, &Matrix61cType)) {
            PyErr_SetString(PyExc_TypeError, "Numc cannot sub ('-') other than a numc matrix");
            return NULL;
        }
    } else {
        second_mat = (Matrix61c*)args;
    }
    if (self->mat->rows <= 0 || self->mat->cols <= 0 || second_mat->mat->rows <= 0 || second_mat->mat->cols <= 0 || self->mat->rows != second_mat->mat->rows || self->mat->cols != second_mat->mat->cols) {
        PyErr_SetString(PyExc_ValueError, "Both your matrices do not have the same dimensions");
        return NULL;
    }
    Matrix61c* result_mat = (Matrix61c*) Matrix61c_new(&Matrix61cType, NULL, NULL);
    int val = allocate_matrix(&result_mat->mat, self->mat->rows, self->mat->cols);
    if (val != 0) {
        PyErr_SetString(PyExc_RuntimeError, "Failed to allocate memory");
        return NULL;
    }
    sub_matrix(result_mat->mat, self->mat, second_mat->mat);
    result_mat->shape = get_shape(second_mat->mat->rows, second_mat->mat->cols);
    return (PyObject*)result_mat;
}

/*
 * NOT element-wise multiplication. The first operand is self, and the second operand
 * can be obtained by casting `args`.
 */
PyObject *Matrix61c_multiply(Matrix61c* self, PyObject *args) {
    Matrix61c* second_mat;
    if (!PyObject_TypeCheck(args, &Matrix61cType)) {
        if (! PyArg_ParseTuple(args, "O", &second_mat)) {
            PyErr_SetString(PyExc_TypeError, "Numc cannot mul ('*') other than a numc matrix");
            return NULL;
        }
        if (!PyObject_TypeCheck(second_mat, &Matrix61cType)) {
            PyErr_SetString(PyExc_TypeError, "Numc cannot mul ('*') other than a numc matrix");
            return NULL;
        }
    } else {
        second_mat = (Matrix61c*)args;
    }
    if (self->mat->rows <= 0 || self->mat->cols <= 0 || second_mat->mat->rows <= 0 || second_mat->mat->cols <= 0 || self->mat->cols != second_mat->mat->rows) {
        PyErr_SetString(PyExc_ValueError, "Dimensions are not appropriate for multiplication");
        return NULL;
    }
    Matrix61c* result_mat = (Matrix61c*) Matrix61c_new(&Matrix61cType, NULL, NULL);
    int val = allocate_matrix(&result_mat->mat, self->mat->rows, second_mat->mat->cols); //rows of m1, cols of m2
    if (val != 0) {
        PyErr_SetString(PyExc_RuntimeError, "Failed to allocate memory");
        return NULL;
    }
    mul_matrix(result_mat->mat, self->mat, second_mat->mat);
    result_mat->shape = get_shape(self->mat->rows, second_mat->mat->cols); //rows of m1, cols of m2
    return (PyObject*)result_mat;
}

/*
 * Negates the given numc.Matrix.
 */
PyObject *Matrix61c_neg(Matrix61c* self) {
    /* TODO: YOUR CODE HERE */
    if (self->mat->rows <= 0 || self->mat->cols <= 0) {
        PyErr_SetString(PyExc_ValueError, "Not an appropriately sized matrix");
        return NULL;
    }
    Matrix61c* result_mat = (Matrix61c*) Matrix61c_new(&Matrix61cType, NULL, NULL);
    int val = allocate_matrix(&result_mat->mat, self->mat->rows, self->mat->cols);
    if (val != 0) {
        PyErr_SetString(PyExc_RuntimeError, "Failed to allocate memory");
        return NULL;
    }
    neg_matrix(result_mat->mat, self->mat);
    result_mat->shape = get_shape(self->mat->rows, self->mat->cols);
    return (PyObject*)result_mat;
}

/*
 * Take the element-wise absolute value of this numc.Matrix.
 */
PyObject *Matrix61c_abs(Matrix61c *self) {
    if (self->mat->rows <= 0 || self->mat->cols <= 0) {
        PyErr_SetString(PyExc_ValueError, "Not an appropriately sized matrix");
        return NULL;
    }
    Matrix61c* result_mat = (Matrix61c*) Matrix61c_new(&Matrix61cType, NULL, NULL);
    int val = allocate_matrix(&result_mat->mat, self->mat->rows, self->mat->cols);
    if (val != 0) {
        PyErr_SetString(PyExc_RuntimeError, "Failed to allocate memory");
        return NULL;
    }
    abs_matrix(result_mat->mat, self->mat);
    result_mat->shape = get_shape(self->mat->rows, self->mat->cols);
    return (PyObject*)result_mat;
}

/*
 * Raise numc.Matrix (Matrix61c) to the `pow`th power. You can ignore the argument `optional`.
 */
PyObject *Matrix61c_pow(Matrix61c *self, PyObject *pow, PyObject *optional) {
    /* TODO: YOUR CODE HERE */
    int pow_int;
    if (self->mat->rows <= 0 || self->mat->cols <= 0) {
        PyErr_SetString(PyExc_ValueError, "Not an appropriately sized matrix");
        return NULL;
    }
    if (self->mat->rows != self->mat->cols){
        PyErr_SetString(PyExc_ValueError, "Not a square matrix");
        return NULL;
    }
    if (!(PyLong_Check(pow))){
        PyErr_SetString(PyExc_TypeError, "Pow is not an integer");
        return NULL;
    } else {
        pow_int = (long) PyLong_AsLong(pow);
    }

    if ((pow_int) < 0){
        PyErr_SetString(PyExc_ValueError, "Pow cannot be negative");
        return NULL;
    }

    Matrix61c* result_mat = (Matrix61c*) Matrix61c_new(&Matrix61cType, NULL, NULL);
    int val = allocate_matrix(&result_mat->mat, self->mat->rows, self->mat->cols);
    if (val != 0) {
        PyErr_SetString(PyExc_RuntimeError, "Failed to allocate memory");
        return NULL;
    }
    pow_matrix(result_mat->mat, self->mat, pow_int);
    result_mat->shape = get_shape(self->mat->rows, self->mat->cols);
    return (PyObject*)result_mat;
}

/*
 * Create a PyNumberMethods struct for overloading operators with all the number methods you have
 * define. You might find this link helpful: https://docs.python.org/3.6/c-api/typeobj.html
 */
PyNumberMethods Matrix61c_as_number = {
    /* TODO: YOUR CODE HERE */
    (binaryfunc)Matrix61c_add, 
    (binaryfunc)Matrix61c_sub, 
    (binaryfunc)Matrix61c_multiply,
    0,
    0,
    (ternaryfunc)Matrix61c_pow,  
    (unaryfunc)Matrix61c_neg, 
    0,
    (unaryfunc)Matrix61c_abs
};


/* INSTANCE METHODS */

/*
 * Given a numc.Matrix self, parse `args` to (int) row, (int) col, and (double/int) val.
 * Return None in Python (this is different from returning null).
 */
PyObject *Matrix61c_set_value(Matrix61c *self, PyObject* args) {
    /* TODO: YOUR CODE HERE */
    if (PyTuple_Size(args) != 3) {
        PyErr_SetString(PyExc_TypeError, "Number of arguments parsed in is not 3");
    }

    int row, col;
    double val;

    if (!PyArg_ParseTuple(args, "iid", &row, &col, &val)) {
        PyErr_SetString(PyExc_TypeError, "Indices can only be integers and val must be a float/int");
        return NULL;
    }
    if (row < 0 || row >= self->mat->rows) {
        PyErr_SetString(PyExc_IndexError, "Row (i) out of bounds");
        return NULL;
    }
    if (col < 0 || col >= self->mat->cols) {
        PyErr_SetString(PyExc_IndexError, "Col (j) out of bounds");
        return NULL;
    }

    set(self->mat, row, col, val);

    Py_INCREF(Py_None);
    return Py_None;

}

/*
 * Given a numc.Matrix `self`, parse `args` to (int) row and (int) col.
 * Return the value at the `row`th row and `col`th column, which is a Python
 * float/int.
 */
PyObject *Matrix61c_get_value(Matrix61c *self, PyObject* args) {
    /* TODO: YOUR CODE HERE */
    if (PyTuple_Size(args) != 2) {
        PyErr_SetString(PyExc_TypeError, "Number of arguments parsed in is not 2");
    }

    int row, col;

    if (!PyArg_ParseTuple(args, "ii", &row, &col)) {
        PyErr_SetString(PyExc_TypeError, "Indices can only be integers");
        return NULL;
    }
    if (row < 0 || row >= self->mat->rows) {
        PyErr_SetString(PyExc_IndexError, "Row (i) out of bounds");
        return NULL;
    }
    if (col < 0 || col >= self->mat->cols) {
        PyErr_SetString(PyExc_IndexError, "Col (j) out of bounds");
        return NULL;
    }

    double value = get(self->mat, row, col);
    return PyFloat_FromDouble(value);
    
}

/*
 * Create an array of PyMethodDef structs to hold the instance methods.
 */
PyMethodDef Matrix61c_methods[] = {
    {"set", (PyCFunction) Matrix61c_set_value, METH_VARARGS, "Given a numc.Matrix self, parse `args` to (int) row, (int) col, and (double/int) val. Return None in Python (this is different from returning null)."},
    {"get", (PyCFunction) Matrix61c_get_value, METH_VARARGS, "Given a numc.Matrix `self`, parse `args` to (int) row and (int) col. Return the value at the `row`th row and `col`th column, which is a Python float/int."},
    {NULL, NULL, 0, NULL} //DO NOT DELETE
};

/* INDEXING */

/*
 * Given a numc.Matrix `self`, index into it with `key`. Return the indexed result.
 */
PyObject *Matrix61c_subscript(Matrix61c* self, PyObject* key) {
    //Conditions: 1) 2D, Integer slice; 2) 2D, single slice; 3) 2D, tuple of slices; 4) 1D, Integer slice; 5) 1D, single slice;
    
    //2D MATRIX
    if ((self->mat->rows)>1 && (self->mat->cols)>1){

        //1. Key is a single number
        if (PyLong_Check(key)){
            int slice_int = (int) PyLong_AsLong(key);
            if (slice_int < 0 || slice_int >= self->mat->rows){
                PyErr_SetString(PyExc_IndexError, "Index out of range");
                return NULL;
            }

            Matrix61c* slice_matrix = (Matrix61c*) Matrix61c_new(&Matrix61cType, NULL, NULL);
            allocate_matrix_ref(&slice_matrix->mat, self->mat, slice_int, 0, 1, self->mat->cols);
            slice_matrix->shape = get_shape(1, self->mat->cols);
            
            return (PyObject*) slice_matrix;

            }
        //2. Key is a single slice
        else if (PySlice_Check(key)){
            Py_ssize_t start;
            Py_ssize_t stop;
            Py_ssize_t step;
            Py_ssize_t slice_length;

            int a =  PySlice_GetIndicesEx(key,self->mat->rows, &start, &stop, &step, &slice_length);
            if (a < 0){
                PyErr_SetString(PyExc_ValueError, "Slice info not valid!");
                return NULL;
            }

            // printf("arrives at slice checkpoint subscript \n");
            // printf("slice length: %d \n", (int) slice_length);
            // printf("rows: %d \n", self->mat->rows);
            if ((int)stop - (int)start > self->mat->rows) {
                PyErr_SetString(PyExc_ValueError, "Slice info not valid!");
                return NULL;
            }

            if((step != 1)||(slice_length < 1)){
                PyErr_SetString(PyExc_ValueError, "Slice info not valid!");
                return NULL;
            }
            Matrix61c* slice_matrix = (Matrix61c*) Matrix61c_new(&Matrix61cType, NULL, NULL);
            allocate_matrix_ref(&slice_matrix->mat, self->mat, start, 0, slice_length, self->mat->cols);
            slice_matrix->shape = get_shape(slice_length, self->mat->cols);
            
            return (PyObject*) slice_matrix;
        }
        //3. Key is a mix of slices and tuples
        else {
            //ensure key is not long or float, and tuple size is 2
            if ((!PyFloat_Check(key) && !PyLong_Check(key)) && PyTuple_Size(key)==2){
                PyObject* slice1 = PyTuple_GetItem(key, 0);//gets the two elements of the arg
                PyObject* slice2 = PyTuple_GetItem(key, 1);   
                // tuple is (slice, slice)
                if (PySlice_Check(slice1) && PySlice_Check(slice2)){
                    //slice 1
                    Py_ssize_t start1;
                    Py_ssize_t stop1;
                    Py_ssize_t step1;
                    Py_ssize_t slice_length1;
                    
                    int a = PySlice_GetIndicesEx(slice1,self->mat->rows, &start1, &stop1, &step1, &slice_length1);
                    if (a < 0){
                        PyErr_SetString(PyExc_ValueError, "Slice info not valid!");
                        return NULL;
                    }

                    if((step1 != 1)||(slice_length1 < 1)){
                        PyErr_SetString(PyExc_ValueError, "Slice info not valid!");
                        return NULL;
                    }
                    //slice 2
                    Py_ssize_t start2;
                    Py_ssize_t stop2;
                    Py_ssize_t step2;
                    Py_ssize_t slice_length2;
                    
                    int b = PySlice_GetIndicesEx(slice2,self->mat->cols, &start2, &stop2, &step2, &slice_length2);
                    if (b < 0){
                        PyErr_SetString(PyExc_ValueError, "Slice info not valid!");
                        return NULL;
                    }

                    if((step2 != 1)||(slice_length2 < 1)){
                        PyErr_SetString(PyExc_ValueError, "Slice info not valid!");
                        return NULL;
                    }


                    Matrix61c* slice_matrix = (Matrix61c*) Matrix61c_new(&Matrix61cType, NULL, NULL);
                    allocate_matrix_ref(&slice_matrix->mat, self->mat, start1, start2, slice_length1, slice_length2);
                    slice_matrix->shape = get_shape(slice_length1, slice_length2);
                    

                    if (slice_length1 == 1 && slice_length2 == 1) {
                        double v = slice_matrix->mat->data[0][0];
                        return PyFloat_FromDouble(v);
                    } else {
                        return (PyObject*) slice_matrix;
                    }

                        
                }
                //if tuple is (slice, int)
                else if (PySlice_Check(slice1) && PyLong_Check(slice2)){
                    //slice 1
                    Py_ssize_t start1;
                    Py_ssize_t stop1;
                    Py_ssize_t step1;
                    Py_ssize_t slice_length1;
                    
                    int a = PySlice_GetIndicesEx(slice1,self->mat->rows, &start1, &stop1, &step1, &slice_length1);
                    if (a < 0){
                        PyErr_SetString(PyExc_ValueError, "Slice info not valid!");
                        return NULL;
                    }

                    if((step1 != 1)||(slice_length1 < 1)){
                        PyErr_SetString(PyExc_ValueError, "Slice info not valid!");
                        return NULL;
                    }
                    
                    int slice_int_2 = (int) PyLong_AsLong(slice2);

                    if (slice_int_2 >= self->mat->cols || slice_int_2 < 0){
                        PyErr_SetString(PyExc_IndexError, "Index out of range!");
                        return NULL;
                    }
                    
                    Matrix61c* slice_matrix = (Matrix61c*) Matrix61c_new(&Matrix61cType, NULL, NULL);
                    allocate_matrix_ref(&slice_matrix->mat, self->mat, start1, slice_int_2, slice_length1, 1);
                    slice_matrix->shape = get_shape(slice_length1, 1);
                    

                    if (slice_length1 == 1) {
                        double v = slice_matrix->mat->data[0][0];
                        return PyFloat_FromDouble(v);
                    } else {
                        return (PyObject*) slice_matrix;
                    }
                }
                //if tuple is (int, slice) 
                else if (PyLong_Check(slice1) && PySlice_Check(slice2)){
                    //integer 1
                    int slice_int_1 = (int) PyLong_AsLong(slice1);

                    if (slice_int_1 >= self->mat->rows || slice_int_1 < 0){
                        PyErr_SetString(PyExc_IndexError, "Index out of range!");
                        return NULL;
                    }
                    
                    //slice 2
                    Py_ssize_t start2;
                    Py_ssize_t stop2;
                    Py_ssize_t step2;
                    Py_ssize_t slice_length2;
                    
                    int a = PySlice_GetIndicesEx(slice2,self->mat->cols, &start2, &stop2, &step2, &slice_length2);
                    if (a < 0){
                        PyErr_SetString(PyExc_ValueError, "Slice info not valid!");
                        return NULL;
                    }


                    if((step2 != 1)||(slice_length2 < 1)){
                        PyErr_SetString(PyExc_ValueError, "Slice info not valid!");
                        return NULL;
                    }
                    
                    Matrix61c* slice_matrix = (Matrix61c*) Matrix61c_new(&Matrix61cType, NULL, NULL);
                    allocate_matrix_ref(&slice_matrix->mat, self->mat, slice_int_1, start2, 1, slice_length2);
                    slice_matrix->shape = get_shape(1, slice_length2);
                    
                    
                    if (slice_length2 == 1) {
                        double v = slice_matrix->mat->data[0][0];
                        return PyFloat_FromDouble(v);
                    } else {
                        return (PyObject*) slice_matrix;
                    }
                }
                //if tuple is (int, int)
                else if (PyLong_Check(slice1) && PyLong_Check(slice2)){
                    //integer 1
                    int slice_int_1 = (int) PyLong_AsLong(slice1);
                    if (slice_int_1 >= self->mat->rows || slice_int_1 < 0){
                        PyErr_SetString(PyExc_IndexError, "Index out of range!");
                        return NULL;
                    }

                    //integer 2
                    int slice_int_2 = (int) PyLong_AsLong(slice2);
                    if (slice_int_2 >= self->mat->cols || slice_int_2 < 0){
                        PyErr_SetString(PyExc_IndexError, "Index out of range!");
                        return NULL;
                    }
                    //convert 1x1 slice int to PyObject
                    Matrix61c* slice_matrix = (Matrix61c*) Matrix61c_new(&Matrix61cType, NULL, NULL);
                    allocate_matrix_ref(&slice_matrix->mat, self->mat, slice_int_1, slice_int_2, 1, 1);
                    slice_matrix->shape = get_shape(1, 1);
                    //printf("VALUEEEEEE %f", slice_matrix->mat->data[0][0]);
                    double v = slice_matrix->mat->data[0][0];
                    return PyFloat_FromDouble(v);
                }
                else {
                    PyErr_SetString(PyExc_TypeError, "Key is not valid");
                    return NULL;
                }

            } 
            //else invalid key
            else {
                PyErr_SetString(PyExc_TypeError, "Key is not valid");
                return NULL;
            }
        }
    } 
    //1D matrix
    else if ((self->mat->rows)==1 || (self->mat->cols)==1){
        //1. Key is a single number
        if (PyLong_Check(key)){
            int slice_int = (int) PyLong_AsLong(key);
            // if cols==1, check slice against rows
      
            //COLUMN = 1
            if (self->mat->cols == 1){
                if (slice_int < 0 || slice_int >= self->mat->rows){
                    PyErr_SetString(PyExc_IndexError, "Index out of range");
                    return NULL;
                }
                Matrix61c* slice_matrix = (Matrix61c*) Matrix61c_new(&Matrix61cType, NULL, NULL);
                allocate_matrix_ref(&slice_matrix->mat, self->mat, slice_int, 0, 1, 1);
                slice_matrix->shape = get_shape(1, 1);
                double v = slice_matrix->mat->data[0][0];
                return PyFloat_FromDouble(v);
            }
            //ROW = 1
            else {
                if (slice_int < 0 || slice_int >= self->mat->cols){
                    PyErr_SetString(PyExc_IndexError, "Index out of range");
                    return NULL;
                }
                //slice_value = PyFloat_FromDouble(self->mat->data[0][slice_int]);
                Matrix61c* slice_matrix = (Matrix61c*) Matrix61c_new(&Matrix61cType, NULL, NULL);
                allocate_matrix_ref(&slice_matrix->mat, self->mat, 0, slice_int, 1, 1);
                slice_matrix->shape = get_shape(1, 1);
                double v = slice_matrix->mat->data[0][0];
                return PyFloat_FromDouble(v);
            }
            
        }
        //2. Key is a single slice
        else if (PySlice_Check(key)){
            Py_ssize_t start;
            Py_ssize_t stop;
            Py_ssize_t step;
            Py_ssize_t slice_length;
            Matrix61c* slice_matrix = (Matrix61c*) Matrix61c_new(&Matrix61cType, NULL, NULL);
        
            // COLUMN = 1
            if (self->mat->cols == 1){
                int a = PySlice_GetIndicesEx(key,self->mat->rows, &start, &stop, &step, &slice_length);
                if (a < 0){
                    PyErr_SetString(PyExc_ValueError, "Slice info not valid!");
                    return NULL;
                }

                if((step != 1) || (slice_length < 1)){
                    PyErr_SetString(PyExc_ValueError, "Slice info not valid!");
                    return NULL;
                }
                allocate_matrix_ref(&slice_matrix->mat, self->mat, start, 0, slice_length, 1);
                slice_matrix->shape = get_shape(slice_length, 1);
               
            }
            // ROW = 1
            else {
                int a = PySlice_GetIndicesEx(key,self->mat->cols, &start, &stop, &step, &slice_length);
                if (a < 0){
                    PyErr_SetString(PyExc_ValueError, "Slice info not valid!");
                    return NULL;
                }
                
                if((step != 1)||(slice_length < 1)){
                    PyErr_SetString(PyExc_ValueError, "Slice info not valid!");
                    return NULL;
                }
                allocate_matrix_ref(&slice_matrix->mat, self->mat, 0, start, 1, slice_length);
                slice_matrix->shape = get_shape(1, slice_length);
                
            }
            //slice_length == 1
            if (slice_length == 1) {
                double v = slice_matrix->mat->data[0][0];
                return PyFloat_FromDouble(v);
            } else {
                return (PyObject*) slice_matrix;
            }
        }
        //3. Key is invalid
        else {
            PyErr_SetString(PyExc_TypeError, "Key is not valid");
            return NULL;
        }
    } 
    else {
        PyErr_SetString(PyExc_RuntimeError, "Operation can only support 1D and 2D matrices");
        return NULL;
    }
}

/*
 * Given a numc.Matrix `self`, index into it with `key`, and set the indexed result to `v`.
 */
int Matrix61c_set_subscript(Matrix61c* self, PyObject *key, PyObject *v) {
    //2D MATRIX
    if ((self->mat->rows) > 1 && (self->mat->cols) > 1) {
        //TypeErrorCheck - only 2nd TypeError case for 2x2 matrix. Check if val is a list.
        //applies to all subcases within 2D
                
        //check if v is null list - edge case
        if (PyList_Check(v)) {
            if (PyList_Size(v) == 0){
                PyErr_SetString(PyExc_ValueError, "v has incorrect length");
                return -1;
            }
        }


        //1. Key is a single number i.e. resulting slice can only be 1xn or nx1 matrix, given a 2D original matrix
        if (PyLong_Check(key)) {
            //Type check: 2D slice
            if (!PyList_Check(v)){
                PyErr_SetString(PyExc_TypeError, "Resulting slice is not 1 by 1, but v is not a list.");
                return -1;
            }
            if ((!PyList_Check(PyList_GetItem(v,(Py_ssize_t) 0)) && PyList_Check(v))) { //v is a 1D list
                for(int i = 0; i < PyList_Size(v); i++) { //check if values inside v are not float/int
                    if (!PyLong_Check(PyList_GetItem(v,(Py_ssize_t) i)) && !PyFloat_Check(PyList_GetItem(v,(Py_ssize_t) i))) {
                        PyErr_SetString(PyExc_ValueError, "Elements of v are not floats or ints");
                    }
                }

                //SizeCheck
                if ((int)PyList_Size(v) != self->mat->cols) {
                    PyErr_SetString(PyExc_ValueError, "v has incorrect length");
                    return -1;
                }

                //IndexOutOfRangeCheck                
                int slice_int = (int) PyLong_AsLong(key);
                if (slice_int < 0 || slice_int >= self->mat->rows) {
                    PyErr_SetString(PyExc_IndexError, "Index out of range");
                    return -1;
                }

                //set slice_matrix
                Matrix61c* val = (Matrix61c*) Matrix61c_subscript(self, key);
                //LOGIC: set val row to v
                for (int c = 0; c<val->mat->cols; c++) {
                    val->mat->data[0][c] = PyLong_AsLong(PyList_GetItem(v,c)); //directly modifies the original matrix, since subscript is being called above
                }

            //if v is not 1D, ValueError
            } else {
                PyErr_SetString(PyExc_ValueError, "v is not a 1D list");
                return -1;
            }
            
            return 0;
        }
    //2. Key is a single slice
        else if (PySlice_Check(key)) { //key is a slice, resulting matrix can be a 1xn or nx1 or 2D matrix, given a 2D original matrix

            if (!PyList_Check(v)) {
                    PyErr_SetString(PyExc_TypeError, "Resulting slice is not 1 by 1, but v is not a list.");
                    return -1;
            }

            Py_ssize_t start;
            Py_ssize_t stop;
            Py_ssize_t step;
            Py_ssize_t slice_length;

            int a =  PySlice_GetIndicesEx(key,self->mat->rows, &start, &stop, &step, &slice_length);
            if (a < 0){
                PyErr_SetString(PyExc_ValueError, "Slice info not valid!");
                return -1;
            }

            if ((int) step != 1) {
                PyErr_SetString(PyExc_ValueError, "Slice info not valid!");
                return -1;
            }

            if ((int) slice_length < 1) {
                PyErr_SetString(PyExc_ValueError, "Slice info not valid!");
                return -1;
                
            } else if ((int) slice_length == 1) { //first check if resulting matrix slice is 1D

                if (((!PyList_Check(PyList_GetItem(v,(Py_ssize_t) 0)) && PyList_Check(v)))) { 
                for(int i = 0; i < PyList_Size(v); i++) { //check if values inside v are not float/int
                    if (!PyLong_Check(PyList_GetItem(v,(Py_ssize_t) i)) && !PyFloat_Check(PyList_GetItem(v,(Py_ssize_t) i))) {
                        PyErr_SetString(PyExc_ValueError, "Elements of v are not floats or ints");
                    }
                }

                //size check - v has wrong length
                if ((int)PyList_Size(v) != self->mat->cols) {
                    PyErr_SetString(PyExc_ValueError, "v has incorrect length");
                    return -1;
                }

                // set slice_matrix
                Matrix61c* val = (Matrix61c*) Matrix61c_subscript(self, key);
                //LOGIC: set 1D matrix val to 1D list v
                for (int c = 0; c<val->mat->cols; c++) {
                    val->mat->data[0][c] = PyLong_AsLong(PyList_GetItem(v,c));//directly modifies the original matrix
                }
                
                return 0; //return 0 if 1D successful
                }

            } else if ((int) slice_length > 1 && (int) slice_length <= self->mat->rows) { //resulting matrix slice is 2D, therefore v needs to be 2D
                //FAT HEADACHE

                for(int i = 0; i < PyList_Size(v); i++) { //check if values inside v are not float/int
                    for (int j = 0; j < PyList_Size(PyList_GetItem(v,(Py_ssize_t) i)); j++) {
                        if (!PyLong_Check(PyList_GetItem(PyList_GetItem(v, i),j)) && !PyFloat_Check(PyList_GetItem(PyList_GetItem(v, i),j))) {
                            PyErr_SetString(PyExc_ValueError, "Elements of v are not floats or ints");
                            return -1;
                        }
                    }

                    //printf("slice length is greater than 1 \n");

                    if (!(PyList_Check(PyList_GetItem(v,(Py_ssize_t) i))) || !PyList_Check(v) || PyList_Size(PyList_GetItem(v,(Py_ssize_t) i)) != self->mat->cols) { //check if v is 2D
                            PyErr_SetString(PyExc_ValueError, "Resulting slice is 2D, but v is not a complete 2D list");
                            return -1;
                    }           
                }

                // v has wrong length in rows
                if ((int)PyList_Size(v) != slice_length) {
                    PyErr_SetString(PyExc_ValueError, "v has incorrect length in rows");
                    return -1;
                }

                // v has wrong length in cols
                if ((int)PyList_Size(PyList_GetItem(v, 0)) != self->mat->cols) {
                    PyErr_SetString(PyExc_ValueError, "v has incorrect length in cols");
                    return -1;
                }

                // set slice_matrix
                Matrix61c* val = (Matrix61c*) Matrix61c_subscript(self, key);
                //LOGIC: set val rows and cols to 2D matrix v
                for (int r = 0; r<val->mat->rows; r++){
                    for (int c = 0; c<val->mat->cols; c++) {
                        val->mat->data[r][c] = PyLong_AsLong(PyList_GetItem(PyList_GetItem(v, r),c)); // directly modifies the original matrix
                    }
                }

                return 0;

            } else { // slice_length is too big
                PyErr_SetString(PyExc_ValueError, "Slice info not valid! Slice length may exceed bounds");
                return -1;
            }
        }  
        // Key is a mix of slices and tuples
        else {
          
            //ensure key is not long or float, and is indeed a tuple of size 2
            if ((!PyFloat_Check(key) && !PyLong_Check(key)) && PyTuple_Size(key)==2){
                PyObject* slice1 = PyTuple_GetItem(key, 0);//gets the two elements of the arg
                PyObject* slice2 = PyTuple_GetItem(key, 1);  
                
                // tuple is (slice, slice)
                if (PySlice_Check(slice1) && PySlice_Check(slice2)){
                    //slice 1
                    Py_ssize_t start1;
                    Py_ssize_t stop1;
                    Py_ssize_t step1;
                    Py_ssize_t slice_length1;
                    
                    int a = PySlice_GetIndicesEx(slice1,self->mat->rows, &start1, &stop1, &step1, &slice_length1);
                    if (a < 0){
                        PyErr_SetString(PyExc_ValueError, "Slice info not valid!");
                        return -1;
                    }

                    if((step1 != 1)||(slice_length1 < 1)){
                        PyErr_SetString(PyExc_ValueError, "Slice info not valid!");
                        return -1;
                    }
                    //slice 2
                    Py_ssize_t start2;
                    Py_ssize_t stop2;
                    Py_ssize_t step2;
                    Py_ssize_t slice_length2;
                    
                    int b = PySlice_GetIndicesEx(slice2,self->mat->cols, &start2, &stop2, &step2, &slice_length2);
                    if (b < 0){
                        PyErr_SetString(PyExc_ValueError, "Slice info not valid!");
                        return -1;
                    }

                    if((step2 != 1)||(slice_length2 < 1)){
                        PyErr_SetString(PyExc_ValueError, "Slice info not valid!");
                        return -1;
                    }

                    // SLICES NEED TO BE CHECKED FOR SPECIFIC CONDITIONS
                    if (slice_length1 == 1 && slice_length2 == 1) { //resulting slice is a 1x1 value
                        if (PyLong_Check(v) || PyFloat_Check(v)) {
                            self->mat->data[start1][start2] = PyLong_AsLong(v);
                            if (self->mat->parent != NULL) {
                                self->mat->parent->data[start1][start2] = PyLong_AsLong(v);
                            }
                        } else {
                            //TypeError: v is not a float but 1x1 slice
                            PyErr_SetString(PyExc_TypeError, "V is not a int or float for 1x1 result slice");
                            return -1;
                        }
                    } else if ((slice_length1 == 1 && slice_length2 > 1) || (slice_length1 > 1 && slice_length2 == 1)) { //resulting slice is 1xn or nx1 matrix

                 
                        for(int i = 0; i < PyList_Size(v); i++) { //check if values inside v are not float/int
                            if (((PyList_Check(PyList_GetItem(v,(Py_ssize_t) i)) || !PyList_Check(v)))) {
                                PyErr_SetString(PyExc_TypeError, "Resulting slice is not 1 by 1, but v is not a list.");
                                return -1;
                            }
                            
                            if (!PyLong_Check(PyList_GetItem(v,(Py_ssize_t) i)) && !PyFloat_Check(PyList_GetItem(v,(Py_ssize_t) i))) {
                                PyErr_SetString(PyExc_ValueError, "Elements of v are not floats or ints");
                                return -1;
                            }
                        }

                        // set slice_matrix
                        Matrix61c* val = (Matrix61c*) Matrix61c_subscript(self, key);
                        //LOGIC: set 1D matrix val to 1D list v

                        //size check - v has wrong length
                        if (slice_length1 == 1) {
                            if ((int)PyList_Size(v) != slice_length2) {
                                PyErr_SetString(PyExc_ValueError, "v has incorrect length");
                                return -1;
                            }
                            for (int c = 0; c<val->mat->cols; c++) {
                                // printf("val->mat->data[0][c] %f \n", val->mat->data[0][c]);
                                val->mat->data[0][c] = PyLong_AsLong(PyList_GetItem(v,c));//directly modifies the original matrix
                            }
                        }
                         else if (slice_length2 == 1) {
                            
                            if ((int)PyList_Size(v) != slice_length1) {
                                PyErr_SetString(PyExc_ValueError, "v has incorrect length");
                                return -1;
                            }
                            for (int r = 0; r<val->mat->rows; r++) {
                                // printf("val->mat->data[r][0] %f \n", val->mat->data[r][0]);
                                val->mat->data[r][0] = PyLong_AsLong(PyList_GetItem(v,r));//directly modifies the original matrix
                            }
                        }

                        return 0; //return 0 if 1D successful
            

                    } else if (slice_length1 > 1 && slice_length2 > 1) { //resulting slice is 2D
                        //FATTEST HEADACHE
                        
                        //Type Check - v is not 1x1
                        if(!PyList_Check(v)){
                            PyErr_SetString(PyExc_TypeError, "Resulting slice is not 1 by 1, but v is not a list.");
                            return -1;
                        }

                        for(int i = 0; i < PyList_Size(v); i++) { //check if values inside v are not float/int
                            for (int j = 0; j < PyList_Size(PyList_GetItem(v,(Py_ssize_t) i)); j++) {
                                if (!PyLong_Check(PyList_GetItem(PyList_GetItem(v, i),j)) && !PyFloat_Check(PyList_GetItem(PyList_GetItem(v, i),j))) {
                                    PyErr_SetString(PyExc_ValueError, "Elements of v are not floats or ints");
                                    return -1;
                                }
                            }

                            //printf("slice length is greater than 1 \n");

                            // printf("slicelength1: %ld \n", slice_length1);
                            // printf("slicelength2: %ld \n", slice_length2);
                            // printf("pylistsize(i) %ld \n", PyList_Size(PyList_GetItem(v,(Py_ssize_t) i)));
                            if (!(PyList_Check(PyList_GetItem(v,(Py_ssize_t) i))) || !PyList_Check(v) || PyList_Size(PyList_GetItem(v,(Py_ssize_t) i)) != slice_length2) { //check if v is 2D
                                    // printf("enter error loop \n");
                                    PyErr_SetString(PyExc_ValueError, "Resulting slice is 2D, but v is not a complete 2D list");
                                    return -1;
                                }           
                        }

                        // v has wrong length in rows
                        if ((int)PyList_Size(v) != slice_length1) {
                            PyErr_SetString(PyExc_ValueError, "v has incorrect length in rows");
                            return -1;
                        }

                        // v has wrong length in cols
                        if ((int)PyList_Size(PyList_GetItem(v, 0)) != slice_length2) {
                            PyErr_SetString(PyExc_ValueError, "v has incorrect length in cols");
                            return -1;
                        }

                        // set slice_matrix
                        Matrix61c* val = (Matrix61c*) Matrix61c_subscript(self, key);
                        //LOGIC: set val rows and cols to 2D matrix v
                        for (int r = 0; r<val->mat->rows; r++){
                            for (int c = 0; c<val->mat->cols; c++) {
                                val->mat->data[r][c] = PyLong_AsLong(PyList_GetItem(PyList_GetItem(v, r),c)); // directly modifies the original matrix
                            }
                        }

                        return 0;

                    
                    } else { // unexpected error case
                        PyErr_SetString(PyExc_ValueError, "Slice info not valid!");
                        return -1;
                    }

                        
                }
                //if tuple is (slice, int)
                else if (PySlice_Check(slice1) && PyLong_Check(slice2)){

                    //slice 1
                    Py_ssize_t start1;
                    Py_ssize_t stop1;
                    Py_ssize_t step1;
                    Py_ssize_t slice_length1;
                    
                    int a = PySlice_GetIndicesEx(slice1,self->mat->rows, &start1, &stop1, &step1, &slice_length1);
                    if (a < 0){
                        PyErr_SetString(PyExc_ValueError, "Slice info not valid!");
                        return -1;
                    }

                    if((step1 != 1)||(slice_length1 < 1)){
                        PyErr_SetString(PyExc_ValueError, "Slice info not valid!");
                        return -1;
                    }
                    
                    int slice_int_2 = (int) PyLong_AsLong(slice2);

                    if (slice_int_2 >= self->mat->cols || slice_int_2 < 0){
                        PyErr_SetString(PyExc_IndexError, "Index out of range!");
                        return -1;
                    }

                    if (slice_length1 == 1) { //1x1 case
                        if (PyLong_Check(v) || PyFloat_Check(v)) {
                            self->mat->data[start1][slice_int_2] = PyLong_AsLong(v);
                            if (self->mat->parent != NULL) {
                                self->mat->parent->data[start1][slice_int_2] = PyLong_AsLong(v);
                            }
                        } else {
                            PyErr_SetString(PyExc_TypeError, "V is not a int or float for 1x1 result slice");
                            return -1;
                        }
                    } else { //nx1 case
                        //Type check: slice is not 1x1, but v is not a list
                        if(!PyList_Check(v)){
                            PyErr_SetString(PyExc_TypeError, "Resulting slice is not 1 by 1, but v is not a list.");
                            return -1;
                        }
                        for(int i = 0; i < PyList_Size(v); i++) { //check if values inside v are not float/int
                            if (((PyList_Check(PyList_GetItem(v,(Py_ssize_t) i)) || !PyList_Check(v)))) {
                                PyErr_SetString(PyExc_ValueError, "v is not an appropriate 1D list");
                                return -1;
                            }
                            
                            if (!PyLong_Check(PyList_GetItem(v,(Py_ssize_t) i)) && !PyFloat_Check(PyList_GetItem(v,(Py_ssize_t) i))) {
                                PyErr_SetString(PyExc_ValueError, "Elements of v are not floats or ints");
                                return -1;
                            }
                        }

                        // set slice_matrix
                        Matrix61c* val = (Matrix61c*) Matrix61c_subscript(self, key);
                        //LOGIC: set 1D matrix val to 1D list v

                        if ((int)PyList_Size(v) != slice_length1) {
                                PyErr_SetString(PyExc_ValueError, "v has incorrect length");
                                return -1;
                        }
                        for (int r = 0; r<val->mat->rows; r++) {
                            // printf("val->mat->data[r][0] %f \n", val->mat->data[r][0]);
                            val->mat->data[r][0] = PyLong_AsLong(PyList_GetItem(v,r));//directly modifies the original matrix
                        }

                    }
                }
                //if tuple is (int, slice) 
                else if (PyLong_Check(slice1) && PySlice_Check(slice2)){
                    //integer 1
                    int slice_int_1 = (int) PyLong_AsLong(slice1);

                    if (slice_int_1 >= self->mat->rows || slice_int_1 < 0){
                        PyErr_SetString(PyExc_IndexError, "Index out of range!");
                        return -1;
                    }
                    
                    //slice 2
                    Py_ssize_t start2;
                    Py_ssize_t stop2;
                    Py_ssize_t step2;
                    Py_ssize_t slice_length2;
                    
                    int a = PySlice_GetIndicesEx(slice2,self->mat->cols, &start2, &stop2, &step2, &slice_length2);
                    if (a < 0){
                        PyErr_SetString(PyExc_ValueError, "Slice info not valid!");
                        return -1;
                    }

                    if((step2 != 1)||(slice_length2 < 1)){
                        PyErr_SetString(PyExc_ValueError, "Slice info not valid!");
                        return -1;
                    }

                    if (slice_length2 == 1) { //1x1 case 
                        if (PyLong_Check(v) || PyFloat_Check(v)) {
                            self->mat->data[slice_int_1][start2] = PyLong_AsLong(v);
                            if (self->mat->parent != NULL) {
                                self->mat->parent->data[slice_int_1][start2] = PyLong_AsLong(v);
                            }
                        } else {//TypeError: Slice is 1x1, but v is not an int/float
                            PyErr_SetString(PyExc_TypeError, "V is not a int or float for 1x1 result slice");
                            return -1;
                        }
                    } else { //1xn case
                        //TypeError - slice is not 1x1, but v is not a list
                        if(!PyList_Check(v)){
                            PyErr_SetString(PyExc_TypeError, "Resulting slice is not 1 by 1, but v is not a list.");
                            return -1;
                        }
                        
                        for(int i = 0; i < PyList_Size(v); i++) { //check if values inside v are not float/int
                            if (((PyList_Check(PyList_GetItem(v,(Py_ssize_t) i)) || !PyList_Check(v)))) {
                                PyErr_SetString(PyExc_ValueError, "v is not an appropriate 1D list");
                                return -1;
                            }
                            
                            if (!PyLong_Check(PyList_GetItem(v,(Py_ssize_t) i)) && !PyFloat_Check(PyList_GetItem(v,(Py_ssize_t) i))) {
                                PyErr_SetString(PyExc_ValueError, "Elements of v are not floats or ints");
                                return -1;
                            }
                        }

                        // set slice_matrix
                        Matrix61c* val = (Matrix61c*) Matrix61c_subscript(self, key);
                        //LOGIC: set 1D matrix val to 1D list v

                        if ((int)PyList_Size(v) != slice_length2) {
                                PyErr_SetString(PyExc_ValueError, "v has incorrect length");
                                return -1;
                        }
                        for (int c = 0; c<val->mat->cols; c++) {
                            // printf("val->mat->data[0][c] %f \n", val->mat->data[0][c]);
                            val->mat->data[0][c] = PyLong_AsLong(PyList_GetItem(v,c));//directly modifies the original matrix
                        }
                    }
                }
                //if tuple is (int, int)
                else if (PyLong_Check(slice1) && PyLong_Check(slice2)){
                    //integer 1
                    int slice_int_1 = (int) PyLong_AsLong(slice1);
                    if (slice_int_1 >= self->mat->rows || slice_int_1 < 0){
                        PyErr_SetString(PyExc_IndexError, "Index out of range!");
                        return -1;
                    }

                    //integer 2
                    int slice_int_2 = (int) PyLong_AsLong(slice2);
                    if (slice_int_2 >= self->mat->cols || slice_int_2 < 0){
                        PyErr_SetString(PyExc_IndexError, "Index out of range!");
                        return -1;
                    }
                    
                    if (PyLong_Check(v) || PyFloat_Check(v)) {
                        self->mat->data[slice_int_1][slice_int_2] = PyLong_AsLong(v);
                        if (self->mat->parent != NULL) {
                            self->mat->parent->data[slice_int_1][slice_int_2] = PyLong_AsLong(v);
                        }
                    } else {//TypeError: Slice is 1x1, but v is not an int/float
                        PyErr_SetString(PyExc_TypeError, "V is not a int or float for 1x1 result slice");
                        return -1;
                    }
                }
                else {
                PyErr_SetString(PyExc_TypeError, "Key is not valid");
                return -1;
                }

            } 
            //else invalid key
            else {
                PyErr_SetString(PyExc_TypeError, "Key is not valid");
                return -1;
            }
        }

    } else if (self->mat->rows == 1 || self->mat->cols == 1) { // 1D ORIGINAL MATRIX
        if (PyLong_Check(key)) {
            if (!PyLong_Check(v)){ //TypeError: slice is 1x1 but v is not a float/int
                PyErr_SetString(PyExc_TypeError, "Resulting slice is 1 by 1, but v is not a float or int");
                return -1;
            }

                //IndexOutOfRangeCheck                
                int slice_int = (int) PyLong_AsLong(key);
                if (slice_int < 0 || slice_int >= self->mat->rows) {
                    PyErr_SetString(PyExc_IndexError, "Index out of range");
                    return -1;
                }

                //LOGIC: set val row to v
                if (self->mat->rows == 1) {
                    self->mat->data[0][slice_int] = PyLong_AsLong(v);
                    if (self->mat->parent != NULL) {
                        self->mat->parent->data[0][slice_int] = PyLong_AsLong(v);
                    }
                } 
                else if (self->mat->cols == 1) {
                    self->mat->data[slice_int][0] = PyLong_AsLong(v);
                    if (self->mat->parent != NULL) {
                        self->mat->parent->data[slice_int][0] = PyLong_AsLong(v);
                    }
                } 

        //if v is 1D
        } else if (PySlice_Check(key)) {

            Py_ssize_t start;
            Py_ssize_t stop;
            Py_ssize_t step;
            Py_ssize_t slice_length;
            //if mat->rows == 1, slice column
            if (self->mat->rows == 1){
                int a =  PySlice_GetIndicesEx(key,self->mat->cols, &start, &stop, &step, &slice_length);
                if (a < 0){
                    PyErr_SetString(PyExc_ValueError, "Slice info not valid!");
                    return -1;
                }
            }
            //if mat->cols == 1, slice row
            else if (self->mat->cols == 1){
                int a =  PySlice_GetIndicesEx(key,self->mat->rows, &start, &stop, &step, &slice_length);
                if (a < 0){
                    PyErr_SetString(PyExc_ValueError, "Slice info not valid!");
                    return -1;
                }
            }

            if ((int) step != 1) {
                PyErr_SetString(PyExc_ValueError, "Slice info not valid!");
                return -1;
            }

            if ((int) slice_length < 1) {
                PyErr_SetString(PyExc_ValueError, "Slice info not valid!");
                return -1;
                
            } 
            //1. if resulting matrix slice is 1x1
            else if ((int) slice_length == 1) { 
                
                if (PyLong_Check(v) || PyFloat_Check(v)) {
                    if (self->mat->rows == 1) {
                        self->mat->data[0][start] = PyLong_AsLong(v);
                        if (self->mat->parent != NULL) {
                            self->mat->parent->data[0][start] = PyLong_AsLong(v);
                        }
                    } else if (self->mat->cols == 1) {
                        self->mat->data[start][0] = PyLong_AsLong(v);
                        if (self->mat->parent != NULL) {
                            self->mat->parent->data[start][0] = PyLong_AsLong(v);
                        }
                    }
                } else {
                    PyErr_SetString(PyExc_TypeError, "V is not a int or float for 1x1 result slice");
                    return -1;
                }

            //2. if mat->cols == 1
            } else if ((int) slice_length > 1 && slice_length <= self->mat->rows && (self->mat->cols == 1)) { //nx1 resulting matrix
                //TypeError
                if (!PyList_Check(v)) {
                    PyErr_SetString(PyExc_TypeError, "Resulting slice is not 1 by 1, but v is not a list.");
                    return -1;
                }
                //1xn case - v error checks
                for(int i = 0; i < PyList_Size(v); i++) { //check if values inside v are not float/int
                    if (((PyList_Check(PyList_GetItem(v,(Py_ssize_t) i)) || !PyList_Check(v)))) {
                        PyErr_SetString(PyExc_ValueError, "v is not an appropriate 1D list");
                        return -1;
                    }
                    
                    if (!PyLong_Check(PyList_GetItem(v,(Py_ssize_t) i)) && !PyFloat_Check(PyList_GetItem(v,(Py_ssize_t) i))) {
                        PyErr_SetString(PyExc_ValueError, "Elements of v are not floats or ints");
                        return -1;
                    }
                }
                // set slice_matrix
                Matrix61c* val = (Matrix61c*) Matrix61c_subscript(self, key);
                //LOGIC: set 1D matrix val to 1D list v

                if ((int)PyList_Size(v) != slice_length) {
                        PyErr_SetString(PyExc_ValueError, "v has incorrect length");
                        return -1;
                }
                for (int r = 0; r<val->mat->rows; r++) {
                    val->mat->data[r][0] = PyLong_AsLong(PyList_GetItem(v,r));//directly modifies the original matrix
                }


            //3. if mat->rows == 1
            } else if ((int) slice_length > 1 && slice_length <= self->mat->cols && (self->mat->rows == 1)) { //1xn resulting matrix
                //TypeError
                if (!PyList_Check(v)) {
                        PyErr_SetString(PyExc_TypeError, "Resulting slice is not 1 by 1, but v is not a list.");
                        return -1;
                    }
                    //1xn case - v error checks
                    for(int i = 0; i < PyList_Size(v); i++) { //check if values inside v are not float/int
                        if (((PyList_Check(PyList_GetItem(v,(Py_ssize_t) i)) || !PyList_Check(v)))) {
                            PyErr_SetString(PyExc_ValueError, "v is not an appropriate 1D list");
                            return -1;
                        }
                        
                        if (!PyLong_Check(PyList_GetItem(v,(Py_ssize_t) i)) && !PyFloat_Check(PyList_GetItem(v,(Py_ssize_t) i))) {
                            PyErr_SetString(PyExc_ValueError, "Elements of v are not floats or ints");
                            return -1;
                        }
                    }
                    // set slice_matrix
                    Matrix61c* val = (Matrix61c*) Matrix61c_subscript(self, key);
                    //LOGIC: set 1D matrix val to 1D list v

                    if ((int)PyList_Size(v) != slice_length) {
                            PyErr_SetString(PyExc_ValueError, "v has incorrect length");
                            return -1;
                    }
                    for (int c = 0; c<val->mat->cols; c++) {
                        val->mat->data[0][c] = PyLong_AsLong(PyList_GetItem(v,c));//directly modifies the original matrix
                    }
            //4. if mat->rows != 1 and mat->cols != 1 (shouldn't happen)
            } else {
                PyErr_SetString(PyExc_ValueError, "Slice info not valid!");
                return -1;
            } 
        
        } 
        else {
            PyErr_SetString(PyExc_ValueError, "Slice info not valid!");
            return -1;
        }
    //self->mat is not 1D or 2D
    } else {
        PyErr_SetString(PyExc_RuntimeError, "Operation can only support 1D and 2D matrices");
        return -1;
    }
    return 0;
}

PyMappingMethods Matrix61c_mapping = {
    NULL,
    (binaryfunc) Matrix61c_subscript,
    (objobjargproc) Matrix61c_set_subscript,
};

/* INSTANCE ATTRIBUTES*/
PyMemberDef Matrix61c_members[] = {
    {
        "shape", T_OBJECT_EX, offsetof(Matrix61c, shape), 0,
        "(rows, cols)"
    },
    {NULL}  /* Sentinel */
};

PyTypeObject Matrix61cType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name = "numc.Matrix",
    .tp_basicsize = sizeof(Matrix61c),
    .tp_dealloc = (destructor)Matrix61c_dealloc,
    .tp_repr = (reprfunc)Matrix61c_repr,
    .tp_as_number = &Matrix61c_as_number,
    .tp_flags = Py_TPFLAGS_DEFAULT |
    Py_TPFLAGS_BASETYPE,
    .tp_doc = "numc.Matrix objects",
    .tp_methods = Matrix61c_methods,
    .tp_members = Matrix61c_members,
    .tp_as_mapping = &Matrix61c_mapping,
    .tp_init = (initproc)Matrix61c_init,
    .tp_new = Matrix61c_new
};


struct PyModuleDef numcmodule = {
    PyModuleDef_HEAD_INIT,
    "numc",
    "Numc matrix operations",
    -1,
    Matrix61c_class_methods
};

/* Initialize the numc module */
PyMODINIT_FUNC PyInit_numc(void) {
    PyObject* m;

    if (PyType_Ready(&Matrix61cType) < 0)
        return NULL;

    m = PyModule_Create(&numcmodule);
    if (m == NULL)
        return NULL;

    Py_INCREF(&Matrix61cType);
    PyModule_AddObject(m, "Matrix", (PyObject *)&Matrix61cType);
    printf("CS61C Fall 2020 Project 4: numc imported!\n");
    fflush(stdout);
    return m;
}