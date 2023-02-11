#include "matrix.h"
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

// Include SSE intrinsics
#if defined(_MSC_VER)
#include <intrin.h>
#elif defined(__GNUC__) && (defined(__x86_64__) || defined(__i386__))
#include <immintrin.h>
#include <x86intrin.h>
#endif

/*
 * Generates a random double between `low` and `high`.
 */
double rand_double(double low, double high) {
    double range = (high - low);
    double div = RAND_MAX / range;
    return low + (rand() / div);
}

/*
 * Generates a random matrix with `seed`.
 */
void rand_matrix(matrix *result, unsigned int seed, double low, double high) {
    srand(seed);
    for (int i = 0; i < result->rows; i++) {
        for (int j = 0; j < result->cols; j++) {
            set(result, i, j, rand_double(low, high));
        }
    }
}

/*
 * Allocate space for a matrix struct pointed to by the double pointer mat with
 * `rows` rows and `cols` columns.
 */
int allocate_matrix(matrix **mat, int rows, int cols) {
    if (rows <= 0 || cols <= 0){ 
        return -1;
    }
    if (mat == NULL){
        return -1;
    }

    *mat = malloc(sizeof(matrix));
    (*mat)->rows = rows;
    (*mat)->cols = cols;
    (*mat)->data = malloc(sizeof(double*) * rows); /* allocates space for rows */
    if ((*mat)->data == NULL) {
        free(mat);
        return -1;
    }
    //single calloc: 
    (*mat)->data[0] = calloc((rows * cols), sizeof(double));
    if ((*mat)->data[0] == NULL){
        free((*mat)->data);
        free(mat);
        return -1;
    }

    //single calloc add here: IMPROVEMENT 
    for(int i = 0; i < rows; i++){
        (*mat)->data[i] = (*mat)->data[0] + cols*i;
    }

    if (rows == 1 || cols == 1){
        (*mat)->is_1d = 1;
    } else {
        (*mat)->is_1d = 0;
    }
    
    (*mat)->ref_cnt = 1;
    (*mat)->parent = NULL; //set parent pointer to NULL
    
    return 0;

}


/*
 * Allocate space for a matrix struct pointed to by `mat` with `rows` rows and `cols` columns.
 */

int allocate_matrix_ref(matrix **mat, matrix *from, int row_offset, int col_offset,
                        int rows, int cols) {
    // error if rows & cols are invalid
    if (rows <= 0 || cols <= 0){ 
        return -1;
    }
    // error if from is valid
    if (from == NULL){
        return -1;
    }

    *mat = malloc(sizeof(matrix)); //allocates space for matrix
    (*mat)->rows = rows;
    (*mat)->cols = cols; 
    (*mat)->data = malloc(sizeof(double*) * rows); /* allocates space for rows */

    // Path compression: if from has a parent, set mat's parent to from's parent
    if (from->parent != NULL){
        (*mat)->parent = from->parent;
    } else {
        (*mat)->parent = from;
    }

    (*mat)->ref_cnt = (from->ref_cnt) + 1;
    from->ref_cnt += 1;
    if (rows == 1 || cols == 1){
        (*mat)->is_1d = 1;
    } else {
        (*mat)->is_1d = 0;
    }
   
    // sets mat->data to from->data's row_offset & col_offset
    for (int r = 0; r<rows; r++){
        (*mat)->data[r] = &(from->data[r + row_offset][col_offset]);
    }
    

    return 0;    
}

/*
 * This function will be called by Python when a numc matrix loses all of its
 * reference pointers.
 */
void deallocate_matrix(matrix *mat) {
    if (mat == NULL){
        return;
    }
    // decrement ref_cnt
    mat->ref_cnt -= 1;

    // if ref_cnt is 0 and no parent, free mat
    if (mat->ref_cnt == 0 && mat->parent == NULL){
        free(mat->data[0]);
        free(mat->data);
        free(mat);
    // otherwise if parent exists
    } else if (mat->parent != NULL) {
        // if parent's ref_cnt is 1, dellocate the parent
        if (mat->parent->ref_cnt == 1) {
            deallocate_matrix(mat->parent); //this is the case when mat->parent = mat
        } else {
            // otherwise decrement the parent's ref_count
            mat->parent->ref_cnt -= 1;
        }
    }
}

/*
 * Return the double value of the matrix at the given row and column.
 * You may assume `row` and `col` are valid.
 */
double get(matrix *mat, int row, int col) {
    return mat->data[row][col];
}

/*
 * Set the value at the given row and column to val. You may assume `row` and
 * `col` are valid
 */
void set(matrix *mat, int row, int col, double val) {
    mat->data[row][col] = val;
}

/*
 * Set all entries in mat to val
 */
void fill_matrix(matrix *mat, double val) {
    // pragma only if rows or cols > 2000
    if (mat->rows > 2000 || mat->cols > 2000) {
        const int NUM_THREADS = 8;
        omp_set_num_threads(NUM_THREADS);

        #pragma omp parallel for    
            for(int i=0; i<mat->rows; i++){
                for(int j=0; j<mat->cols; j++){
                    mat->data[i][j] = val;
                }
            }
    } else {
        for(int i=0; i<mat->rows; i++){
            for(int j=0; j<mat->cols; j++){
                mat->data[i][j] = val;
            }
        }
    }
}

/*
 * Store the result of adding mat1 and mat2 to `result`.
 * Return 0 upon success and a nonzero value upon failure.
 */

int add_matrix(matrix *result, matrix *mat1, matrix *mat2) {
    if((mat1->rows != mat2->rows)||(mat1->cols != mat2->cols)){
        return 1;
    } 

    const int NUM_THREADS = 8;
    omp_set_num_threads(NUM_THREADS);

    #pragma omp parallel for

    for (int i = 0; i < ((mat1->rows*mat1->cols)/16)*16 ; i+=16){
        
        double *block_start_mat1 = mat1->data[0] + i;
        double *block_start_mat2 = mat2->data[0] + i;
        double *block_start_result = result->data[0] + i;
        __m256d data1 = _mm256_loadu_pd(block_start_mat1);
        __m256d data2 = _mm256_loadu_pd(block_start_mat2);
        __m256d sum = _mm256_add_pd(data1, data2);
        _mm256_storeu_pd(block_start_result, sum);

        block_start_mat1 = mat1->data[0] + i + 4;
        block_start_mat2 = mat2->data[0] + i + 4;
        block_start_result = result->data[0] + i + 4;
        data1 = _mm256_loadu_pd(block_start_mat1);
        data2 = _mm256_loadu_pd(block_start_mat2);
        sum = _mm256_add_pd(data1, data2);
        _mm256_storeu_pd(block_start_result, sum);

        block_start_mat1 = mat1->data[0] + i + 8;
        block_start_mat2 = mat2->data[0] + i + 8;
        block_start_result = result->data[0] + i + 8;
        data1 = _mm256_loadu_pd(block_start_mat1); 
        data2 = _mm256_loadu_pd(block_start_mat2);
        sum = _mm256_add_pd(data1, data2);
        _mm256_storeu_pd(block_start_result, sum);
        
        block_start_mat1 = mat1->data[0] + i + 12;
        block_start_mat2 = mat2->data[0] + i + 12;
        block_start_result = result->data[0] + i + 12;
        data1 = _mm256_loadu_pd(block_start_mat1);
        data2 = _mm256_loadu_pd(block_start_mat2);
        sum = _mm256_add_pd(data1, data2);
        _mm256_storeu_pd(block_start_result, sum);
    }

    //tail case: remaining elements 
    for (int i=((mat1->rows*mat1->cols)/16)*16 ; i<mat1->rows*mat1->cols; i++){
        result->data[0][i] = mat1->data[0][i] + mat2->data[0][i];
    }

    return 0;
}

/*
 * Store the result of subtracting mat2 from mat1 to `result`. 
 * Return 0 upon success and a nonzero value upon failure.
 */
int sub_matrix(matrix *result, matrix *mat1, matrix *mat2) {
    if((mat1->rows != mat2->rows)||(mat1->cols != mat2->cols)){
        return 1;
    } 
    const int NUM_THREADS = 8;
    omp_set_num_threads(NUM_THREADS);

    #pragma omp parallel for

    for (int i = 0; i < ((mat1->rows*mat1->cols)/16)*16 ; i+=16){
    
        double *block_start_mat1 = mat1->data[0] + i;
        double *block_start_mat2 = mat2->data[0] + i;
        double *block_start_result = result->data[0] + i;
        __m256d data1 = _mm256_loadu_pd(block_start_mat1);
        __m256d data2 = _mm256_loadu_pd(block_start_mat2);
        __m256d sum = _mm256_sub_pd(data1, data2);
        _mm256_storeu_pd(block_start_result, sum);

        block_start_mat1 = mat1->data[0] + i + 4;
        block_start_mat2 = mat2->data[0] + i + 4;
        block_start_result = result->data[0] + i + 4;
        data1 = _mm256_loadu_pd(block_start_mat1);
        data2 = _mm256_loadu_pd(block_start_mat2);
        sum = _mm256_sub_pd(data1, data2);
        _mm256_storeu_pd(block_start_result, sum);

        block_start_mat1 = mat1->data[0] + i + 8;
        block_start_mat2 = mat2->data[0] + i + 8;
        block_start_result = result->data[0] + i + 8;
        data1 = _mm256_loadu_pd(block_start_mat1); 
        data2 = _mm256_loadu_pd(block_start_mat2);
        sum = _mm256_sub_pd(data1, data2);
        _mm256_storeu_pd(block_start_result, sum);
        
        block_start_mat1 = mat1->data[0] + i + 12;
        block_start_mat2 = mat2->data[0] + i + 12;
        block_start_result = result->data[0] + i + 12;
        data1 = _mm256_loadu_pd(block_start_mat1);
        data2 = _mm256_loadu_pd(block_start_mat2);
        sum = _mm256_sub_pd(data1, data2);
        _mm256_storeu_pd(block_start_result, sum);
    }

    // tail case: remaining elements 
    for (int i=((mat1->rows*mat1->cols)/16)*16 ; i<mat1->rows*mat1->cols; i++){
        result->data[0][i] = mat1->data[0][i] - mat2->data[0][i];
    }

    return 0;
}

/*
 * Store the result of multiplying mat1 and mat2 to `result`.
 * Return 0 upon success and a nonzero value upon failure.
 */

int mul_matrix(matrix *result, matrix *mat1, matrix *mat2) {
    if((mat1->cols != mat2->rows)){
        return 1;
    } 

    
    const int NUM_THREADS = 8; 
    omp_set_num_threads(NUM_THREADS);

    // RESET RESULT TO 0
    memset(result->data[0], 0, (result->rows*result->cols*sizeof(double)));

    #pragma omp parallel for
        for(int i=0; i<mat1->rows; i+=1){
            for(int k=0; k<(mat1->cols)/4*4; k+=4){
                __m256d load_val_m1_0 = _mm256_set1_pd(mat1->data[i][k]); //mat1[i][k]
                __m256d load_val_m1_1 = _mm256_set1_pd(mat1->data[i][k+1]); //mat1[i][k+1]
                __m256d load_val_m1_2 = _mm256_set1_pd(mat1->data[i][k+2]); //mat1[i][k+2]
                __m256d load_val_m1_3 = _mm256_set1_pd(mat1->data[i][k+3]); //mat1[i][k+3]


                for(int j=0; j<mat2->cols/4*4; j+=4) {
                    //j+0
                    double *block_start_mat2 = mat2->data[0] + k * mat2->cols + j; 
                    double *block_start_result = result->data[0] + i * mat2->cols + j; //data[i][j]

                    __m256d load_val_m2 = _mm256_loadu_pd(block_start_mat2); //mat2[k][j]
                    __m256d result_load = _mm256_loadu_pd(block_start_result);
                    result_load = _mm256_fmadd_pd(load_val_m1_0, load_val_m2, result_load);

                    // //j+1
                    block_start_mat2 = mat2->data[0] + (k+1)  * mat2->cols + j; 
                    block_start_result = result->data[0] + i * mat2->cols + j; //data[i][j]
                    
                    load_val_m2 = _mm256_loadu_pd(block_start_mat2); //mat2[k][j
                    result_load = _mm256_fmadd_pd(load_val_m1_1, load_val_m2, result_load);
                    
                    // //j+2
                    block_start_mat2 = mat2->data[0] + (k+2) * mat2->cols + j; 
                    block_start_result = result->data[0] + i * mat2->cols + j; //data[i][j]
                    
                    load_val_m2 = _mm256_loadu_pd(block_start_mat2); //mat2[k][j]
                    result_load = _mm256_fmadd_pd(load_val_m1_2, load_val_m2, result_load);

                    // //j+3
                    block_start_mat2 = mat2->data[0] + (k+3) * mat2->cols + j; 
                    block_start_result = result->data[0] + i * mat2->cols + j; //data[i][j]
                    
                    load_val_m2 = _mm256_loadu_pd(block_start_mat2); //mat2[k][j]
                    result_load = _mm256_fmadd_pd(load_val_m1_3, load_val_m2, result_load);
                   
                    _mm256_storeu_pd(block_start_result, result_load);
                }
                
            
                //tail case for j
                for(int j=(mat2->cols/4*4); j<(mat2->cols); j++){
                    result->data[i][j] += mat1->data[i][k]*mat2->data[k][j];
                    result->data[i][j] += mat1->data[i][k+1]*mat2->data[k+1][j];
                    result->data[i][j] += mat1->data[i][k+2]*mat2->data[k+2][j];
                    result->data[i][j] += mat1->data[i][k+3]*mat2->data[k+3][j];
                }
            }
            // tail case for k
            for (int k = mat1->cols/4*4; k <(mat1->cols); k++){ 
                for (int j = 0; j< mat2->cols; j++){
                    result->data[i][j] += mat1->data[i][k]*mat2->data[k][j];
                }
            }

        }
    return 0; 
}

int pow_matrix(matrix *result, matrix *mat, int pow) {

    if((mat->cols != mat->rows) || pow < 0) {
        return 1;
    }

    if(mat->cols!=result->cols || mat->rows!=result->rows){
        return 1;
    }
    
    matrix *output = NULL;
    allocate_matrix(&output, mat->rows, mat->cols);

    matrix *input = NULL;
    allocate_matrix(&input, mat->rows, mat->cols);
    
    memcpy(input->data[0], mat->data[0], (input->rows*input->cols*sizeof(double)));
    memset(result->data[0], 0, (result->rows*result->cols*sizeof(double)));

    for (int i = 0; i<mat->rows ; i++){ 
        result->data[0][i*mat->rows + i] = 1;
    }   

    // 2. repeated squaring
    while (pow > 0){
        while ((pow & 1) == 0){
            pow = pow / 2;   
            mul_matrix(output, input, input);
            memcpy(input->data[0], output->data[0], (input->rows*input->cols*sizeof(double)));
        }
        pow = pow - 1;
        mul_matrix(output, input, result);//
        
        memcpy(result->data[0], output->data[0], (input->rows*input->cols*sizeof(double)));
    }

    return 0;
}

// SIMD AND OMP negative
int neg_matrix(matrix *result, matrix *mat) {
    if (mat == NULL) {
        return 1;
    }

    const int NUM_THREADS = 8;
    omp_set_num_threads(NUM_THREADS);

    #pragma omp parallel for
    for (int i = 0; i < ((mat->rows*mat->cols)/16)*16 ; i+=16){//+=4 before. /12*12 instead 
        double *block_start_mat = mat->data[0] + i;
        double *block_start_result = result->data[0] + i;
        __m256d data = _mm256_loadu_pd(block_start_mat);
        __m256d negation = _mm256_xor_pd(data, _mm256_set1_pd(-0.0));
        _mm256_storeu_pd(block_start_result, negation);

        block_start_mat = mat->data[0] + i + 4;
        block_start_result = result->data[0] + i + 4;
        data = _mm256_loadu_pd(block_start_mat);
        negation = _mm256_xor_pd(data, _mm256_set1_pd(-0.0));
        _mm256_storeu_pd(block_start_result, negation);

        block_start_mat = mat->data[0] + i + 8;
        block_start_result = result->data[0] + i + 8;
        data = _mm256_loadu_pd(block_start_mat);
        negation = _mm256_xor_pd(data, _mm256_set1_pd(-0.0));
        _mm256_storeu_pd(block_start_result, negation);

        block_start_mat = mat->data[0] + i + 12;
        block_start_result = result->data[0] + i + 12;
        data = _mm256_loadu_pd(block_start_mat);
        negation = _mm256_xor_pd(data, _mm256_set1_pd(-0.0));
        _mm256_storeu_pd(block_start_result, negation);
    }

    // tail case: remaining elements 
    for (int i=((mat->rows*mat->cols)/16)*16 ; i<mat->rows*mat->cols; i++){
        result->data[0][i] = -(mat->data[0][i]);
    }

    return 0;
}


int neg_matrix_naive(matrix *result, matrix *mat) {
    for(int i=0; i<mat->rows; i++){
        for(int j=0; j<mat->cols; j++){
            double val = -(mat->data[i][j]);
            set(result, i, j, val);
        }
    }
    return 0;
}

// SIMD AND OMP absolute value
int abs_matrix(matrix *result, matrix *mat) {    
    if (mat == NULL) {
        return 1;
    }

    const int NUM_THREADS = 8;
    omp_set_num_threads(NUM_THREADS);

    #pragma omp parallel for
        for (int i = 0; i < ((mat->rows*mat->cols)/16)*16 ; i+=16){//+=4 before. /12*12 instead 
            double *block_start_mat = mat->data[0] + i;
            double *block_start_result = result->data[0] + i;
            __m256d data = _mm256_loadu_pd(block_start_mat);
            __m256d abs_val = _mm256_andnot_pd(_mm256_set1_pd(-0.0), data);
            _mm256_storeu_pd(block_start_result, abs_val);

            block_start_mat = mat->data[0] + i + 4;
            block_start_result = result->data[0] + i + 4;
            data = _mm256_loadu_pd(block_start_mat);
            abs_val = _mm256_andnot_pd(_mm256_set1_pd(-0.0), data);
            _mm256_storeu_pd(block_start_result, abs_val);

            block_start_mat = mat->data[0] + i + 8;
            block_start_result = result->data[0] + i + 8;
            data = _mm256_loadu_pd(block_start_mat);
            abs_val = _mm256_andnot_pd(_mm256_set1_pd(-0.0), data);
            _mm256_storeu_pd(block_start_result, abs_val);

            block_start_mat = mat->data[0] + i + 12;
            block_start_result = result->data[0] + i + 12;
            data = _mm256_loadu_pd(block_start_mat);
            abs_val = _mm256_andnot_pd(_mm256_set1_pd(-0.0), data);
            _mm256_storeu_pd(block_start_result, abs_val);
        }

    // tail case: remaining elements 
    for (int i=((mat->rows*mat->cols)/16)*16 ; i<mat->rows*mat->cols; i++){
        if ((mat->data[0][i]) < 0) {
            result->data[0][i] = -(mat->data[0][i]);
        } else {
            result->data[0][i] = (mat->data[0][i]);
        }
    }

    return 0;
}


/*
 * Store the result of taking the absolute value element-wise to `result`.
 * Return 0 upon success and a nonzero value upon failure.
 */
int abs_matrix_naive(matrix *result, matrix *mat) {    
     for(int i=0; i<mat->rows; i++){
        for(int j=0; j<mat->cols; j++){
            double val = mat->data[i][j];
            if (val<0){
                set(result, i, j, -val);
            }
            else {
                set(result, i, j, val);
            }
        }
    }
    return 0;
}
