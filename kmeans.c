#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include <math.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>

// structs
typedef struct point point;
typedef struct point_list_node point_list_node;

// constants
#define INPUT_ERR -1
#define GENERAL_ERR -2
#define SUCCESS 0

// functions defenition
point* malloc_point(int);
void free_point(point*);
point_list_node* malloc_point_list_node(int dim);
void free_point_list_node(point_list_node*);
int parse_one_point(PyObject*, point*);
int parse_data_points(PyObject*, point_list_node**);
int parse_centroids(PyObject*, point***);
double calculate_distance(point*, point*);
void find_nearest_centroid(point**, int, point*);
void find_all_nearest_centroids(point**, int, point_list_node*);
void count_group_sizes(point_list_node*, int*);
int calc_new_centroids(point_list_node*, int, point***);
double calc_max_centroid_distance(point**, point**, int);
void free_centroids(point**, int);
int kmeans(point_list_node*, int, int, double, point***);
int output_centroids(PyObject*, point**, int);
int add_centroid_to_list(PyObject*, point*, int);
void free_resources(point**, point_list_node*, int);
static PyObject* kmeans_to_py(PyObject*, PyObject*);


// python module settings
static PyMethodDef KmeansMethods[] = {
    {"fit",  (PyCFunction)kmeans_to_py, METH_VARARGS,
     PyDoc_STR("Runs the kmeans algorithm.")},
    {NULL, NULL, 0, NULL}
};


static struct PyModuleDef mykmeanssp = {
    PyModuleDef_HEAD_INIT,
    "mykmeanssp",
    NULL, 
    -1,
    KmeansMethods
};


// structs
struct point{
    double *vector;
    int dim;
    int centroid_index;
};


struct point_list_node {
    point *value;
    point_list_node *next;
};


// initialize the module
PyMODINIT_FUNC PyInit_mykmeanssp(void)
{
    return PyModule_Create(&mykmeanssp);
}


// the function that python runs
static PyObject* kmeans_to_py(PyObject *self, PyObject *args)
{
    double eps;
    long max_iter;
    PyObject* data_points;
    PyObject* init_centroids;
    int k;
    int result;
    PyObject* final_centroids;

    // parse arguments from python
    if (!PyArg_ParseTuple(args, "OOdl", &data_points, &init_centroids, &eps, &max_iter)) {
        return NULL;
    }

    k = PyObject_Length(init_centroids); // number of centroids
    point_list_node *points = NULL;
    point** centroids = NULL;

    result = parse_data_points(data_points, &points);
    if (result != SUCCESS) {
        free_resources(centroids, points, k);
        Py_RETURN_NONE;
    }

    result = parse_centroids(init_centroids, &centroids);
    if (result != SUCCESS) {
        free_resources(centroids, points, k);
        Py_RETURN_NONE;
    }

    result = kmeans(points, k, max_iter, eps, &centroids);
    if (result != SUCCESS) {
        free_resources(centroids, points, k);
        Py_RETURN_NONE;
    }

    // creates list to return to python
    final_centroids = PyList_New(k);
    if (final_centroids == NULL) {
        free_resources(centroids, points, k);
        Py_RETURN_NONE;
    }
    result = output_centroids(final_centroids, centroids, k);
    if (result != SUCCESS) {
        free_resources(centroids, points, k);
        Py_RETURN_NONE;
    }

    free_resources(centroids, points, k);
    return Py_BuildValue("O", final_centroids);
}

// allocates memory for one point with dim dimentions
// if an error occures, returns NULL
point* malloc_point(int dim) {
    point* p = (point*)malloc(sizeof(point));
    if (p != NULL) {
        p->dim = dim;
        p->centroid_index = -1;
        p->vector = (double*)malloc(sizeof(double)*dim);
        if (p->vector != NULL) {
            return p;
        }
        free(p);
        p = NULL;
    }
    return p;
}

// frees point
void free_point(point* p) {
    if (p == NULL){
        return;
    }
    free(p->vector);
    free(p);
}

// allocates memory for one point list node, including the point itself
// if an error occures, returns NULL
point_list_node* malloc_point_list_node(int dim) {
    point_list_node* pln = (point_list_node*)malloc(sizeof(point_list_node));
    if (pln != NULL) {
        pln->next = NULL;
        pln->value = malloc_point(dim);
        if (pln->value != NULL) {
            return pln;
        }
        free(pln);
        pln = NULL;
    }
    return pln;
}

// frees point list node struct
void free_point_list_node(point_list_node *points) {
    point_list_node *next;
    while (points != NULL) {
        next = points->next;
        free_point(points->value);
        free(points);
        points = next;
    }
}

// point_data is python list with the coordination of a point
// this function parse the python list to the c struct point
int parse_one_point(PyObject* point_data, point* p) {
    int i;
    PyObject *item;
    int dim = PyObject_Length(point_data);
    for(i = 0; i < dim ; i++) {
        item = PyList_GetItem(point_data, i);
        if (!PyFloat_Check(item))
            return GENERAL_ERR;
        p->vector[i] = PyFloat_AsDouble(item);
    }
    return SUCCESS;
}

// parses all of the data points from python
// points is a python list of python lists. each list is one point.
// at the end, head is a pointer to a linked list with all of the points.
int parse_data_points(PyObject* points, point_list_node** head){
    int i;
    int dim;
    int result;
    PyObject* my_point;
    point_list_node *prev = NULL;
    point_list_node *next = NULL;
    int points_number = PyObject_Length(points);
    if (points_number <= 0){
        return GENERAL_ERR;
    }

    PyObject* first_point = PyList_GetItem(points, 0);
    if (first_point == NULL) {
        return GENERAL_ERR;
    }
    dim = PyObject_Length(first_point);
    *head = malloc_point_list_node(dim);
    if (*head == NULL) {
        return GENERAL_ERR;
    }
    next = *head;

    // in each iteration we parse one point and put it on the list
    for(i=0; i<points_number; i++) {
        my_point = PyList_GetItem(points, i);
        if (my_point == NULL){
            return GENERAL_ERR;
        }
        result = parse_one_point(my_point, next->value);
        if (result != SUCCESS) {
            return result;
        }
        next->next = malloc_point_list_node(dim);
        prev = next;
        next = next->next;
        if (next == NULL) {
            return GENERAL_ERR;
        }
    }
    prev->next = NULL;
    free_point_list_node(next);
    return SUCCESS;
}

// parses all of the centroids
// returns k centroids in the array of points centroids.
int parse_centroids(PyObject* init_centroids, point*** centroids){
    int result;
    int i;
    int dim;

    PyObject* my_centroid;
    int k = PyObject_Length(init_centroids);
    if (k <= 0) {
        return GENERAL_ERR;
    }
    *centroids = (point**)malloc(sizeof(point*)*k);
    if (*centroids == NULL) {
        return GENERAL_ERR;
    }

    PyObject* first_point = PyList_GetItem(init_centroids, 0);
    if (first_point == NULL) {
        return GENERAL_ERR;
    }
    dim = PyObject_Length(first_point);

    // in each iteration we parse one centroid and put it in the array
    for (i=0; i<k; i++) {
        (*centroids)[i] = malloc_point(dim);
        my_centroid = PyList_GetItem(init_centroids, i);
        if (my_centroid == NULL){
            return GENERAL_ERR;
        }
        result = parse_one_point(my_centroid, (*centroids)[i]);
        if (result != SUCCESS){
            return GENERAL_ERR;
        }
    }
    return SUCCESS;
}

// calculates distance between p1 and p2
double calculate_distance(point* p1, point* p2) {
    double distance = 0;
    int i;
    for(i = 0; i < p1->dim; i++) {
        distance += pow(p1->vector[i] - p2->vector[i], 2);
    }
    return sqrt(distance);
}

// set p->centroid_index to the index of the centroid which is closest to p
void find_nearest_centroid(point* centroids[], int k, point* p) {
    double min_dist = -1;
    double distance;
    int i;
    for(i=0; i<k; i++) {
        distance = calculate_distance(centroids[i], p);
        if(distance < min_dist || min_dist == -1) {
            p->centroid_index = i;
            min_dist = distance;
        }
    }
}

// scanning the list points and set the centroid_index of all of the points
void find_all_nearest_centroids(point* centroids[], int k, point_list_node* points) {
    while(points != NULL){
        find_nearest_centroid(centroids, k, points->value);
        points = points->next;
    }
}

// count the sizes of the groups of the centroids
// - the number of points which are closest to each centroid.
void count_group_sizes(point_list_node* points, int* sizes) {
    while (points!=NULL){
        sizes[points->value->centroid_index]++;
        points = points->next;
    }
}

// calculates the new centroids from the groups of the data points
// returns the new centroids in the array centroids
int calc_new_centroids(point_list_node* points, int k, point*** centroids) {
    int i;
    int dim;
    int d;
    point* p;
    int *sizes = (int*)malloc(sizeof(int)*k);
    if (sizes == NULL) {
        return GENERAL_ERR;
    }
    for (i=0; i<k; i++){
        sizes[i] = 0;
    }
    *centroids = (point**)malloc(sizeof(point*)*k);
    if (*centroids == NULL) {
        free(sizes);
        return GENERAL_ERR;
    }

    dim = points->value->dim;
    for(i = 0; i<k; i++) {
        (*centroids)[i] = malloc_point(dim);
        if ((*centroids)[i] == NULL) {
            free_centroids(*centroids, i);
            free(sizes);
            return GENERAL_ERR;
        }
    }
    // initialize memory
    for(i=0; i<k; i++){
        for(d=0; d<dim; d++) {
            ((*centroids)[i]->vector)[d]=0;
        }
    }

    count_group_sizes(points, sizes);

    while(points != NULL) {
        p = points->value;
        for(d=0; d < dim; d++) {
            ((*centroids)[p->centroid_index]->vector)[d] += (p->vector)[d]/sizes[p->centroid_index];
        }
        points = points->next;
    }
    free(sizes);
    return SUCCESS;
}

// calculates the ditance of each centroid i to the new centroid i
// and returns the maximum distance
double calc_max_centroid_distance(point* new_centroids[], point* old_centroids[], int k) {
    double max_distance = -1;
    double current_distance;
    int i;
    for(i = 0; i<k; i++) {
        current_distance = calculate_distance(new_centroids[i], old_centroids[i]);
        if (current_distance > max_distance) {
            max_distance = current_distance;
        }
    }
    return max_distance;
}

// free the array centroids (with size k)
void free_centroids(point* centroids[], int k) {
    int i;

    if (centroids == NULL){
        return;
    }
    for(i=0; i<k; i++) {
        free_point(centroids[i]);
    }
    free(centroids);
}

// run kmeans algorithm
// returns the final centroids in the array centroids
int kmeans(point_list_node* points, int k, int max_iter, double eps, point*** centroids) {
    point** new_centroids = NULL;
    double err;
    int result;
    int i;
    for(i=0; i<max_iter; i++) {
        find_all_nearest_centroids(*centroids, k, points);
        result = calc_new_centroids(points, k, &new_centroids);
        if (result != SUCCESS) {
            return result;
        }
        err = calc_max_centroid_distance(new_centroids, *centroids, k);
        free_centroids(*centroids, k);
        *centroids = new_centroids;
        new_centroids = NULL;
        if (err < eps) {
            break;
        }
    }
    return SUCCESS;
}

// creates python list of lists that has the centroids
int output_centroids(PyObject* final_centroids, point* centroids[], int k) {
    int result;
    int i;
    for(i=0; i<k; i++) {
        result = add_centroid_to_list(final_centroids, centroids[i], i);
        if (result != SUCCESS) {
            return result;
        }
    }
    return SUCCESS;
}

// creates python list which is the centroid centroid, 
// and put it in centroids_list as index index.
int add_centroid_to_list(PyObject* centroids_list, point* centroid, int index) {
    int result;
    int d;
    PyObject* myDouble;
    PyObject* this_centroid = PyList_New(centroid->dim);
    Py_INCREF(this_centroid);
    if (this_centroid == NULL) {
        return GENERAL_ERR;
    }
    for(d=0; d<centroid->dim; d++) {
        myDouble = Py_BuildValue("d", centroid->vector[d]);
        if (myDouble == NULL) {
            return GENERAL_ERR;
        }

        result = PyList_SetItem(this_centroid, d, myDouble);
        if (result != SUCCESS) {
            return GENERAL_ERR;
        }
    }
    result = PyList_SetItem(centroids_list, index, this_centroid);
    return SUCCESS;
}

// free memory of centroids and the linked list of the data points
void free_resources(point** centroids, point_list_node* points, int k) {
    free_centroids(centroids, k);
    free_point_list_node(points);
}
