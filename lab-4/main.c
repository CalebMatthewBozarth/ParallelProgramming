#include <stdlib.h>
#include <stdio.h>
#include <omp.h>
#include <time.h>
#include <stdbool.h>
#include <sys/time.h>
#include <memory.h>

//static const long Num_To_Sort = 1000000000;
static const long Num_To_Sort = 100000000;

// Swap two array elements
void swap(int *a, int *b) {
    int t = *a;
    *a = *b;
    *b = t;
}

// This function takes the last element as a pivot.
// It places the pivot in the correct spot of the sorted array
// It sorts smaller elements before it, and larger elements after it
// This function is where all of the sorting really happens
int partition(int *arr, int low, int high) {
    int pivot = arr[high];    // pivot
    int i = (low - 1);  // Index of smaller element
    for (int j = low; j <= high- 1; j++)
    {
        // If current element is smaller than the pivot
        if (arr[j] < pivot)
        {
            i++;    // increment index of smaller element
            swap(&arr[i], &arr[j]);
        }
    }

    swap(&arr[i + 1], &arr[high]);
    return (i + 1);
}

// Sequential version of your sort
// If you're implementing the PSRS algorithm, you may ignore this section
void sort_s(int *arr, int low, int high) {
    if (low < high) { // Continue as long as the low and high have not crossed each other
        /* pi is partitioning index, arr[pi] is now
           at right place */
        int pi = partition(arr, low, high);

        // Recursive function call
        sort_s(arr, low, pi - 1); // Before pivot index
        sort_s(arr, pi + 1, high); // After pivot index
    }
}

// Parallel version of your sort
void sort_p(int *arr, int low, int high) {
    if (low < high) { // Continue as long as the low and high have not crossed each other
        /* pi is partitioning index, arr[pi] is now
           at right place */
        int pi = partition(arr, low, high);

        // Place each recursive function call in its own task
#pragma omp task
        sort_s(arr, low, pi - 1); // Before pivot index is one task
#pragma omp task
        sort_s(arr, pi + 1, high); // After pivot index is another task
    }
}

// All this function does is initially call the sequential sort function
void startQuickSortSequential(int arr[], int low, int high) {
    sort_s(arr, low, high);
}

// This function declares a parallel block of code and calls the parallel sort function
void startQuickSortParallel(int arr[], int low, int high) {
#pragma omp parallel
    {
// This is because parallelism does not work very well recursively.
// This will perform better than doing #pragma omp section
// Since I declare #pragma omp single, in the sort function I can make each recursive call its own task
#pragma omp single
        sort_p(arr, low, high);
    };
}

int main() {
    int *arr_s = malloc(sizeof(int) * Num_To_Sort);
    long chunk_size = Num_To_Sort / omp_get_max_threads();
#pragma omp parallel num_threads(omp_get_max_threads())
    {
        int p = omp_get_thread_num();
        unsigned int seed = (unsigned int) time(NULL) + (unsigned int) p;
        long chunk_start = p * chunk_size;
        long chunk_end = chunk_start + chunk_size;
        for (long i = chunk_start; i < chunk_end; i++) {
            arr_s[i] = rand_r(&seed);
        }
    }

    // Copy the array so that the sorting function can operate on it directly.
    // Note that this doubles the memory usage.
    // You may wish to test with slightly smaller arrays if you're running out of memory.
    int *arr_p = malloc(sizeof(int) * Num_To_Sort);
    memcpy(arr_p, arr_s, sizeof(int) * Num_To_Sort);

    struct timeval start, end;

    printf("Timing sequential...\n");
    gettimeofday(&start, NULL);
    startQuickSortSequential(arr_s, 0, Num_To_Sort-1); // This function calls sort_s
    gettimeofday(&end, NULL);
    printf("Took %f seconds\n\n", end.tv_sec - start.tv_sec + (double) (end.tv_usec - start.tv_usec) / 1000000);
//    for(long i = 0; i < Num_To_Sort; i++) {
//        printf("%d, ", arr_s[i]);
//    }
//    printf("\n\n");
    free(arr_s);

    printf("Timing parallel...\n");
    gettimeofday(&start, NULL);
    startQuickSortParallel(arr_p, 0, Num_To_Sort-1); // This function calls sort_p
    gettimeofday(&end, NULL);
    printf("Took %f seconds\n\n", end.tv_sec - start.tv_sec + (double) (end.tv_usec - start.tv_usec) / 1000000);
//    for(long i = 0; i < Num_To_Sort; i++) {
//        printf("%d, ", arr_p[i]);
//    }
//    printf("\n\n");
    free(arr_p);

    return 0;
}

