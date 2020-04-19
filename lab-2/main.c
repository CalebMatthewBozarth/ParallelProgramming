#include <stdlib.h>
#include <stdio.h>
#include <omp.h>
#include <time.h>
#include <sys/time.h>

static const long Num_To_Add = 1000000000;
static const double Scale = 10.0 / RAND_MAX;

// Adds up each element of the numbers array serial
long add_serial(const char *numbers) {
    long sum = 0;
    for (long i = 0; i < Num_To_Add; i++) {
        sum += numbers[i];
    }
    return sum;
}

// Adds up each element of the numbers array in parallel
long add_parallel(const char *numbers) {
    long sum = 0;
    int numThreads = omp_get_max_threads(); // Number of threads
    long n = Num_To_Add / numThreads; // Number of elements to compute
    long i; // i index variable to be declared as private for each thread
    int thread_num; // used as a private variable for each thread
    int start, stop; // start / stop used for chunking up the array into different parts


#pragma omp parallel num_threads(omp_get_max_threads()) private(i, thread_num, start, stop) reduction(+: sum) // This block of code will be using parallelization with private variables for each thread
    {
        thread_num = omp_get_thread_num(); // Thread number
        start = thread_num * n; // Start point for this thread
        stop = (thread_num + 1) * n; // Stopping point for this thread

// For loop to go through the chunk allocated to each thread
        for (i = start; i < stop; i++) {
            sum += numbers[i];
        }
    }
    return sum;
}

int main() {
    char *numbers = malloc(sizeof(long) * Num_To_Add);

    long chunk_size = Num_To_Add / omp_get_max_threads();
#pragma omp parallel num_threads(omp_get_max_threads())
    {
        int p = omp_get_thread_num();
        unsigned int seed = (unsigned int) time(NULL) + (unsigned int) p;
        long chunk_start = p * chunk_size;
        long chunk_end = chunk_start + chunk_size;
        for (long i = chunk_start; i < chunk_end; i++) {
            numbers[i] = (char) (rand_r(&seed) * Scale);
        }
    }

    struct timeval start, end;

    printf("Timing sequential...\n");
    gettimeofday(&start, NULL);
    long sum_s = add_serial(numbers);
    gettimeofday(&end, NULL);
    printf("Took %f seconds\n\n", end.tv_sec - start.tv_sec + (double) (end.tv_usec - start.tv_usec) / 1000000);

    printf("Timing parallel...\n");
    gettimeofday(&start, NULL);
    long sum_p = add_parallel(numbers);
    gettimeofday(&end, NULL);
    printf("Took %f seconds\n\n", end.tv_sec - start.tv_sec + (double) (end.tv_usec - start.tv_usec) / 1000000);

    printf("Sum serial: %ld\nSum parallel: %ld", sum_s, sum_p);

    free(numbers);
    return 0;
}

