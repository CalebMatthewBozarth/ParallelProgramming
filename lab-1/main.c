#include <stdlib.h>
#include <stdio.h>
#include <omp.h>
#include <time.h>
#include <sys/time.h>

// Returns a random value between -1 and 1
double getRand(unsigned int *seed) {
    return (double) rand_r(seed) * 2 / (double) (RAND_MAX) - 1;
}

// Calculates PI using one thread (Monte Carlo Method)
long double Calculate_Pi_Sequential(long long number_of_tosses) {
    long double pi_estimate; // Variable to hold the estimate of PI
    unsigned int seed = (unsigned int) time(NULL); // Seed random num
    double number_in_circle = 0; // Variable to keep track of how many numbers are in the circle
    for(int toss = 0; toss < number_of_tosses; toss++) { // Sequential loop (using one thread)
        double x = getRand(&seed);
        double y = getRand(&seed);
        double distance_squared = (x*x) + (y*y);
        if(distance_squared <= 1) { // The number is inside the circle
            number_in_circle++;
        }
    }
    pi_estimate = (4*number_in_circle)/((double) number_of_tosses); // 4 * how many points in the circle / how many throws
    return pi_estimate;
}

// Calculate PI using 4 threads (that is how many my computer has) (Monte Carlo Method)
long double Calculate_Pi_Parallel(long long number_of_tosses) {
    long double pi_estimate; // Variable to hold the estimate of PI
    double number_in_circle = 0; // Variable to keep track of how many numbers are in the circle
#pragma omp parallel num_threads(omp_get_max_threads()) // This block of code will be using parallelization
    {
        unsigned int seed = (unsigned int) time(NULL) + (unsigned int) omp_get_thread_num(); // Seed random num
#pragma omp for reduction(+: number_in_circle) // Run the for loop in parallel (give each thread a local variable for number_in_circle)
        for(int toss = 0; toss < number_of_tosses; toss++) {
            double x = getRand(&seed);
            double y = getRand(&seed);
            double distance_squared = (x*x) + (y*y);

            if(distance_squared <= 1) { // The number is inside the circle
                number_in_circle++;
            }
        }
        pi_estimate = (4*number_in_circle)/((double) number_of_tosses); // 4 * how many points in the circle / how many throws
    }

    return pi_estimate;
}

int main() {
    struct timeval start, end;

    long long num_tosses = 10000000;

    printf("Timing sequential...\n");
    gettimeofday(&start, NULL);
    long double sequential_pi = Calculate_Pi_Sequential(num_tosses);
    gettimeofday(&end, NULL);
    printf("Took %f seconds\n\n", end.tv_sec - start.tv_sec + (double) (end.tv_usec - start.tv_usec) / 1000000);

    printf("Timing parallel...\n");
    gettimeofday(&start, NULL);
    long double parallel_pi = Calculate_Pi_Parallel(num_tosses);
    gettimeofday(&end, NULL);
    printf("Took %f seconds\n\n", end.tv_sec - start.tv_sec + (double) (end.tv_usec - start.tv_usec) / 1000000);

    // This will print the result to 10 decimal places
    printf("π = %.10Lf (sequential)\n", sequential_pi);
    printf("π = %.10Lf (parallel)", parallel_pi);

    return 0;
}