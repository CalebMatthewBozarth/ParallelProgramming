#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <omp.h>
#include <time.h>
#include <sys/time.h>

typedef struct Complex {
    long double real;
    long double imaginary;
} Complex;
typedef unsigned char RGB_Pixel[3];
static const unsigned char MAX_RGB_VAL = 255;

static const int Image_Width = 5000;
static const int Image_Height = 5000;
static const int Max_Iterations = 1000;

static const Complex Focus_Point = {.real = -0.5, .imaginary = 0};
static const long double Zoom = 2;

// We use the coloring schema outlined from https://solarianprogrammer.com/2013/02/28/mandelbrot-set-cpp-11/
void calc_colors(RGB_Pixel *colors) {
    int numThreads = omp_get_max_threads(); // Number of threads
    long n = Max_Iterations / numThreads; // Number of elements to compute
    int i; // i index variable to be declared as private for each thread
    int thread_num; // used as a private variable for each thread
    int start, stop; // start / stop used for chunking up the array into different parts
#pragma omp parallel num_threads(omp_get_max_threads()) private(i, thread_num, start, stop) // This block of code will be using parallelization
    {
        thread_num = omp_get_thread_num(); // Thread number
        start = thread_num * n; // Start point for this thread
        stop = (thread_num + 1) * n; // Stopping point for this thread

        for (i = start; i < stop; i++) {
            double t = (double) i / Max_Iterations;

            colors[i][0] = (unsigned char) (9 * (1 - t) * t * t * t * MAX_RGB_VAL);
            colors[i][1] = (unsigned char) (15 * (1 - t) * (1 - t) * t * t * MAX_RGB_VAL);
            colors[i][2] = (unsigned char) (8.5 * (1 - t) * (1 - t) * (1 - t) * t * MAX_RGB_VAL);
        }
    }

};

/* The Mandelbrot set is defined by all numbers which do not diverge for fc(z) = z^2 + c,
 * where C is a complex number. Generally, we run the algorithm until we hit a cutoff number of iterations.
 * We can end the iterations early if we know that the sum of the complex coefficients is <= 4.
 * because if that happens we know it'll diverge.
 *
 * To draw the set, we map the real value to the x-axis, and the imaginary value to the y-axis.
 * We then use the number of iterations to escape to calculate the color of the pixel
 *
 * To convert the resulting PPM, you may use http://www.imagemagick.org
 */

int main(int argc, const char **argv) {
    struct timeval start, end;

    printf("Timing parallel...\n");
    gettimeofday(&start, NULL);
    RGB_Pixel *pixels = malloc(sizeof(RGB_Pixel) * Image_Width * Image_Height);

    RGB_Pixel colors[Max_Iterations + 1];
    calc_colors(colors);
    colors[Max_Iterations][0] = MAX_RGB_VAL;
    colors[Max_Iterations][1] = MAX_RGB_VAL;
    colors[Max_Iterations][2] = MAX_RGB_VAL;

    // Calculate scaling values to map the bounds of the Mandelbrot area to the pixel grid
    const Complex min_bounds = {.real = Focus_Point.real - Zoom, .imaginary = Focus_Point.imaginary - Zoom};
    const Complex max_bounds = {.real = Focus_Point.real + Zoom, .imaginary = Focus_Point.imaginary + Zoom};
    const Complex scale = {
            .real = (max_bounds.real - min_bounds.real) / Image_Width,
            .imaginary = (max_bounds.real - min_bounds.real) / Image_Height
    };

    // Loop through the image pixels

#pragma omp parallel num_threads(omp_get_max_threads()) // This block of code will be using parallelization
    {
#pragma omp for collapse(2)
        for (int img_y = 0; img_y < Image_Height; img_y++) {
            for (int img_x = 0; img_x < Image_Width; img_x++) {
                // Find the value of C in the Mandelbrot range corresponding to this pixel
                Complex c = {
                        .real = min_bounds.real + img_x * scale.real,
                        .imaginary = min_bounds.imaginary + img_y * scale.imaginary
                };

                // Check if the current pixel is in the Mandelbrot set
                // We use the optimizations from https://randomascii.wordpress.com/2011/08/13/faster-fractals-through-algebra/
                Complex z = {.real = 0, .imaginary = 0};
                Complex z_squared = {.real = 0, .imaginary = 0};

                int iterations = 0;
                while (z_squared.real + z_squared.imaginary <= 4 && iterations < Max_Iterations) {
                    z.imaginary = z.real * z.imaginary;
                    z.imaginary += z.imaginary;
                    z.imaginary += c.imaginary;

                    z.real = z_squared.real - z_squared.imaginary + c.real;

                    z_squared.real = z.real * z.real;
                    z_squared.imaginary = z.imaginary * z.imaginary;

                    iterations++;
                }

                pixels[img_y * Image_Width + img_x][0] = colors[iterations][0];
                pixels[img_y * Image_Width + img_x][1] = colors[iterations][1];
                pixels[img_y * Image_Width + img_x][2] = colors[iterations][2];
            }
        }
    }
    gettimeofday(&end, NULL);
    printf("Took %f seconds\n\n", end.tv_sec - start.tv_sec + (double) (end.tv_usec - start.tv_usec) / 1000000);

    FILE *fp = fopen("MandelbrotSet.ppm", "wb");
    fprintf(fp, "P6\n %d %d\n %d\n", Image_Width, Image_Height, MAX_RGB_VAL);
    fwrite(pixels, sizeof(RGB_Pixel), Image_Width * Image_Width, fp);
    fclose(fp);

    free(pixels);
    free(colors);

    return 0;
}