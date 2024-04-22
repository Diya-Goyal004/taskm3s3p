#include <stdio.h>
#include <stdlib.h>
#include <CL/cl.h> // Include the OpenCL header for OpenCL functions and definitions
#include <chrono>  // Include for measuring execution time

#define PRINT 1  // Define a flag for conditional printing

// Global variable to store vector size, with a default value
int SZ = 100000000;

// Global pointers for vectors and their output
int *v1, *v2, *v_out;

// OpenCL objects for memory buffers, device, context, program, kernel, queue, and events
cl_mem bufV1, bufV2, bufV_out;
cl_device_id device_id;
cl_context context;
cl_program program;
cl_kernel kernel;
cl_command_queue queue;
cl_event event = NULL;
int err; // Error handling variable

// Function declarations
cl_device_id create_device();
void setup_openCL_device_context_queue_kernel(const char *filename, const char *kernelname);
cl_program build_program(cl_context ctx, cl_device_id dev, const char *filename);
void setup_kernel_memory();
void copy_kernel_args();
void free_memory();
void init(int *&A, int size);
void print(int *A, int size);

// Main function to run the OpenCL code
int main(int argc, char **argv) {
    // If an argument is provided, set the vector size accordingly
    if (argc > 1) {
        SZ = atoi(argv[1]);
    }

    // Initialize the vectors with random data
    init(v1, SZ);
    init(v2, SZ);
    init(v_out, SZ);

    // Set the global work size for OpenCL
    size_t global[1] = {(size_t)SZ};

    // Print initial vector data
    print(v1, SZ);
    print(v2, SZ);

    // Setup OpenCL environment: device, context, queue, and kernel
    setup_openCL_device_context_queue_kernel("./vector_ops.txt", "vector_add_ocl");

    // Setup memory for kernel execution
    setup_kernel_memory();

    // Set kernel arguments
    copy_kernel_args();

    // Start measuring kernel execution time
    auto start = std::chrono::high_resolution_clock::now();

    // Execute the kernel
    clEnqueueNDRangeKernel(queue, kernel, 1, NULL, global, NULL, 0, NULL, &event);
    clWaitForEvents(1, &event); // Wait for the kernel to finish execution

    // Read the output data from the OpenCL device to host
    clEnqueueReadBuffer(queue, bufV_out, CL_TRUE, 0, SZ * sizeof(int), &v_out[0], 0, NULL, NULL);

    // Print the output data
    print(v_out, SZ);

    // Stop measuring time and calculate the elapsed time
    auto stop = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> elapsed_time = stop - start;

    // Display the kernel execution time
    printf("Kernel Execution Time: %f ms\n", elapsed_time.count());

    // Free all allocated memory and OpenCL objects
    free_memory();
}

// Function to initialize a vector with random data
void init(int *&A, int size) {
    A = (int *)malloc(sizeof(int) * size); // Allocate memory for the vector

    for (long i = 0; i < size; i++) {
        A[i] = rand() % 100; // Initialize with random integers from 0 to 99
    }
}

// Function to print a vector's content
void print(int *A, int size) {
    if (PRINT == 0) { // If PRINT is disabled, do nothing
        return;
    }

    // Conditionally print the start and end of large vectors
    if (PRINT == 1 && size > 15) {
        for (long i = 0; i < 5; i++) {
            printf("%d ", A[i]);
        }
        printf(" ..... "); // Ellipsis for omitted elements
        for (long i = size - 5; i < size; i++) {
            printf("%d ", A[i]);
        }
    } else { // Otherwise, print all elements
        for (long i = 0; i < size; i++) {
            printf("%d ", A[i]);
        }
    }
    printf("\n----------------------------\n");
}

// Function to free memory and release OpenCL objects
void free_memory() {
    clReleaseMemObject(bufV1);
    clReleaseMemObject(bufV2);
    clReleaseMemObject(bufV_out);

    clReleaseKernel(kernel);
    clReleaseCommandQueue(queue);
    clReleaseProgram(program);
    clReleaseContext(context);

    free(v1);
    free(v2);
    free(v_out); 
}

// Function to set kernel arguments
void copy_kernel_args() {
    err = clSetKernelArg(kernel, 0, sizeof(int), (void *)&SZ);
    err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&bufV1);
    err |= clSetKernelArg(kernel, 2, sizeof(cl_mem), (void *)&bufV2);
    err |= clSetKernelArg(kernel, 3, sizeof(cl_mem), (void *)&bufV_out);

    if (err < 0) {
        perror("Couldn't create a kernel argument");
        printf("Error code = %d", err);
        exit(1);
    }
}

// Function to allocate memory for OpenCL buffers
void setup_kernel_memory() {
    bufV1 = clCreateBuffer(context, CL_MEM_READ_WRITE, SZ * sizeof(int), NULL, NULL);
    bufV2 = clCreateBuffer(context, CL_MEM_READ_WRITE, SZ * sizeof(int), NULL, NULL);
    bufV_out = clCreateBuffer(context, CL_MEM_READ_WRITE, SZ * sizeof(int), NULL, NULL);

    // Write data to buffers
    clEnqueueWriteBuffer(queue, bufV1, CL_TRUE, 0, SZ * sizeof(int), &v1[0], 0, NULL, NULL);
    clEnqueueWriteBuffer(queue, bufV2, CL_TRUE, 0, SZ * sizeof(int), &v2[0], 0, NULL, NULL);
}

// Function to set up OpenCL device, context, queue, and kernel
void setup_openCL_device_context_queue_kernel(const char *filename, const char *kernelname) {
    device_id = create_device(); // Create the OpenCL device
    cl_int err;

    // Create the OpenCL context
    context = clCreateContext(NULL, 1, &device_id, NULL, NULL, &err);
    if (err < 0) {
        perror("Couldn't create a context");
        exit(1);
    }

    // Build the OpenCL program from source
    program = build_program(context, device_id, filename);

    // Create the command queue
    queue = clCreateCommandQueueWithProperties(context, device_id, 0, &err);
    if (err < 0) {
        perror("Couldn't create a command queue");
        exit(1);
    }

    // Create the OpenCL kernel
    kernel = clCreateKernel(program, kernelname, &err);
    if (err < 0) {
        perror("Couldn't create a kernel");
        printf("Error code = %d", err);
        exit(1);
    }
}

// Function to build an OpenCL program from a source file
cl_program build_program(cl_context ctx, cl_device_id dev, const char *filename) {
    cl_program program;
    FILE *program_handle; // File handle to read the source
    char *program_buffer; // Buffer for source code
    size_t program_size, log_size;

    // Open the source file
    program_handle = fopen(filename, "r");
    if (program_handle == NULL) {
        perror("Couldn't find the program file");
        exit(1);
    }

    // Read the source code
    fseek(program_handle, 0, SEEK_END);
    program_size = ftell(program_handle);
    rewind(program_handle);
    program_buffer = (char *)malloc(program_size + 1); 
    program_buffer[program_size] = '\0'; // Null-terminate the source
    fread(program_buffer, sizeof(char), program_size, program_handle);
    fclose(program_handle); // Close the file

    // Create the OpenCL program
    program = clCreateProgramWithSource(ctx, 1, (const char **)&program_buffer, &program_size, &err);
    if (err < 0) {
        perror("Couldn't create the program");
        exit(1);
    }
    free(program_buffer); // Free the buffer

    // Build the OpenCL program
    err = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
    if (err < 0) {
        // If there's a build error, retrieve and display the build log
        clGetProgramBuildInfo(program, dev, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
        char *program_log = (char *)malloc(log_size + 1);
        program_log[log_size] = '\0';
        clGetProgramBuildInfo(program, dev, CL_PROGRAM_BUILD_LOG, log_size + 1, program_log, NULL);
        printf("%s\n", program_log);
        free(program_log); // Free the build log
        exit(1);
    }

    return program; // Return the built program
}

// Function to create an OpenCL device
cl_device_id create_device() {
   cl_platform_id platform;
   cl_device_id dev;
   int err;

   // Identify an OpenCL platform
   err = clGetPlatformIDs(1, &platform, NULL);
   if(err < 0) {
      perror("Couldn't identify a platform");
      exit(1);
   }

   // Attempt to get a GPU device
   err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &dev, NULL);
   if(err == CL_DEVICE_NOT_FOUND) {
      // If no GPU found, fall back to a CPU device
      err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_CPU, 1, &dev, NULL);
   }
   if(err < 0) {
      perror("Couldn't access any devices");
      exit(1);
   }

   return dev; // Return the device identifier
}
