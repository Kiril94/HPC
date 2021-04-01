#include <vector>
#include <iostream>
#include <H5Cpp.h>
#include <chrono>
#include <cmath>
#include <numeric>
#include <argparse.hpp>
#include <omp.h>
#include <cassert>
#include <mpi.h>

// Get the number of processes
int mpi_size;

// Get the rank of the process
int mpi_rank;

constexpr auto dt = 0.05;   // Size of the integration time step
constexpr auto g = 9.80665; // Gravitational acceleration
constexpr auto dx = 1;      // Integration step size in the horizontal direction
constexpr auto dy = 1;      // Integration step size in the horizontal direction

/** Representation of a two dimensional shape */
class Shape {
public:
    uint64_t rows;   // Number of rows
    uint64_t cols;   // Number of columns
    uint64_t total;  // Total number of elements

    /** Create a new shape
     *
     * @param rows  Number of rows
     * @param cols  Number of columns
     */
    Shape(uint64_t rows, uint64_t cols) : rows(rows), cols(cols), total(rows * cols) {}
};

/** Help function to write the `data` to screen.
 *
 * @param data       The data to print.
 * @param shape      The shape of `data`.
 * @param show_stat  Whether to print additional statistics.
 */
void print(std::vector<double> &data, const Shape &shape, bool show_stat = true) {
    for (uint64_t i = 0; i < shape.rows; ++i) {
        for (uint64_t j = 0; j < shape.cols; ++j) {
            if (!std::signbit(data[i * shape.cols + j])) {
                std::cout << " ";
            }
            std::cout << std::scientific << data[i * shape.cols + j] << " ";
        }
        std::cout << "\n";
    }
    if (show_stat) {
        std::cout << "[shape: (" << shape.rows << ", " << shape.cols << ")"
                  << ", min: " << *std::min_element(data.begin(), data.end())
                  << ", max: " << *std::max_element(data.begin(), data.end())
                  << ", avg: " << std::accumulate(data.begin(), data.end(), 0.0) / data.size()
                  << ", checksum: " << std::accumulate(data.begin(), data.end(), 0.0)
                  << "]\n";
    }
    std::cout << std::endl;
}

/** Representation of a water world including ghost lines, which is a "1-cell padding" of rows and columns
 *  around the world. These ghost lines is a technique to implement periodic boundary conditions. */
class Water {
public:
    Shape shape; // The shape of the water world including ghost lines.
    std::vector<double> u; // The speed in the horizontal direction.
    std::vector<double> v; // The speed in the vertical direction.
    std::vector<double> e; // The water elevation.

    Water(Shape shape) : shape(shape), u(shape.total, 0), v(shape.total, 0), e(shape.total, -100000) {}
};

/** Create a new water world. The would contains a droplet in the center.
 *
 * @note This function differ from the sequential version since it needs the global offset in order to
 *       place the water droplet correctly.
 *
 * @param shape         The local shape of the world including ghost lines
 * @param global_shape  The global shape of the world excluding ghost lines
 * @param offset_d0     The global offset in the vertical direction
 * @param offset_d1     The global offset in the horizontal direction
 * @return
 */
Water createWater(Shape shape, Shape global_shape, uint64_t offset_d0, uint64_t offset_d1) {
    Water w(shape);
    for (uint64_t i = 1; i < w.shape.rows - 1; ++i) {
        for (uint64_t j = 1; j < w.shape.cols - 1; ++j) {
            int64_t ii = i - (global_shape.rows - 2) / 2 + offset_d0;
            int64_t jj = j - (global_shape.cols - 2) / 2 + offset_d1;
            w.e[i * w.shape.cols + j] = std::exp(-0.02 * (ii * ii + jj * jj));
        }
    }
    return w;
}

/** Help function to remove the vertical and/or the horizontal ghost lines.
 *
 * @param data        The input data, which could be the water elevation or the speed.
 * @param shape       The shape of `data` including ghost lines.
 * @param vertical    Whether to remove vertical ghost lines.
 * @param horizontal  Whether to remove horizontal ghost lines.
 * @return
 */
std::vector<double> remove_ghost_lines(const std::vector<double> &data, const Shape &shape,
                                       bool vertical, bool horizontal) {
    uint64_t v = vertical ? 1 : 0;
    uint64_t h = horizontal ? 1 : 0;
    std::vector<double> ret;
    for (uint64_t i = h; i < shape.rows - h; ++i) {
        for (uint64_t j = v; j < shape.cols - v; ++j) {
            ret.push_back(data[i * shape.cols + j]);
        }
    }
    return ret;
}


/** Exchange the horizontal ghost lines i.e. copy the second data row to the very last data row and vice versa.
 *
 * @param data   The data update, which could be the water elevation `e` or the speed in the horizontal direction `u`.
 * @param shape  The shape of data including the ghost lines.
 */
void exchange_horizontal_ghost_lines(std::vector<double> &data, Shape shape, int const &mpi_rank, int const &mpi_size) {
        // Copy guard cells for other ranks to buffer
    int top_rank = (mpi_rank+1) % mpi_size;
    int bot_rank;
    if(mpi_rank==0){
        bot_rank = mpi_size-1;
    }
    else{
        bot_rank = (mpi_rank-1) % mpi_size;//This would give -1 for rank 0
    }
    //std::cout<<"top rank: "<<top_rank<<'\n';
    //std::cout<<"bottom rank: "<<bot_rank<<'\n';
    std::vector<double> emit_top(shape.cols);
    std::vector<double> emit_bot(shape.cols);
    std::vector<double> recv_top(shape.cols);
    std::vector<double> recv_bot(shape.cols);
    
    for (uint64_t i = 0; i < shape.cols; ++i) {
    emit_top[i] = data[i+shape.cols]; // copy 2nd row to buffer
    emit_bot[i] = data[i+(shape.rows-2)*shape.cols]; // copy 2nd last column to buffer
    }
    MPI_Request req[4];
    MPI_Isend(&emit_top[0], shape.cols, MPI_DOUBLE, top_rank, 0, MPI_COMM_WORLD, &req[0]);
    MPI_Isend(&emit_bot[0], shape.cols, MPI_DOUBLE, bot_rank, 0, MPI_COMM_WORLD , &req[1]);
    MPI_Irecv(&recv_top[0], shape.cols, MPI_DOUBLE, top_rank, 0, MPI_COMM_WORLD, &req[2]);
    MPI_Irecv(&recv_bot[0], shape.cols, MPI_DOUBLE, bot_rank, 0, MPI_COMM_WORLD, &req[3]);
    MPI_Waitall(4, &req[0], MPI_STATUSES_IGNORE);
    
    for (uint64_t i = 0; i < shape.cols; ++i) {
        data[i] = recv_top[i]; // insert buffer in to 1st row
        data[i+(shape.rows-1)*shape.cols] = recv_bot[i]; // insert buffer in to last row
    }
}

/** Exchange the vertical ghost lines i.e. copy the second data column to the rightmost data column and vice versa.
 *
 * @param data   The data update, which could be the water elevation `e` or the speed in the vertical direction `v`.
 * @param shape  The shape of data including the ghost lines.
 */
void exchange_vertical_ghost_lines(std::vector<double> &data, Shape shape) {
    for (uint64_t i = 0; i < shape.rows; ++i) {
        const uint64_t left_ghost = i * shape.cols + 0;
        const uint64_t right_water = i * shape.cols + shape.cols - 2;
        const uint64_t right_ghost = i * shape.cols + shape.cols - 1;
        const uint64_t left_water = i * shape.cols + 1;
        data[left_ghost] = data[right_water];
        data[right_ghost] = data[left_water];
    }
}



/** One integration step
 *
 * @param w The water world to update.
 */
void integrate(Water &w, int const &mpi_rank, int const &mpi_size) {
    
    
    exchange_horizontal_ghost_lines(w.e, w.shape, mpi_rank, mpi_size);
    exchange_horizontal_ghost_lines(w.v, w.shape, mpi_rank, mpi_size);

    exchange_vertical_ghost_lines(w.e, w.shape);
    exchange_vertical_ghost_lines(w.u, w.shape);

    const uint64_t stride = w.shape.cols;
    for (uint64_t i = 1; i < w.shape.rows - 1; ++i) {
        for (uint64_t j = 1; j < w.shape.cols - 1; ++j) {
            w.u[i * stride + j] = w.u[i * stride + j] - dt * g * (w.e[i * stride + j + 1] - w.e[i * stride + j]) / dx;
            w.v[i * stride + j] = w.v[i * stride + j] - dt * g * (w.e[(i + 1) * stride + j] - w.e[i * stride + j]) / dy;
        }
    }

    for (uint64_t i = 1; i < w.shape.rows - 1; ++i) {
        for (uint64_t j = 1; j < w.shape.cols - 1; ++j) {
            w.e[i * stride + j] = w.e[i * stride + j] - dt * (w.u[i * stride + j] - w.u[i * stride + j - 1]) / dx -
                                  dt * (w.v[i * stride + j] - w.v[(i - 1) * stride + j]) / dy;
        }
    }
}

/** Write data to a hdf5 file
 *
 * @param group  The hdf5 group to write in
 * @param name   The name of the data
 * @param shape  The shape of the data
 * @param data   The data
 */
void write_hdf5(H5::Group &group, const std::string &name, const std::vector <hsize_t> &shape,
                const std::vector<double> &data) {
    H5::DataSpace dataspace(static_cast<int>(shape.size()), &shape[0]);
    H5::DataSet dataset = group.createDataSet(name.c_str(), H5::PredType::NATIVE_DOUBLE, dataspace);
    dataset.write(&data[0], H5::PredType::NATIVE_DOUBLE);
}

/** Write a history of the water heights to a HDF5 file
 *
 * @param water_history  Vector of the all water worlds to write
 * @param filename       The output filename of the HDF5 file
 */
void write_hdf5(const std::vector <std::vector<double>> &square_matrix_history, const std::string &filename) {
    H5::H5File file(filename, H5F_ACC_TRUNC);
    for (uint64_t i = 0; i < square_matrix_history.size(); ++i) {
        H5::Group group(file.createGroup("/" + std::to_string(i)));
        if (std::sqrt(square_matrix_history[i].size()) * std::sqrt(square_matrix_history[i].size()) !=
            square_matrix_history[i].size()) {
            throw std::invalid_argument("write_hdf5() - the square_matrix matrices must be squares");
        }
        uint64_t size = std::sqrt(square_matrix_history[i].size());
        write_hdf5(group, "water", {size, size}, square_matrix_history[i]);
    }
}

/** Simulation of shallow water
 *
 * @param num_of_iterations  The number of time steps to simulate
 * @param size               The global size of the water world excluding ghost lines
 * @param output_filename    The filename of the written water world history (HDF5 file)
 */
void simulate(uint64_t num_of_iterations, uint64_t size, const std::string &output_filename, int const &mpi_rank, int const &mpi_size) {
    // We pad the world with ghost lines (two in each dimension)
       
    uint64_t local_rows = size/mpi_size;//number of local rows
    Shape local_shape = Shape(size + 2, local_rows + 2);
    Shape local_shape_no_gl = Shape(size, local_rows);
    //Shape global_shape = Shape(size + 2, size + 2);
    Shape global_shape_no_gl = Shape(size, size);
        
    //The offset should depend on the rank now
    uint64_t start_row = local_rows * mpi_rank;
    //std::cout<<"rank number:"<<mpi_rank<<'\n';
    //std::cout<<"local row:"<<start_row<<'\n';
    uint64_t start_col = 0;

    Water local_water_world = createWater(local_shape, global_shape_no_gl, start_row, start_col);
    //Water water_world_global
    MPI_Barrier(MPI_COMM_WORLD);
    
    std::vector <std::vector<double>> water_history;
    double checksum = 0;
    auto begin = std::chrono::steady_clock::now();
    for (uint64_t t = 0; t < num_of_iterations; ++t) {
        integrate(local_water_world, mpi_rank, mpi_size);
        if (!output_filename.empty()) {
            // TODO: gather the water elevation `e` on rank zero
            MPI_Barrier(MPI_COMM_WORLD);
                
            std::vector<double> local_water_world_no_gl_e = remove_ghost_lines(local_water_world.e,\
                            local_shape, true, true);
            uint64_t num_elements_local = local_shape_no_gl.cols * local_shape_no_gl.rows;
            std::vector<double> water_world_e(size*size);
       
            uint64_t start_index_global = mpi_rank*global_shape_no_gl.cols*local_rows;
            if (mpi_rank==0){
                MPI_Gather(&local_water_world_no_gl_e[0], num_elements_local, MPI_DOUBLE,\
                        &water_world_e[start_index_global], num_elements_local, \
                        MPI_DOUBLE, 0, MPI_COMM_WORLD);
            }
            else{
                MPI_Gather(&local_water_world_no_gl_e[0], num_elements_local, MPI_DOUBLE,
                           NULL, num_elements_local, MPI_DOUBLE, 0,MPI_COMM_WORLD);
            }
            
            MPI_Barrier(MPI_COMM_WORLD);
            // Only rank zero should save the water history
            if (mpi_rank == 0) {
                std::vector<double> e = water_world_e;
                water_history.push_back(e);
                std::cout << t << " -- min: " << *std::min_element(e.begin(), e.end())
                          << ", max: " << *std::max_element(e.begin(), e.end())
                          << ", avg: " << std::accumulate(e.begin(), e.end(), 0.0) / e.size()
                          << "\n";
                checksum += std::accumulate(e.begin(), e.end(), 0.0);
            }
        }
    }
    // Make sure that all MPI-processes are finished.
    MPI_Barrier(MPI_COMM_WORLD);

    // Rank zero writes the water history to file
    if (mpi_rank == 0) {
        if (!output_filename.empty()) {
            write_hdf5(water_history, output_filename);
        }
        auto end = std::chrono::steady_clock::now();
        if (!output_filename.empty()) {
            std::cout << "checksum: " << checksum << std::endl;
        }
        std::cout << "elapsed time: " << (end - begin).count() / 1000000000.0 << " sec" << std::endl;
    }
}

/** Main function that parses the command line and start the simulation */
int main(int argc, char **argv) {

    // Initialize the MPI environment
    MPI_Init(&argc, &argv);

    util::ArgParser args(argc, argv);
    int64_t size, iterations;
    if (args.cmdOptionExists("--iter")) {
        iterations = std::stoi(args.getCmdOption("--iter"));
        if (iterations < 0) {
            throw std::invalid_argument("iter must be a positive integer (e.g. --iter 100)");
        }
    } else {
        throw std::invalid_argument("You must specify the number of iterations (e.g. --iter 100)");
    }
    if (args.cmdOptionExists("--size")) {
        size = std::stoi(args.getCmdOption("--size"));
        if (size < 0) {
            throw std::invalid_argument("size most be a positive integer (e.g. --size 100)");

        }
    } else {
        throw std::invalid_argument("You must specify the size of the water, which is assumed squired e.g. " \
                                    "--size 100 is a 100 by 100 water world)");
    }
    const std::string &output_filename = args.getCmdOption("--out");

    // Get the number of processes
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);

    // Get the rank of the process
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);

    // Get the name of the processor
    char processor_name[MPI_MAX_PROCESSOR_NAME];
    int name_len;
    MPI_Get_processor_name(processor_name, &name_len);

    // Print off a hello world message
    printf("Shallow Water running on `%s`, rank %d out of %d.\n", processor_name, mpi_rank, mpi_size);

    simulate(iterations, size, output_filename, mpi_rank, mpi_size );

    MPI_Finalize();

    return 0;
}
