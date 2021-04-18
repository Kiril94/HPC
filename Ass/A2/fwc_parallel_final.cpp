#include <vector>
#include <iostream>
#include <H5Cpp.h>
#include <chrono>
#include <cmath>
#include <numeric>
#include <argparse.hpp>
#include <omp.h>

/** Representation of a flat world */
class World {
public:
    // The size of the world in the latitude dimension
    uint64_t latitude;
    // The size of the world in the longitude dimension
    uint64_t longitude;
    // The temperature of each coordinate of the world.
    // NB: it is up to the calculation to interpret this vector in two dimension.
    std::vector<double> data;
    // The measure of the diffuse reflection of solar radiation at each world coordinate.
    // See: <https://en.wikipedia.org/wiki/Albedo>
    // NB: this vector has the same length as `data` and must be interpret in two dimension as well.
    std::vector<double> albedo_data;

    /** Create a new flat world.
     *
     * @param latitude     The size of the world in the latitude dimension.
     * @param longitude    The size of the world in the longitude dimension.
     * @param temperature  The initial temperature (the whole world starts with the same temperature).
     * @param albedo_data  The measure of the diffuse reflection of solar radiation at each world coordinate.
     *                     This vector must have the size: `latitude * longitude`.
     */
    World(uint64_t latitude, uint64_t longitude, double temperature,
          std::vector<double> albedo_data) : latitude(latitude), longitude(longitude),
                                             data(latitude * longitude, temperature),
                                             albedo_data(std::move(albedo_data)) {}
};

/** Warm the world based on the position of the sun.
 *
 * @param world      The world to warm.
 * @param world_time The time of day, which defines the position of the sun.
 */

/** Warm the world based on the position of the sun.
 *
 * @param world      The world to warm.
 * @param world_time The time of day, which defines the position of the sun.
 */
void radiation(World &world, double world_time) {
    double sun_angle = std::cos(world_time);
    double sun_intensity = 865.0;
    double sun_long = (std::sin(sun_angle) * (world.longitude / 2)) + world.longitude / 2.;
    double sun_lat = world.latitude / 2.;
    double sun_height = 100. + std::cos(sun_angle) * 100.;
    double sun_height_squared = sun_height * sun_height;
    #pragma omp parallel for shared(world)
    for (uint64_t i = 0; i < world.latitude; ++i) {
        for (uint64_t j = 0; j < world.longitude; ++j) {
            // Euclidean distance between the sun and each earth coordinate
            double dist = sqrt((sun_lat - i) * (sun_lat - i) + (sun_long - j) * (sun_long - j) + sun_height_squared);
            world.data[i * world.longitude + j] += \
                                          (sun_intensity / dist) * (1. - world.albedo_data[i * world.longitude + j]);
        }
    }
}

/** Heat radiated to space
 *
 * @param world  The world to update.
 */
void energy_emmision(World &world) {
    #pragma omp parallel for shared(world)
    for (uint64_t i = 0; i < world.latitude * world.longitude; ++i) {
        world.data[i] *= 0.99;
    }
}

/** Heat diffusion
 *
 * @param world  The world to update.
 */
void diffuse(World &world) {
    std::vector<double> tmp = world.data;
    for (uint64_t k = 0; k < 10; ++k) {
        #pragma omp parallel for shared(tmp, world)
        for (uint64_t i = 1; i < world.latitude - 1; ++i) {
            for (uint64_t j = 1; j < world.longitude - 1; ++j) {
                // 5 point stencil
                double center = world.data[i * world.longitude + j];
                double left = world.data[(i - 1) * world.longitude + j];
                double right = world.data[(i + 1) * world.longitude + j];
                double up = world.data[i * world.longitude + (j - 1)];
                double down = world.data[i * world.longitude + (j + 1)];          
                tmp[i * world.longitude + j] = (center + left + right + up + down) / 5.;               
            }
        }
        std::swap(world.data, tmp);
    }
}

/** One integration step at `world_time`
 *
 * @param world      The world to update.
 * @param world_time The time of day, which defines the position of the sun.
 */
void integrate(World &world, double world_time) {
    radiation(world, world_time);
    energy_emmision(world);
    diffuse(world);
}

/** Read a world model from a HDF5 file
 *
 * @param filename The path to the HDF5 file.
 * @return         A new world based on the HDF5 file.
 */
World read_world_model(const std::string &filename) {
    H5::H5File file(filename, H5F_ACC_RDONLY);
    H5::DataSet dataset = file.openDataSet("world");
    H5::DataSpace dataspace = dataset.getSpace();

    if (dataspace.getSimpleExtentNdims() != 2) {
        throw std::invalid_argument("Error while reading the model: the number of dimension must be two.");
    }

    if (dataset.getTypeClass() != H5T_FLOAT or dataset.getFloatType().getSize() != 8) {
        throw std::invalid_argument("Error while reading the model: wrong data type, must be double.");
    }

    hsize_t dims[2];
    dataspace.getSimpleExtentDims(dims, NULL);
    std::vector<double> data_out(dims[0] * dims[1]);
    dataset.read(data_out.data(), H5::PredType::NATIVE_DOUBLE, dataspace, dataspace);
    std::cout << "World model loaded -- latitude: " << (unsigned long) (dims[0]) << ", longitude: "
              << (unsigned long) (dims[1]) << std::endl;
    return World(static_cast<uint64_t>(dims[0]), static_cast<uint64_t>(dims[1]), 293.15, std::move(data_out));
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

/** Write a history of the world temperatures to a HDF5 file
 *
 * @param world_history  Vector of the all worlds to write
 * @param filename       The output filename of the HDF5 file
 */
void write_hdf5(const std::vector <World> &world_history, const std::string &filename) {

    H5::H5File file(filename, H5F_ACC_TRUNC);

    for (uint64_t i = 0; i < world_history.size(); ++i) {
        H5::Group group(file.createGroup("/" + std::to_string(i)));
        write_hdf5(group, "world", {world_history[i].latitude, world_history[i].longitude}, world_history[i].data);
    }
}

/** Simulation of a flat word climate
 *
 * @param num_of_iterations  Number of time steps to simulate
 * @param model_filename     The filename of the world model to use (HDF5 file)
 * @param output_filename    The filename of the written world history (HDF5 file)
 */
void simulate(uint64_t num_of_iterations, const std::string &model_filename, const std::string &output_filename) {

    World world = read_world_model(model_filename);

    const double t_div = world.longitude / 36.0;
    std::vector <World> world_history;
    double checksum = 0;
    auto begin = std::chrono::steady_clock::now();
    for (uint64_t t = 0; t < num_of_iterations; ++t) {
        integrate(world, t / t_div);
        checksum += std::accumulate(world.data.begin(), world.data.end(), 0.0);
        if (!output_filename.empty()) {
            world_history.push_back(world);
            /*std::cout << t << " -- min: " << *std::min_element(world.data.begin(), world.data.end())
                      << ", max: " << *std::max_element(world.data.begin(), world.data.end())
                      << ", avg: " << std::accumulate(world.data.begin(), world.data.end(), 0.0) / world.data.size()
                      << "\n";*/
            checksum += std::accumulate(world.data.begin(), world.data.end(), 0.0);
        }
    }
    if (!output_filename.empty()) {
        write_hdf5(world_history, output_filename);
    }
    auto end = std::chrono::steady_clock::now();
    if (!output_filename.empty()) {
        std::cout << "checksum: " << checksum << std::endl;
    }
    std::cout << "elapsed time: " << (end - begin).count() / 1000000000.0 << " sec" << std::endl;
    std::cout << "checksum: " << checksum << std::endl;
}

/** Main function that parses the command line and start the simulation */
int main(int argc, char **argv) {
    util::ArgParser args(argc, argv);
    int64_t iterations;
    std::string model_filename;
    if (args.cmdOptionExists("--iter")) {
        iterations = std::stoi(args.getCmdOption("--iter"));
        if (iterations < 0) {
            throw std::invalid_argument("iter most be a positive integer (e.g. --iter 100)");
        }
    } else {
        throw std::invalid_argument("You must specify the number of iterations (e.g. --iter 100)");
    }
    if (args.cmdOptionExists("--model")) {
        model_filename = args.getCmdOption("--model");
    } else {
        throw std::invalid_argument("You must specify the model to simulate "
                                    "(e.g. --model world_models/small_model.hdf5)");
    }
    const std::string &output_filename = args.getCmdOption("--out");
    simulate(static_cast<uint64_t>(iterations), model_filename, output_filename);
    return 0;
}
