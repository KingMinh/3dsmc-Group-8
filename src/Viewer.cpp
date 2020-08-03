#include "Viewer.h"

Viewer::Viewer(const ImageSource& i, const Grid& g)
    : image(i), grid(g), buffer(g.voxels.size()), colors(g.voxels.size()), viewer("Carving") {
    viewer.setOffScreenRendering();

    double voxelWidth = grid.x_length / grid.dimension;
	double voxelHeight = grid.y_length / grid.dimension;
	double voxelDepth = grid.z_length / grid.dimension;

    double startX = -grid.x_length / 2;
	double startY = -grid.y_length / 2;
	double startZ = -grid.z_length / 2;

    size_t dim = grid.dimension;
    size_t dim_sq = dim * dim;

    #pragma omp parallel for shared(buffer) schedule(dynamic, 64)
    for (size_t i = 0; i < grid.voxels.size(); i++) {
        size_t x = i / dim_sq;
        size_t y = (i / dim) % dim;
        size_t z = i % dim;
    
        buffer[i] = cv::Vec3f(
            startX + (x - 0.5) * voxelWidth, 
            startY + (y - 0.5) * voxelHeight, 
            startZ + (z - 0.5) * voxelDepth);
    }

}

void Viewer::draw() {
    // TODO: Test different scheduling methods
    #pragma omp parallel for shared(buffer) schedule(dynamic, 64)
    for (size_t i = 0; i < grid.voxels.size(); i++) {

        if (grid.voxels[i]) {
            colors[i] = grid.voxelsColor[i];
        } else {
            buffer[i] = cv::Vec3f::all( std::numeric_limits<float>::quiet_NaN());
        }
    }

    cv::Mat cloudPoints(buffer.size(), 1, CV_32FC3, buffer.data());
    cv::Mat cloudColors(colors.size(), 1, CV_8UC4, colors.data());

    cv::viz::WCloud cloud(cloudPoints, cloudColors);
    cloud.setRenderingProperty(cv::viz::POINT_SIZE, 10);
    
    viewer.showWidget("Cloud", cloud);
    viewer.spinOnce(0);
}