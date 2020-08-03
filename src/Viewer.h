#pragma once

#include <opencv2/opencv.hpp>
#include <opencv2/viz.hpp>

#include "Grid.h"
#include "ImageSource.h"
#include "Marker.h"
#include "Segmentation.h"

class Viewer {
    const ImageSource& image;
    const Grid& grid;

    std::vector<cv::Vec3f> buffer;
    std::vector<uint32_t> colors;

	cv::viz::Viz3d viewer;
public:
    Viewer(const ImageSource& image, const Grid& grid);

    void draw();

};


