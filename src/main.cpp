#include <iostream>
#include <string>
#include <string_view>
#include <sstream>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include "ImageSource.h"
#include "Marker.h"
#include "Segmentation.h"
#include "Grid.h"

using namespace cv;
using namespace std::literals;

bool ends_with(const std::string &str, const std::string &end) {
	if (str.length() >= end.length()) {
		return str.compare(str.length() - end.length(), end.length(), end) == 0;
	}
	else {
		return false;
	}
}

int main(int argc, char **argv) {

	char *inFile = nullptr, *outFile = nullptr, *paramsFile = nullptr;
	for (int i = 1; i < argc; i++) {
		std::string_view arg{ argv[i] };
		if (arg == "--in"sv && i + 1 < argc)
			inFile = argv[++i];
		else if (arg == "--out"sv && i + 1 < argc)
			outFile = argv[++i];
		else if (arg == "--params"sv && i + 1 < argc)
			paramsFile = argv[++i];
		else
			std::cerr << "Unrecognized argument: " << arg << std::endl;
	}

	if (!inFile || !outFile || !paramsFile) {
		std::cerr << "Usage: 3dsmc [OPTIONS] --in <PATH> --out <PATH> --params <PATH>" << std::endl;
		return 1;
	}

	// create voxel grid
	Grid grid(80, 0.2f, 0.2f, 0.2f);

	std::string directory = std::string(inFile);
	std::string imagePath = directory + "/*.jpg";
	std::vector<std::string> filenames;
	glob(imagePath, filenames);

	std::string backgroundPath;
	std::cout << "Please enter the background path: " << std::endl;
	std::cin >> backgroundPath;
	std::unique_ptr<Segmentation> segmentation = std::make_unique<CleanplateSegmentation>(backgroundPath);

	namedWindow("markers", WINDOW_NORMAL);
	namedWindow("segmentation", WINDOW_NORMAL);
	resizeWindow("markers", 1920, 1080);
	resizeWindow("segmentation", 1920, 1080);

	for (size_t i = 0; i < filenames.size(); i++) {
		std::unique_ptr<ImageSource> image;

		image = std::make_unique<StillImageSource>(filenames[i], paramsFile);

		Marker marker(*image);
		segmentation->update(*image);

		imshow("markers", marker.visualize());
		imshow("segmentation", segmentation->get_mask());

		std::string outputPath = std::string{ outFile };
		std::stringstream ss;
		ss << i;

		if (!marker.translationVectors.empty()) {

			imwrite(outputPath + "/marker" + ss.str() + ".png", marker.visualize());

			std::cout << "Processing image " << ss.str() << std::endl;
			grid.CarveMaskColor(marker.translationVectors[0], marker.rotationVectors[0], segmentation->get_mask(),
				image->get_camera_matrix(), image->get_distortion_coefficients(), image->get_frame());

		}
		else {
			std::cout << "Could not find marker in image " << ss.str() << std::endl;
		}

		char c = static_cast<char>(waitKey(1));
		// ESC Key
		if (c == 27) break;
	}

	// write mesh file
	std::stringstream ss;
	ss << outFile << "/mesh.off";
	if (!grid.WriteMeshColor(ss.str())) {
		std::cout << "Failed to write mesh!\nCheck file path!" << std::endl;
		return -1;
	}

	return 0;
}
