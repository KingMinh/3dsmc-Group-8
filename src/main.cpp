#include <iostream>
#include <string>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <omp.h>

#include "Args.h"
#include "ImageSource.h"
#include "Marker.h"
#include "Segmentation.h"
#include "Grid.h"
#include "Viewer.h"
#include "Trace.h"

using namespace cv;

bool ends_with(const std::string& str, const std::string& end) {
	if (str.length() >= end.length()) {
		return str.compare(str.length() - end.length(), end.length(), end) == 0;
	}
	else {
		return false;
	}
}

int main(int argc, char** argv) {
	Arguments args;

	std::unique_ptr<ImageSource> image;

	if (parse_args(args, argc, argv) < 0) {
		std::cerr << "Failed to parse arguments, call with --help for more information" << std::endl;
		return -1;
	}

	if (args.input.index() == 0) {
		auto& file = std::get<std::string>(args.input);

		if (ends_with(file, ".png") || ends_with(file, ".jpg")) {
			image = std::make_unique<StillImageSource>(file, args.config);
		}
		else if (ends_with(file, ".mp4")) {
			image = std::make_unique<VideoImageSource>(file, args.config);
		}
		else {
			std::cerr << "Unrecognised input type: " << file << std::endl;
			return -1;
		}
	}
	else if (args.input.index() == 1) {
		image = std::make_unique<VideoImageSource>(std::get<int>(args.input), args.config);
	}
	else {
		return -1;
	}

	if (!image || !image->is_open()) {
		std::cerr << "error opening input file" << std::endl;
		return -1;
	}

	std::unique_ptr<Segmentation> segmentation;

	switch (args.mode) {
	case SegmentMode::FirstFrame:
		segmentation = std::make_unique<CleanplateSegmentation>(args.cleanPlate.value());
		break;
	case SegmentMode::ChromaBlue:
		segmentation = ChromaSegmentation::Blue();
		break;
	case SegmentMode::ChromeGreen:
		segmentation = ChromaSegmentation::Green();
		break;
	case SegmentMode::ChromaWhite:
		segmentation = ChromaSegmentation::White();
		break;
	case SegmentMode::Watershed:
		segmentation = std::make_unique<WatershedSegmentation>();
		break;
	default:
		break;
	}

	if (!segmentation) {
		std::cerr << "no background segmentation mode specified" << std::endl;
		return -1;
	}

	omp_set_num_threads(omp_get_max_threads());

	// create voxel grid
	Grid grid(64, 0.1f, 0.1f, 0.05f);
	Viewer viewer(*image, grid);

	bool has_next = false;

	namedWindow("markers", WINDOW_NORMAL);
	namedWindow("segmentation", WINDOW_NORMAL);
	resizeWindow("markers", 1920, 1080);
	resizeWindow("segmentation", 1920, 1080);
	MarkerTracker markerTracker;
	int frame_counter = 0;
	do {
		Trace fullFrame("frame " + std::to_string(frame_counter++));

		Trace traceMarker("Marker");
		Marker marker(*image, args.markerLength);
		traceMarker.end();		

		Trace::call("Segmentation", [&]() { segmentation->update(*image); });

		imshow("markers", marker.visualize());
		imshow("segmentation", segmentation->get_mask());

		if (auto location = markerTracker.getFirstMarkerLoc(marker)) {
			Trace carving("carving");
			grid.CarveMaskColor(location->translation, location->rotation, segmentation->get_mask(),
			               image->get_camera_matrix(), image->get_distortion_coefficients(), image->get_frame());
		}

		{
			Trace draw("draw");
			viewer.draw();
		}

		fullFrame.end();

		char c = static_cast<char>(waitKey(1));
		// ESC Key
		if (c == 27) break;
	} while ((has_next = image->next()));

	// Quit immediately if video/stream was stopped via ESC key
	if (!has_next)
		waitKey(0);

	if (!grid.WriteMeshColor(args.get_output_filepath("mesh.off"))) {
		std::cout << "Failed to write mesh!\nCheck file path!" << std::endl;
		return -1;
	}
	

	return 0;
}
