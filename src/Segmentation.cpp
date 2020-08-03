#include "Segmentation.h"

using namespace cv;

ChromaSegmentation::ChromaSegmentation(cv::Scalar lower, cv::Scalar upper)
		: lower(std::move(lower)), upper(std::move(upper)) {}

void ChromaSegmentation::update(ImageSource &image) {
	cv::Mat input_hsv;
	cv::cvtColor(image.get_frame(), input_hsv, cv::COLOR_BGR2HSV);
	cv::GaussianBlur(input_hsv, input_hsv, cv::Size(11, 11), 0, 0);
	// TODO: This is rudimentary only. Tweak values & potentially use a better algorithm
	cv::inRange(input_hsv, lower, upper, mask);
	cv::bitwise_not(mask, mask);
	cv::Mat element = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(11, 11));
	cv::morphologyEx(mask, mask, MORPH_CLOSE, element);
}

std::unique_ptr<ChromaSegmentation> ChromaSegmentation::Green() {
	return std::make_unique<ChromaSegmentation>(cv::Scalar(50, 120, 180), cv::Scalar(70, 255, 255));
}

std::unique_ptr<ChromaSegmentation> ChromaSegmentation::Blue() {
	return std::make_unique<ChromaSegmentation>(cv::Scalar(90, 80, 100), cv::Scalar(130, 255, 255));
}

std::unique_ptr<ChromaSegmentation> ChromaSegmentation::White() {
	return std::make_unique<ChromaSegmentation>(cv::Scalar(0, 0, 180), cv::Scalar(255, 25, 255));
}

CleanplateSegmentation::CleanplateSegmentation(const std::string &file) {
	firstFrame = cv::imread(file, 1);

	if (firstFrame.empty()) {
		std::cerr << "Error opening cleanplate image" << std::endl;
		std::exit(1);
	}
}

void CleanplateSegmentation::update(ImageSource &image) {
	cv::Mat dst, gray;
	cv::absdiff(image.get_frame(), firstFrame, dst);
	cv::cvtColor(dst, gray, cv::COLOR_RGB2GRAY);

	// Tweaks probably dependent on lighting & background
	int filterWidth, filterHeight;
	filterWidth = 121;
	filterHeight = filterWidth;
	int thresholdVal = 35;
	cv::GaussianBlur(gray, gray, cv::Size(filterWidth, filterHeight), 0);
	//medianBlur(gray, gray, 121);
	cv::threshold(gray, mask, thresholdVal, 255, cv::THRESH_BINARY);
}

WatershedSegmentation::WatershedSegmentation() = default;

void WatershedSegmentation::update(ImageSource &image) {
	// depends strongly on color of object and reflections
	//works with white background
	cv::Mat inputImageCopy = image.get_frame().clone();
	// Change the background from white to black, since that will help later to extract
	// better results during the use of Distance Transform
	Mat blackwhite;
	cvtColor(inputImageCopy, blackwhite, COLOR_BGR2GRAY);
	threshold(blackwhite, blackwhite, 0, 255, THRESH_BINARY | THRESH_OTSU);
	#pragma omp parallel for
	for (int i = 0; i < inputImageCopy.rows; i++) {
		for (int j = 0; j < image.get_frame().cols; j++) {
			if (blackwhite.at<uchar>(i, j) == 255) {
				inputImageCopy.at<Vec3b>(i, j)[0] = 0;
				inputImageCopy.at<Vec3b>(i, j)[1] = 0;
				inputImageCopy.at<Vec3b>(i, j)[2] = 0;
			}
		}
	}

	Mat kernel = (Mat_<float>(3, 3) <<
	                                1, 1, 1,
			1, -8, 1,
			1, 1, 1); // an approximation of second derivative, a quite strong kernel
	// do the laplacian filtering as it is
	Mat imgLaplacian;
	filter2D(inputImageCopy, imgLaplacian, CV_32F, kernel);
	Mat sharp;
	inputImageCopy.convertTo(sharp, CV_32F);
	Mat imgResult = sharp - imgLaplacian;
	// convert back to 8bits gray scale
	imgResult.convertTo(imgResult, CV_8UC3);
	imgLaplacian.convertTo(imgLaplacian, CV_8UC3);
	// imshow( "Laplace Filtered Image", imgLaplacian );
	//imshow("New Sharped Image", imgResult);
	// Create binary image from source image
	Mat bw;
	cvtColor(imgResult, bw, COLOR_BGR2GRAY);
	threshold(bw, bw, 40, 255, THRESH_BINARY | THRESH_OTSU);
	//imshow("Binary Image", bw);


	// Perform the distance transform algorithm
	Mat dist;
	distanceTransform(bw, dist, DIST_L2, 3);
	// Normalize the distance image for range = {0.0, 1.0}
	normalize(dist, dist, 0, 1.0, NORM_MINMAX);
	//imshow("Distance Transform Image", dist);
	// Threshold to obtain the peaks
	// This will be the markers for the foreground objects
	threshold(dist, dist, 0.4, 1.0, THRESH_BINARY);
	// Dilate a bit the dist image
	Mat kernel1 = Mat::ones(3, 3, CV_8U);
	dilate(dist, dist, kernel1);
	//imshow("Peaks", dist);


	//find contours
	Mat dist_8u;
	dist.convertTo(dist_8u, CV_8U);
	// Find total markers
	std::vector<std::vector<Point> > contours;
	findContours(dist_8u, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
	// Create the marker image for the watershed algorithm
	Mat markers = Mat::zeros(dist.size(), CV_32S);
	// Draw the foreground markers
	for (size_t i = 0; i < contours.size(); i++) {
		drawContours(markers, contours, static_cast<int>(i), Scalar(static_cast<int>(i) + 1), -1);
	}
	// Draw the background marker
	circle(markers, Point(10, 10), 5, Scalar(255), -1);

	//Mat tmp;
	//normalize(markers, tmp, 255, 0, NORM_MINMAX, CV_8UC1);
	//imshow("Markers", tmp);


	// Perform the watershed algorithm
	watershed(imgResult, markers);
	Mat mark;
	markers.convertTo(mark, CV_8U);
	bitwise_not(mark, mark);


	// Generate random colors,
	std::vector<Vec3b> randomColors;
	for (size_t i = 0; i < contours.size(); i++) {
		int b = theRNG().uniform(0, 256);
		int g = theRNG().uniform(0, 256);
		int r = theRNG().uniform(0, 256);
		randomColors.emplace_back((uchar) b, (uchar) g, (uchar) r);
	}
	// Create the result image
	Mat dst = Mat::zeros(markers.size(), CV_8UC3);
	// Fill labeled objects with random colors, if there are several objects in the scene, which you want so seperatly analyze
	//for now all detected objects in the forground will have the color white
	#pragma omp parallel for
	for (int i = 0; i < markers.rows; i++) {
		for (int j = 0; j < markers.cols; j++) {
			int index = markers.at<int>(i, j);
			if (index > 0 && index <= static_cast<int>(contours.size())) {

				dst.at<Vec3b>(i, j) = Vec3b(255, 255,
				                            255);    //if no color seperation is needed white is used else use randomColors[index - 1];
			}

		}
	}
	mask = dst.clone();
}
