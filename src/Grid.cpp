#include "Grid.h"
#include "Trace.h"
#include "Mesh.h"
#include <iostream>
#include <fstream>
#include <string>
#include <fstream>
#include <opencv2/opencv.hpp>

using namespace cv;

// fill list of voxels, set values that are determined by measuring the object (in meters)
// (0,0,0) is the middle of the marker
Grid::Grid(int dim, float x, float y, float z) {
	this->dimension = dim;
	this->x_length = x;
	this->y_length = y;
	this->z_length = z;

	voxels.resize(dim * dim * dim, true);
	voxelsColor.resize(dim * dim * dim, 0xFFFFFFFFU );
}

// there are a lot of duplicate vertices in there, might want to optimize this
bool Grid::WriteMesh(const std::string &filename) {
	std::ofstream outFile(filename);
	if (!outFile.is_open()) return false;

	Mesh m;
	MarchingCubes(*this, m);
	m.WriteOff(outFile);

	return true;
}

bool Grid::WriteMeshColor(const std::string& filename) {
	std::ofstream outFile(filename);
	if (!outFile.is_open()) return false;

	Mesh m;
	MarchingCubes(*this, m);
	m.WriteOffColor(outFile);

	return true;
}

void Grid::Carve(cv::Vec3d t, cv::Vec3d r, cv::Mat mask, cv::Mat cam) {
	// get extrinsic camera matrix.
	// rot + t describes the marker relative to the camera, we want the inverse, so invert the matrix.
	// Inverse of a rotation is the transpose. Inverse of a translation is the negative.
	cv::Affine3d extr{ r };
	extr.linear(extr.linear().t());
	// order of operations is important!
	extr = extr * cv::Affine3d(Matx33d::eye(), -t);
	// get the inverse of the intrinsic camera matrix.
	cv::Affine3d intr{ cam.inv() };

	// cut off all parts of the mesh outside the camera view (using the intrinsic matrix).
	// this is done by clipping the mesh with four planes (corresponding to the four edges of the image).

	// at this point, the origin of the coordinate system is the marker and the relevant stuff is all at -z.
	// So, we need to transform the space accordingly.
	extr = extr.translate(Vec3d(0, 0, -z_length / 2));

	// unproject all corners into "world space" (first from image to camera, then camera to world).
	auto bounds = cv::boundingRect(mask);
	auto c00 = extr * intr * Vec3d(bounds.x, bounds.y, 1);
	auto c01 = extr * intr * Vec3d(bounds.x, bounds.y + bounds.height + 1, 1);
	auto c10 = extr * intr * Vec3d(bounds.x + bounds.width + 1, bounds.y, 1);
	auto c11 = extr * intr * Vec3d(bounds.x + bounds.width + 1, bounds.y + bounds.height + 1, 1);
	auto orig = extr * Vec3d();

	// Get a normal for each clip plane. Order is chosen so that the normal points "inside".
	// Even though we call them normals, they are not unit vectors! (does not matter, we only care about direction).
	auto n_left = c01.cross(c00);
	auto n_right = c10.cross(c11);
	auto n_top = c00.cross(c10);
	auto n_bottom = c11.cross(c01);
	CarveClipPlane(n_left, orig.dot(n_left));
	CarveClipPlane(n_right, orig.dot(n_right));
	CarveClipPlane(n_top, orig.dot(n_top));
	CarveClipPlane(n_bottom, orig.dot(n_bottom));
}

void Grid::CarveClipPlane(cv::Vec3d n, double orig) {
	double voxelWidth = x_length / dimension;
	double voxelHeight = y_length / dimension;
	double voxelDepth = z_length / dimension;

	double startX = -x_length / 2;
	double startY = -y_length / 2;
	double startZ = -z_length / 2;


	auto it = voxels.begin();

	// FIXME: this is a slow algorithm. Faster ones exist.

	for (int i = 0; i < dimension; i++) {
		// center decides whether inside or outside.
		auto x = startX + (i + 0.5) * voxelWidth;
		for (int j = 0; j < dimension; j++) {
			auto y = startY + (j + 0.5) * voxelHeight;
			for (int k = 0; k < dimension; k++, it++) {
				auto z = startZ + (k + 0.5) * voxelDepth;

				cv::Vec3d center(x, y, z);
				auto dist = n.dot(center);
				if (dist < orig) {
					*it = false;
				}
			}
		}
	}
}



void Grid::CarveMask(InputArray tvec, InputArray rvec, Mat mask, InputArray cameraMatrix, InputArray distCoeffs) {
	double voxelWidth = x_length / dimension;
	double voxelHeight = y_length / dimension;
	double voxelDepth = z_length / dimension;

	double startX = -x_length / 2;
	double startY = -y_length / 2;
	double startZ = -z_length / 2;

	//std::cout << "mask size: " << mask.size() << std::endl;
	auto it = voxels.begin();

	for (int i = 0; i < dimension; i++) {
		// center decides whether inside or outside.
		auto x = startX + (i + 0.5) * voxelWidth;
		for (int j = 0; j < dimension; j++) {
			auto y = startY + (j + 0.5) * voxelHeight;
			for (int k = 0; k < dimension; k++, it++) {
				if (!*it) continue;
				auto z = startZ + (k + 0.5) * voxelDepth;

				// project voxel centers into image
				cv::Point3f center(x, y, z);
				std::vector<Point3f> centerPoint;
				centerPoint.push_back(center);
				std::vector<Point2f> imagePoint;
				projectPoints(centerPoint, rvec, tvec, cameraMatrix, distCoeffs, imagePoint);
				int xs = imagePoint[0].x;
				int ys = imagePoint[0].y;
				if (xs < 0 || ys < 0 || xs >= mask.size().width || ys >= mask.size().height)
					continue;

				// compare corresponding pixel to mask
				if (mask.at<unsigned char>(ys, xs) == 0)
					*it = false;
					
					
			}
		}
	}
}

void Grid::CarveMaskColor(InputArray tvec, InputArray rvec, Mat mask, InputArray cameraMatrix, InputArray distCoeffs, Mat image) {
	double voxelWidth = x_length / dimension;
	double voxelHeight = y_length / dimension;
	double voxelDepth = z_length / dimension;

	double startX = -x_length / 2;
	double startY = -y_length / 2;
	double startZ = -z_length / 2;

	// TODO: Test different scheduling methods
	#pragma omp parallel for schedule(dynamic, 2)
	for (int i = 0; i < dimension; i++) {
		// center decides whether inside or outside.
		auto x = startX + (i + 0.5) * voxelWidth;
		for (int j = 0; j < dimension; j++) {
			auto y = startY + (j + 0.5) * voxelHeight;
			for (int k = 0; k < dimension; k++) {
				auto voxel = voxels[k + dimension * (j + i * dimension)];

				if (!voxel) continue;

				auto z = startZ + (k + 0.5) * voxelDepth;

				// project voxel centers into image
				cv::Point3f center(x, y, z);
				std::vector<Point3f> centerPoint;
				centerPoint.push_back(center);
				std::vector<Point2f> imagePoint;
				projectPoints(centerPoint, rvec, tvec, cameraMatrix, distCoeffs, imagePoint);
				int xs = imagePoint[0].x;
				int ys = imagePoint[0].y;
				if (xs < 0 || ys < 0 || xs >= mask.size().width || ys >= mask.size().height) {
					voxel = false;
					continue;
				}
				// compare corresponding pixel to mask
				if (mask.at<unsigned char>(ys, xs) == 0) {
					voxel = false;
				}

				//image is in BGR notation	
				voxelsColor[k + dimension * (j + i * dimension)] = (image.at<Vec3b>(ys,xs).val[2]) |
						(image.at<Vec3b>(ys,xs).val[1] << 8) |
						(image.at<Vec3b>(ys,xs).val[0] << 16) ;
			}
		}
	}
}

//// code for debugging the frustum and voxel container while carving.
//	{
//		std::ofstream fs{"output/cube.obj"};
//		for (auto x : {-1, 1}) {
//			for (auto y : {-1, 1}) {
//				for (auto z : {-1, 1}) {
//					fs << "v " << x * x_length * 0.5 << ' ' << y * y_length * 0.5 << ' ' << z * z_length * 0.5 << '\n';
//				}
//			}
//		}
//		fs << "l 1 2 4 3 1 5 6 8 7 3\nl 5 7\nl 2 6\nl 4 8\n";
//	}
//	{
//		static int frame = 0;
//		std::string fname = "output/frust";
//		fname += std::to_string(frame++);
//		std::ofstream fs{fname + ".obj"};
//
//		fs << "v " << orig[0] << ' ' << orig[1] << ' ' << orig[2] << '\n';
//		fs << "v " << c00[0] << ' ' << c00[1] << ' ' << c00[2] << '\n';
//		fs << "v " << c01[0] << ' ' << c01[1] << ' ' << c01[2] << '\n';
//		fs << "v " << c10[0] << ' ' << c10[1] << ' ' << c10[2] << '\n';
//		fs << "v " << c11[0] << ' ' << c11[1] << ' ' << c11[2] << '\n';
//		fs << "l 1 2\nl 1 3\nl 1 4\nl 1 5\n";
//	}

