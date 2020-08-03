#pragma once

#include <vector>
#include <array>
#include <iosfwd>
#include "Grid.h"


struct RGB {
	uint8_t blue;
	uint8_t green;
	uint8_t red;
	uint8_t alpha;
};

class Mesh {
private:
	struct vertex {
		float x, y, z;
	};
	struct triangle {
		size_t v1, v2, v3;
	};

	std::vector<vertex> verts;
	std::vector<triangle> faces;
	std::vector<RGB> facesColor;
	std::vector<RGB> vertsColor;

public:
	size_t AddVertex(float x, float y, float z);

	size_t AddVertex(std::array<float, 3> &v) {
		return AddVertex(v[0], v[1], v[2]);
	}

	void AddFace(size_t v1, size_t v2, size_t v3);

	void AddFaceColor(uint8_t r, uint8_t g, uint8_t b);
	void AddVertColor(uint8_t r, uint8_t g, uint8_t b);

	void WriteOff(std::ostream &out);
	void WriteOffColor(std::ostream& out);
};

void MarchingCubes(const Grid &g, Mesh &m);
