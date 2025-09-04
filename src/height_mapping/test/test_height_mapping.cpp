#include "HeightMapping.hpp"
#include <iostream>

int main() {
  height_mapping::Params P; height_mapping::HeightMap m(P);
  m.ensureOrigin(0,0);
  std::vector<height_mapping::Point3f> pts = {{0.0f,0.0f,0.1f},{0.05f,0.0f,0.12f}};
  m.ingestPoints(pts);
  m.recenterIfNeeded(0.0, 0.0);
  cv::Mat raw, filled; height_mapping::SubgridMeta meta;
  m.generateSubgrid(0.0,0.0,0.0, raw, filled, meta);
  std::cout << "ok " << raw.at<float>(raw.rows/2, raw.cols/2) << "\n";
}
