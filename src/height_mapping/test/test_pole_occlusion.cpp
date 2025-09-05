#include <iostream>
#include <fstream>
#include "HeightMapping.hpp"

int main() {
  height_mapping::Params P;
  P.Wb = 200; P.Hb = 200; P.res = 0.05;   // 10 m × 10 m big grid
  P.Wq = 100; P.Hq = 100; P.res_q = 0.05; // subgrid same size/res
  height_mapping::HeightMap map(P);

  // Robot at origin; ensure origin and insert a little ramp + step
  float robot_x = 0.0; float robot_y = 0.0; float robot_yaw = 0.0;
  map.ensureOrigin(robot_x, robot_y);
  std::vector<height_mapping::Point3f> pts;
  for (int k = 0; k < 200000; ++k) {
    float x = -3.0f + 6.0f*float(rand())/RAND_MAX;
    float y = -3.0f + 6.0f*float(rand())/RAND_MAX;
    float z = 0.2f + 0.25f * x;               // shallow slope
    // if (x > 1.0f && x < 1.2f) z -= 0.5f;     // small “step down”
    if (y < 1.2f && y > 0.8f && x < 1.2f) {
      if (x < 1.1f) {
        continue; // occlusion
      } else {
        for (float j = 0.0f; j <= 1.0001f; j += 0.1f) {
          // pts.push_back({x, y, z * j + (1 - j) * 1.0f}); // vertical pole
        }
      }
    }
    pts.push_back({x,y,z});
  }
  map.ingestPoints(pts);

  // Query subgrid at robot (0,0,0 yaw)
  cv::Mat raw, filled;
  height_mapping::SubgridMeta meta;
  map.generateSubgrid(robot_x, robot_y, robot_yaw, raw, filled, meta);

  float center = filled.at<float>(raw.rows/2, raw.cols/2);
  std::cout << "Center height ~ " << center << " m\n";

  height_mapping::save_png(raw,    "sub_raw.png");
  height_mapping::save_png(filled, "sub_filled.png");
  std::cout << "Wrote sub_raw.png and sub_filled.png\n";
  return 0;
}
