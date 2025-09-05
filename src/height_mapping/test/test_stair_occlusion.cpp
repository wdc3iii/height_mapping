#include <iostream>
#include <fstream>
#include "HeightMapping.hpp"

int main() {
  height_mapping::Params P;
  P.Wb = 200; P.Hb = 200; P.res = 0.05;   // 10 m × 10 m big grid
  P.Wq = 100; P.Hq = 100; P.res_q = 0.05; // subgrid same size/res
  P.min_support = 4;
  height_mapping::HeightMap map(P);

  // Robot at origin; ensure origin and insert a little ramp + step
  float robot_x = 0.0; float robot_y = 0.0; float robot_yaw = 0.0;
  map.ensureOrigin(robot_x, robot_y);
  std::vector<height_mapping::Point3f> pts;
  float max_x = -1000.0f, min_x = 1000.0f;
  float max_y = -1000.0f, min_y = 1000.0f;
  for (int k = 0; k < 200000; ++k) {
    float x = 10.0f*(float(rand())/RAND_MAX - 0.5);
    float y = 10.0f*(float(rand())/RAND_MAX - 0.5);
    float z = 0.5f * x;               // shallow slope
    // if (x > 1.0f && x < 1.2f) z -= 0.5f;     // small “step down”
    if (abs(x) < 2.2f && y < 1.2f && y > 0.8f) {
      if (!(abs(x) < 2.1f && y < 1.1f & y > 0.9f)) {
        for (float j = 0.0f; j <= 1.0001f; j += 0.01f) {
          pts.push_back({x, y, z * j + (1 - j) * 1.0f}); // vertical pole
        }
      }
    } else if (abs(x) < 3.0f && abs(y) < 2.0f) {
      if (static_cast<int>(std::floor(x / 0.5f)) % 2 == 0) {
        float height = std::floor(x) * 0.5f;
        pts.push_back({x,y,height});
      }
    } else if (abs(x) > 3.0f) {
      float height = (x > 0) ? 3.0f * 0.5f : -3.0f * 0.5f;
      pts.push_back({x,y,height});
    } else {
      pts.push_back({x,y,z});
    }

    if (x > max_x) max_x = x;
    if (x < min_x) min_x = x;
    if (y > max_y) max_y = y;
    if (y < min_y) min_y = y;
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
  std::cout << "Wrote sub_raw.png and sub_filled.png\nmin_x, max_x, min_y, max_y = "
            << min_x << ", " << max_x << ", " << min_y << ", " << max_y << "\n";
  return 0;
}
