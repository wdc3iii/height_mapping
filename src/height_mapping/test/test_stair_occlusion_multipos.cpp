#include <iostream>
#include <fstream>
#include "HeightMapping.hpp"

const float STAIR_WIDTH = 2.0f;
const float STAIR_LENGTH = 3.0f;
const float STAIR_SLOPE = 0.4f;

int main() {
  height_mapping::Params P;
  P.Wb = 200; P.Hb = 200; P.res = 0.05;   // 10 m × 10 m big grid
  P.Wq = 100; P.Hq = 100; P.res_q = 0.05; // subgrid same size/res
  P.min_support = 4;
  height_mapping::HeightMap map(P);

  // Define lists of robot positions
  // std::vector<float> robot_x_list = {0.0f, 5.0f, 10.0f, 15.0f, 20.0f};
  std::vector<float> robot_x_list = {0.0f, 0.2f, 0.4f, 0.6f, 2.2f};
  // std::vector<float> robot_x_list = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
  std::vector<float> robot_y_list = {0.0f, 0.0, 0.0, 0.0, 0.0};
  std::vector<float> robot_z_list = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
  std::vector<float> robot_yaw_list = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f};

  // Loop through each robot position
  for (int pos_idx = 0; pos_idx < 5; ++pos_idx) {
    float robot_x = robot_x_list[pos_idx];
    float robot_y = robot_y_list[pos_idx];
    float robot_z = robot_z_list[pos_idx];
    float robot_yaw = robot_yaw_list[pos_idx];
    map.recenterIfNeeded(robot_x, robot_y, robot_z);
    std::string move_raw_filename = "post_move_raw" + std::to_string(pos_idx) + ".png";
    std::string move_filled_filename = "post_move_filled" + std::to_string(pos_idx) + ".png";
    cv::Mat post_move_raw, post_move_filled;
    height_mapping::SubgridMeta post_move_meta;
    map.generateSubgrid(robot_x, robot_y, robot_yaw, post_move_raw, post_move_filled, post_move_meta);
    height_mapping::save_png(post_move_raw, move_raw_filename);
    height_mapping::save_png(post_move_raw, move_filled_filename);

    std::cout << "Processing position " << pos_idx << ": x=" << robot_x 
              << ", y=" << robot_y << ", z=" << robot_z 
              << ", yaw=" << robot_yaw << std::endl;

    map.ensureOrigin(robot_x, robot_y, robot_z);
    std::vector<height_mapping::Point3f> pts;
    float max_x = -1000.0f, min_x = 1000.0f;
    float max_y = -1000.0f, min_y = 1000.0f;
    
    for (float r = -5.0; r <= 5.0; r += 0.01) {
      for (float c = -5.0; c < 5.0; c += 0.01) {
        float x = r;
        float y = c;
        float z = STAIR_SLOPE * x;               // shallow slope
        if (abs(x) < 2.2f && y < 1.2f && y > 0.8f) {
          if (!(abs(x) < 2.1f && y < 1.1f & y > 0.9f)) {
            for (float j = 0.0f; j <= 1.0001f; j += 0.01f) {
              pts.push_back({x, y, z * j + (1 - j) * 1.0f}); // vertical pole
            }
          }
        } else if (abs(x) < STAIR_LENGTH && abs(y) < STAIR_WIDTH) {
          if (static_cast<int>(std::floor(x / 0.5f)) % 2 == 0) {
            float height = std::floor(x) * STAIR_SLOPE;
            pts.push_back({x,y,height});
          }
        } else if (abs(x) > STAIR_LENGTH) {
          float height = (x > 0) ? STAIR_LENGTH * STAIR_SLOPE : -STAIR_LENGTH * STAIR_SLOPE;
          pts.push_back({x,y,height});
        } else {
          pts.push_back({x,y,z});
        }

        if (x > max_x) max_x = x;
        if (x < min_x) min_x = x;
        if (y > max_y) max_y = y;
        if (y < min_y) min_y = y;
      }
    }
    map.ingestPoints(pts);
    

    // Query subgrid at current robot position
    cv::Mat raw, filled;
    height_mapping::SubgridMeta meta;
    map.generateSubgrid(robot_x, robot_y, robot_yaw, raw, filled, meta);

    float center = filled.at<float>(raw.rows/2, raw.cols/2);
    std::cout << "Center height ~ " << center << " m\n";

    // Save with indexed filenames to avoid overwriting
    std::string raw_filename = "sub_raw_pos" + std::to_string(pos_idx) + ".png";
    std::string filled_filename = "sub_filled_pos" + std::to_string(pos_idx) + ".png";
    
    height_mapping::save_png(raw, raw_filename);
    height_mapping::save_png(filled, filled_filename);
    
    std::cout << "Wrote " << raw_filename << " and " << filled_filename << "\n";
    std::cout << "min_x, max_x, min_y, max_y = "
              << min_x << ", " << max_x << ", " << min_y << ", " << max_y << "\n\n";
  }
  return 0;
}