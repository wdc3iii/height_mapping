#pragma once
#include <vector>
#include <cstdint>
#include <cmath>
#include <mutex>
#include <limits>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>  // connectedComponents

namespace height_mapping {

struct Point3f { float x,y,z; };

struct SubgridMeta {
  // bottom-left of subgrid in world (map) coords, and yaw (radians)
  double origin_x{0.0}, origin_y{0.0}, yaw{0.0};
  float  resolution{0.05f};
  int width{0}, height{0};
};

struct Params {
  // Big axis-aligned grid (ring buffer)
  double res = 0.05;
  int Wb = 400, Hb = 400;
  double max_h = 2.0;
  double z_min = -1.0, z_max = 2.0;
  double drop_thresh = 0.07;
  int    min_support = 4;
  double shift_thresh = 0.5;     // meters, recenter threshold

  // Subgrid generation
  int    Wq = 200, Hq = 200;
  double res_q = 0.05;
  double occluded_fraction_threshold = 0.5;  // 0..1
};

class HeightMap {
public:
  explicit HeightMap(const Params& p);
  ~HeightMap() = default;

  // (Re)initialize big grid and internal buffers
  void reset();

  // Set/ensure initial origin so robot is centered on big grid
  void ensureOrigin(double robot_x, double robot_y);
  bool haveOrigin() const { return have_origin_; }

  // Recenter the big grid (ring buffer shifts) if robot deviates enough
  void recenterIfNeeded(double robot_x, double robot_y);

  // Ingest a batch of points already in the world/map frame
  // (fastLIO cloud → TF → points → call this)
  void ingestPoints(const std::vector<Point3f>& pts);

  // Generate a robot-centered, robot-aligned subgrid (raw + filled)
  // Returns sub_raw and sub_filled; fills out meta.
  void generateSubgrid(double robot_x, double robot_y, double robot_yaw,
                       cv::Mat& sub_raw, cv::Mat& sub_filled, SubgridMeta& meta) const;

  // Optional: expose big-grid debug snapshots (thread-safe copies)
  void snapshotBig(cv::Mat& height, cv::Mat& known, cv::Mat& occluded) const;

  // Accessors for params
  const Params& params() const { return params_; }

private:
  inline size_t idxRB(int i, int j) const noexcept {
    const int ii = (i + start_i_) % Hb_;
    const int jj = (j + start_j_) % Wb_;
    return static_cast<size_t>(ii) * static_cast<size_t>(Wb_) + static_cast<size_t>(jj);
  }

  void shiftRingBuffer_(int si, int sj);

  // Brushfire fill (occluded clusters) on a subgrid
  void brushfireFill_(const cv::Mat& sub_raw, const cv::Mat& sub_occ, const cv::Mat& sub_edge,
                      cv::Mat& sub_filled) const;

  // Bilinear sampling from big grid → one subgrid cell
  void sampleBilinear_(double wx, double wy,
                       float& h_val, uint8_t& occ_val, uint8_t& edge_touch) const;

private:
  Params params_;

  // Big grid storage (ring buffer)
  int Wb_, Hb_;
  double res_;
  double max_h_, zmin_, zmax_, drop_thresh_;
  int min_support_;
  double shift_thresh_;

  mutable std::mutex m_;
  std::vector<float>   height_b_;
  std::vector<uint8_t> known_b_;
  std::vector<uint8_t> occ_b_;
  std::vector<float>   stamp_b_;
  int start_i_{0}, start_j_{0};
  double origin_x_{0.0}, origin_y_{0.0};
  bool have_origin_{false};

  // Per-scan temp buckets (reused each call to ingestPoints)
  std::vector<float> temp_min_;
  std::vector<int>   temp_cnt_;
};

} // namespace lem
