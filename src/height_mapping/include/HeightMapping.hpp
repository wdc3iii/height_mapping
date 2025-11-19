#pragma once
#include <vector>
#include <cstdint>
#include <mutex>
#include <atomic>
#include <limits>
#include <cmath>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>


namespace height_mapping {

struct Point3f { float x,y,z; };

struct SubgridMeta {
  double origin_x{0.0}, origin_y{0.0}, yaw{0.0};
  float  resolution{0.05f};
  int width{0}, height{0};
};

struct Params {
  // Big grid
  double res = 0.05;
  int    Wb = 400, Hb = 400;
  double max_h = 2.0;
  double z_min = -10.0, z_max = 10.0;
  double drop_thresh = 0.07;
  int    min_support = 4;
  double shift_thresh = 0.5;

  // Subgrid
  int    Wq = 200, Hq = 200;
  double res_q = 0.05;
  bool   subgrid_bilinear_interp = true; // Use bilinear interpolation (vs nearest-neighbor)

  // --- NEW: histogram / connectivity settings ---
  double z_hist_bin = 0.02;      // bin size [m], choose <= δ/2
  double z_connect_delta = 0.05; // δ: tolerate gaps up to this [m]
  int    z_bin_min_count = 2;    // min bin occupancy to count as "present"
  double z_shift_thresh = 0.1;   // min z movement before recentering bins [m]
};

void save_png(const cv::Mat& m, const std::string& path);

class HeightMap {
public:
  explicit HeightMap(const Params& p);
  void reset();

  void ensureOrigin(double robot_x, double robot_y, double robot_z);
  void recenterIfNeeded(double robot_x, double robot_y, double robot_z);

  void ingestPoints(const std::vector<Point3f>& pts);

  void generateSubgrid(double rx, double ry, double rYaw,
                       cv::Mat& sub_raw, cv::Mat& sub_filled, SubgridMeta& meta) const;

  void snapshotBig(cv::Mat& H, cv::Mat& K, cv::Mat& O) const;
  bool haveOrigin() const { return have_origin_; }

  // Get big map metadata for point cloud conversion
  SubgridMeta getBigMapMeta() const;

private:
  inline size_t idxRB(int i, int j) const noexcept {
    const int ii = (i + start_i_) % Hb_;
    const int jj = (j + start_j_) % Wb_;
    return static_cast<size_t>(ii) * static_cast<size_t>(Wb_) + static_cast<size_t>(jj);
  }

  void shiftRingBuffer_(int si, int sj);


  // --- NEW: per-cell z-histogram aggregator (robot-relative) ---
  struct ZAgg {
    std::vector<uint8_t> bins; // size B_
    int min_bin;               // lowest non-empty bin index (or B_ if none)
    int top_conn_from_min;     // highest δ-connected bin from min_bin
    float h_min;               // cached meters (absolute world coords)
    float h_conn_max;          // cached meters (absolute world coords)
    ZAgg() : min_bin(0), top_conn_from_min(-1), h_min(0), h_conn_max(0) {}
  };

  inline void zaggInit_(ZAgg& a) const {
    a.bins.assign(B_, 0u);
    a.min_bin = B_;
    a.top_conn_from_min = -1;
    a.h_min = static_cast<float>(robot_z_ + max_h_);
    a.h_conn_max = static_cast<float>(robot_z_ + max_h_);
  }

  inline void zaggInsert_(ZAgg& a, float z); // update bins & caches
  void recenterHistogramBounds_(double new_robot_z); // recenter z bins if needed
  void cycleBins_(ZAgg& agg, int shift_bins); // cycle bins up/down by shift_bins

private:
  // Params & sizes
  Params params_;
  int    Wb_, Hb_;
  double res_;
  double max_h_, zmin_, zmax_, drop_thresh_;
  int    min_support_;
  double shift_thresh_;

  // Histogram config (derived)
  int    B_;             // number of bins
  double bin_size_;      // = params_.z_hist_bin
  int    max_empty_bins_; // allowed consecutive empty bins within δ
  int    bin_min_count_; // occupancy threshold per bin
  double z_shift_thresh_; // z movement threshold for recentering
  double hist_z_center_; // current center z of histogram (robot-relative)
  double hist_z_min_, hist_z_max_; // current histogram bounds (absolute)

  // Big grid ring buffer & origin
  mutable std::mutex m_;
  std::vector<float>   height_b_;
  std::vector<uint8_t> known_b_;
  std::vector<uint8_t> occ_b_;
  std::vector<float>   stamp_b_;
  int start_i_{0}, start_j_{0};
  double origin_x_{0.0}, origin_y_{0.0}, robot_z_{0.0};
  std::atomic<bool> have_origin_{false};

  // NEW: per-cell z-aggregator + cached boundary-connected height
  std::vector<ZAgg>  zagg_b_;   // size Wb*Hb
  std::vector<float> hconn_b_;  // cached h_conn_max per cell (meters)

  // Per-scan buckets (track which cells got points this scan)
  std::vector<int>   temp_cnt_;

  // New members (private)
  std::vector<float> filled_b_;   // big-grid solved height (ring-buffer layout)
  bool have_prev_fill_ = false;   // warm-start available
  // Optional: last full-grid solution as a cv::Mat for warm start convenience
  cv::Mat prev_fill_full_;        // Hb_ x Wb_, row-major (no ring), optional

  // Bump whenever the big grid's alignment/content changes (shift/clear).
  std::uint64_t rb_version_ = 0;

  // New methods (private)
  void solveGlobalFill_();                 // run after ingestPoints()
  inline void sampleFilled_(double wx, double wy, float& h_fill) const;
};

} // namespace height_mapping
