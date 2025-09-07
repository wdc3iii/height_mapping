#include "HeightMapping.hpp"
#include <iostream>
#include <algorithm>

#ifdef HEIGHT_MAPPING_PROFILE
#include <chrono>
#define PROFILE_START(name) auto start_##name = std::chrono::high_resolution_clock::now()
#define PROFILE_END(name) \
  do { \
    auto end_##name = std::chrono::high_resolution_clock::now(); \
    auto dur_##name = std::chrono::duration_cast<std::chrono::microseconds>(end_##name - start_##name); \
    std::cout << "[PROFILE] " << #name << ": " << dur_##name.count() << " μs\n"; \
  } while(0)
#define PROFILE_SAVE_PNG(mat, path) save_png(mat, path)
#else
#define PROFILE_START(name) do {} while(0)
#define PROFILE_END(name) do {} while(0)
#define PROFILE_SAVE_PNG(mat, path) do {} while(0)
#endif

namespace height_mapping {

void save_png(const cv::Mat& m, const std::string& path) {
  double minv, maxv;
  cv::minMaxLoc(m, &minv, &maxv);

  // Normalize to 0–255 range (8-bit single channel)
  cv::Mat norm;
  cv::normalize(m, norm, 0, 255, cv::NORM_MINMAX, CV_8U);

  // Apply a color map (JET is blue→green→red)
  cv::Mat color;
  cv::applyColorMap(norm, color, cv::COLORMAP_JET);

  // Save as PNG
  cv::imwrite(path, color);
}

// In-place harmonic fill on a mask. If init_interior==true, interior is
// set to 0.5*(lo+hi) before iterations; otherwise the existing dst values
// are used as a warm start.
static void harmonicFillROI_(cv::Mat& dst, const cv::Mat& mask,
                             float lo, float hi,
                             int iters = 80, float eps = 1e-3f,
                             bool init_interior = true)
{
  CV_Assert(dst.type() == CV_32FC1 && mask.type() == CV_8UC1);
  const int outHeight = dst.rows, W = dst.cols;

  if (init_interior) {
    const float initv = 0.5f * (lo + hi);
    dst.setTo(initv, mask);
  }

  // Gauss–Seidel on 4-neighbor Laplacian
  for (int it = 0; it < iters; ++it) {
    float max_delta = 0.f;

    for (int i = 1; i < outHeight-1; ++i) {
      const uint8_t* mask_row = mask.ptr<uint8_t>(i);
      float* dst_row = dst.ptr<float>(i);
      const float* up  = dst.ptr<float>(i-1);
      const float* dn  = dst.ptr<float>(i+1);
      for (int j = 1; j < W-1; ++j) {
        if (!mask_row[j]) continue; // boundary pixel stays fixed
        float newv = 0.25f * (up[j] + dn[j] + dst_row[j-1] + dst_row[j+1]);
        if (newv < lo) newv = lo;
        else if (newv > hi) newv = hi;
        float delta = std::abs(newv - dst_row[j]);
        if (delta > max_delta) max_delta = delta;
        dst_row[j] = newv;
      }
    }
    if (max_delta < eps) break;
  }
}

HeightMap::HeightMap(const Params& p)
: params_(p),
  Wb_(p.Wb), Hb_(p.Hb), res_(p.res),
  max_h_(p.max_h), zmin_(p.z_min), zmax_(p.z_max),
  drop_thresh_(p.drop_thresh), min_support_(p.min_support),
  shift_thresh_(p.shift_thresh)
{
  // Ensure subgrid fits inside big grid under worst-case yaw and shift
  if (params_.Wb / 2 * params_.res <
      params_.shift_thresh + std::sqrt(2.0) * params_.Wq / 2 * params_.res_q) {
    const int new_Wq = static_cast<int>(std::floor(
      (params_.Wb / 2 * params_.res - params_.shift_thresh) / std::sqrt(2.0) * 2 / params_.res_q));
    std::cout << "Warning: subgrid Wq too large for big grid, reducing from "
              << params_.Wq << " to " << new_Wq << std::endl;
    params_.Wq = std::max(1, new_Wq);
  }
  if (params_.Hb / 2 * params_.res <
      params_.shift_thresh + std::sqrt(2.0) * params_.Hq / 2 * params_.res_q) {
    const int new_Hq = static_cast<int>(std::floor(
      (params_.Hb / 2 * params_.res - params_.shift_thresh) / std::sqrt(2.0) * 2 / params_.res_q));
    std::cout << "Warning: subgrid Hq too large for big grid, reducing from "
              << params_.Hq << " to " << new_Hq << std::endl;
    params_.Hq = std::max(1, new_Hq);
  }

  // Derive histogram config
  bin_size_       = params_.z_hist_bin;
  z_shift_thresh_ = params_.z_shift_thresh;
  // Create enough bins to cover a reasonable height range around robot
  const double hist_range = 2 * std::ceil(max_h_ + z_shift_thresh_ + 0.1) / bin_size_;
  B_              = std::max(1, static_cast<int>(std::ceil(hist_range / bin_size_)));
  max_empty_bins_ = std::max(0, static_cast<int>(std::floor(params_.z_connect_delta / bin_size_)) - 1);
  bin_min_count_  = std::max(1, params_.z_bin_min_count);
  hist_z_center_  = 0.0; // will be set in ensureOrigin
  hist_z_min_     = 0.0;
  hist_z_max_     = 0.0;

  reset();
}

void HeightMap::reset() {
  const size_t Nb = static_cast<size_t>(Wb_) * static_cast<size_t>(Hb_);
  height_b_.assign(Nb, static_cast<float>(max_h_ + robot_z_));
  known_b_.assign(Nb, 0);
  occ_b_.assign(Nb, 1);
  stamp_b_.assign(Nb, 0.0f);

  // Per-cell z histograms and delta-connected max height
  zagg_b_.resize(Nb);
  hconn_b_.assign(Nb, static_cast<float>(max_h_ + robot_z_));
  for (size_t i = 0; i < Nb; ++i) zaggInit_(zagg_b_[i]);

  temp_min_.assign(Nb, std::numeric_limits<float>::infinity());
  temp_cnt_.assign(Nb, 0);
  start_i_ = start_j_ = 0;
  origin_x_ = origin_y_ = 0.0;
  have_origin_ = false;

  // Global fill buffers (optional; kept as-is)
  filled_b_.clear();
  prev_fill_full_.release();
  have_prev_fill_ = false;
}

void HeightMap::ensureOrigin(double robot_x, double robot_y, double robot_z) {
  if (have_origin_) return;
  origin_x_ = robot_x - 0.5 * Wb_ * res_;
  origin_y_ = robot_y - 0.5 * Hb_ * res_;
  robot_z_ = robot_z;
  
  // Initialize histogram bounds around robot z position
  const double hist_range = B_ * bin_size_;
  hist_z_center_ = robot_z;
  hist_z_min_    = robot_z - hist_range / 2.0;
  hist_z_max_    = robot_z + hist_range / 2.0;
  
  have_origin_ = true;
}

void HeightMap::shiftRingBuffer_(int si, int sj) {
  std::lock_guard<std::mutex> lk(m_);

  // Helper to initialize a cell to default state
  auto wipe_cell = [&](size_t id) {
    height_b_[id] = static_cast<float>(max_h_ + robot_z_);
    known_b_[id]  = 0;
    occ_b_[id]    = 1;
    stamp_b_[id]  = 0.0f;
    zaggInit_(zagg_b_[id]);  // Reset histogram - preserves zagg data in overlapping regions!
    hconn_b_[id]  = static_cast<float>(max_h_ + robot_z_);
  };

  // Only wipe the newly exposed columns/rows, not the overlapping regions
  if (sj != 0) {
    // Update start index FIRST
    start_j_ = (start_j_ - sj) % Wb_; if (start_j_ < 0) start_j_ += Wb_;
    
    // Wipe only the new columns that came into view
    const int ncols = std::abs(sj);
    const int dir = (sj > 0) ? 1 : -1;
    for (int k = 0; k < ncols; ++k) {
      int col = (dir < 0) ? ((Wb_ - 1 - k + start_j_) % Wb_) : ((k + start_j_) % Wb_);
      for (int row = 0; row < Hb_; ++row) {
        size_t id = static_cast<size_t>(row) * Wb_ + static_cast<size_t>(col);
        wipe_cell(id);
      }
    }
  }
  
  if (si != 0) {
    // Update start index FIRST  
    start_i_ = (start_i_ - si) % Hb_; if (start_i_ < 0) start_i_ += Hb_;
    
    // Wipe only the new rows that came into view
    const int nrows = std::abs(si);
    const int dir = (si < 0) ? 1 : -1;
    for (int k = 0; k < nrows; ++k) {
      int row = (dir > 0) ? ((Hb_ - 1 - k + start_i_) % Hb_) : ((k + start_i_) % Hb_);
      size_t row_off = static_cast<size_t>(row) * Wb_;
      for (int col = 0; col < Wb_; ++col) {
        size_t id = row_off + static_cast<size_t>(col);
        wipe_cell(id);
      }
    }
  }
  
  ++rb_version_;
}

void HeightMap::recenterIfNeeded(double robot_x, double robot_y, double robot_z) {
  if (!have_origin_) return;
  
  // Check x,y recentering
  const double cx = origin_x_ + 0.5 * Wb_ * res_;
  const double cy = origin_y_ + 0.5 * Hb_ * res_;
  const double dx = robot_x - cx;
  const double dy = robot_y - cy;
  bool need_xy_shift = (std::abs(dx) >= shift_thresh_ || std::abs(dy) >= shift_thresh_);
  
  // Check z recentering
  const double dz = robot_z - hist_z_center_;
  bool need_z_shift = (std::abs(dz) >= z_shift_thresh_);
  
  robot_z_ = robot_z;
  
  if (need_z_shift) {
    recenterHistogramBounds_(robot_z);
  }
  
  if (need_xy_shift) {
    const int sj = static_cast<int>(dx / res_);   //casting as int truncates towards zero
    const int si = static_cast<int>(dy / res_);
    if (si != 0 || sj != 0) {
      origin_x_ += sj * res_;
      origin_y_ += si * res_;
      shiftRingBuffer_(-si, -sj);
    }
  }
}

// --- per-cell histogram update ---
inline void HeightMap::zaggInsert_(ZAgg& agg, float z) {
  // Map z to bin index using current histogram bounds
  int bin = static_cast<int>(std::floor((z - hist_z_min_) / bin_size_));
  if (bin < 0 || bin >= B_) return;   // z is outside current histogram range

  uint8_t& count = agg.bins[bin];     // Increment the appropriate count, if bin not saturated
  if (count < 255) ++count;
  
  auto bin_present = [&](int idx)->bool {
    return (idx >= 0 && idx < B_) && (agg.bins[idx] >= bin_min_count_);
  };  // Check if a bin is occupied, i.e. has more than bin_min_count_ points

  if (agg.min_bin == B_ || bin < agg.min_bin) {  // The bin is lower than existing min, update the min_bin
    agg.min_bin = bin;
    int top = agg.min_bin, gaps = 0;
    for (int t = agg.min_bin + 1; t < B_; ++t) {
      if (bin_present(t)) { top = t; gaps = 0; }
      else if (++gaps > max_empty_bins_) break;
    }
    agg.top_conn_from_min = top;
  } else if (agg.top_conn_from_min >= 0 && bin <= agg.top_conn_from_min + max_empty_bins_) {
    int top = agg.top_conn_from_min, gaps = 0;
    for (int t = top + 1; t < B_; ++t) {
      if (bin_present(t)) { top = t; gaps = 0; }
      else if (++gaps > max_empty_bins_) break;
    }
    agg.top_conn_from_min = top;
  }

  // Update cached heights using current histogram bounds
  if (agg.min_bin < B_) {
    agg.h_min = static_cast<float>(hist_z_min_ + (agg.min_bin + 0.5) * bin_size_);
    int top = (agg.top_conn_from_min >= 0) ? agg.top_conn_from_min : agg.min_bin;
    agg.h_conn_max = static_cast<float>(hist_z_min_ + (top + 0.5) * bin_size_);
  } else {
    agg.h_min = agg.h_conn_max = static_cast<float>(max_h_ + robot_z_);
  }
}

void HeightMap::recenterHistogramBounds_(double new_robot_z) {
  const double old_center = hist_z_center_;
  const double old_min = hist_z_min_;
  
  // Calculate how many bins we need to shift
  const double z_shift = new_robot_z - old_center;
  const int shift_bins = static_cast<int>(z_shift / bin_size_);
  
  if (shift_bins == 0) return; // No shifting needed

  // Update histogram bounds to be centered around new robot z
  const double hist_range = B_ * bin_size_;
  hist_z_center_ += shift_bins * bin_size_;
  hist_z_min_ += shift_bins * bin_size_;
  hist_z_max_ += shift_bins * bin_size_;
  
  // Cycle all histogram bins
  std::lock_guard<std::mutex> lk(m_);
  const size_t N = static_cast<size_t>(Wb_) * static_cast<size_t>(Hb_);
  for (size_t i = 0; i < N; ++i) {
    cycleBins_(zagg_b_[i], shift_bins);
  }
}

void HeightMap::cycleBins_(ZAgg& agg, int shift_bins) {
  if (shift_bins == 0 || agg.bins.empty()) return;
  
  const int B = static_cast<int>(agg.bins.size());
  std::vector<uint8_t> new_bins(B, 0);
  
  // Copy bins to new positions, wrapping around
  for (int old_idx = 0; old_idx < B; ++old_idx) {
    if (agg.bins[old_idx] > 0) {
      int new_idx = old_idx - shift_bins;
      // Clamp to valid range instead of wrapping
      if (new_idx >= 0 && new_idx < B) {
        new_bins[new_idx] = agg.bins[old_idx];
      }
    }
  }
  
  agg.bins = std::move(new_bins);
  
  // Update bin indices
  if (agg.min_bin < B) {
    agg.min_bin -= shift_bins;
    if (agg.min_bin < 0 || agg.min_bin >= B) {
      agg.min_bin = B; // Mark as invalid
    }
  }
  
  if (agg.top_conn_from_min >= 0) {
    agg.top_conn_from_min -= shift_bins;
    if (agg.top_conn_from_min < 0 || agg.top_conn_from_min >= B) {
      agg.top_conn_from_min = -1; // Mark as invalid
    }
  }
  
  // Recalculate heights with new bounds
  if (agg.min_bin < B) {
    agg.h_min = static_cast<float>(hist_z_min_ + (agg.min_bin + 0.5) * bin_size_);
    int top = (agg.top_conn_from_min >= 0) ? agg.top_conn_from_min : agg.min_bin;
    agg.h_conn_max = static_cast<float>(hist_z_min_ + (top + 0.5) * bin_size_);
  } else {
    agg.h_min = agg.h_conn_max = static_cast<float>(max_h_ + robot_z_);
  }
}

void HeightMap::solveGlobalFill_()
{
  static int run_counter = 0;
  const std::string run_suffix = "_" + std::to_string(run_counter++);
  
  PROFILE_START(total_solve);
  
  // ---------- Snapshot under one short lock ----------
  PROFILE_START(copy_from_ringbuffer);
  cv::Mat big_height(Hb_, Wb_, CV_32FC1);
  cv::Mat big_boundary_conn(Hb_, Wb_, CV_32FC1); // delta-connected max
  cv::Mat big_occluded(Hb_, Wb_, CV_8UC1);       // 1=occluded, 0=observed
  std::uint64_t version_snapshot = 0;

  {
    std::lock_guard<std::mutex> lk(m_);
    version_snapshot = rb_version_;
    for (int i = 0; i < Hb_; ++i) {
      float*  h_row  = big_height.ptr<float>(i);
      float*  b_row  = big_boundary_conn.ptr<float>(i);
      uint8_t* o_row = big_occluded.ptr<uint8_t>(i);
      for (int j = 0; j < Wb_; ++j) {
        const size_t id = idxRB(i, j);
        h_row[j]  = height_b_[id];
        b_row[j]  = hconn_b_[id];
        o_row[j]  = occ_b_[id];
      }
    }
  }
  PROFILE_END(copy_from_ringbuffer);

  PROFILE_SAVE_PNG(big_height, "big_height0" + run_suffix + ".png");
  PROFILE_SAVE_PNG(big_boundary_conn, "big_boundary_conn0" + run_suffix + ".png");
  PROFILE_SAVE_PNG(big_occluded, "big_occluded0" + run_suffix + ".png");

  // ---------- Build occlusion mask & edge clusters ----------
  PROFILE_START(connected_components);
  // cv::Mat occ_mask = (big_occluded > 0); // CV_8U, 0/255 interior=in mask  TODO: why not just use big_occluded directly?
  cv::Mat occluded_labels;
  const int n_labels = cv::connectedComponents(big_occluded, occluded_labels, 4, CV_32S);
  PROFILE_END(connected_components);
  std::vector<uint8_t> cluster_touches_edge(std::max(1, n_labels), 0);

  auto mark_if_edge = [&](int i, int j) {
    if (!big_occluded.at<uint8_t>(i,j)) return;
    const int lbl = occluded_labels.at<int>(i,j);
    cluster_touches_edge[lbl] = 1;
  };
  // loop over edge squares, mark which clusters touch the edge
  for (int j = 0; j < Wb_; ++j) { mark_if_edge(0, j); mark_if_edge(Hb_-1, j); }
  for (int i = 0; i < Hb_; ++i) { mark_if_edge(i, 0); mark_if_edge(i, Wb_-1); }

  // Any occluded cell whose component touches the outer border: set to max and remove from mask
  PROFILE_START(set_boundary_occlusions);
  // TODO: do we need the additional clones? we have already coppied height_b
  for (int i = 0; i < Hb_; ++i) {
    uint8_t* big_occluded_row = big_occluded.ptr<uint8_t>(i);
    float*   big_height_row = big_height.ptr<float>(i);
    for (int j = 0; j < Wb_; ++j) {
      if (!big_occluded_row[j]) continue;
      const int lbl = occluded_labels.at<int>(i,j);
      if (cluster_touches_edge[lbl]) {
        big_height_row[j] = static_cast<float>(max_h_ + robot_z_);
        big_occluded_row[j] = 0;
      }
    }
  }
  PROFILE_END(set_boundary_occlusions);
  PROFILE_SAVE_PNG(big_occluded * 255, "big_occluded1" + run_suffix + ".png");
  PROFILE_SAVE_PNG(big_height, "big_height1" + run_suffix + ".png");

  // ---------- Dirichlet boundary only for the PDE (do NOT persist into output) ----------
  PROFILE_START(update_boundary_conditions);
  // Boundary = observed cells that are 4-neighbors of currently occluded cells
  for (int i = 0; i < Hb_; ++i) {
    uint8_t* mrow = big_occluded.ptr<uint8_t>(i);
    float* h_row = big_height.ptr<float>(i);
    for (int j = 0; j < Wb_; ++j) {
      if (!mrow[j]) {
        continue;
      }
      bool updated = false;
      if (i > 0     && !big_occluded.at<uint8_t>(i-1,j)) {
        // neighbor is known
        big_height.at<float>(i-1,j)= big_boundary_conn.at<float>(i-1,j);
      }
      if (i+1 < Hb_ && !big_occluded.at<uint8_t>(i+1,j)) {
        // neighbor is known
        big_height.at<float>(i+1,j) = big_boundary_conn.at<float>(i+1,j);
      }
      if (j > 0     && !big_occluded.at<uint8_t>(i,j-1)) {
        // neighbor is known
        h_row[j-1] = big_boundary_conn.at<float>(i,j-1);
      }
      if (j+1 < Wb_ && !big_occluded.at<uint8_t>(i,j+1)) {
        // neighbor is known
        h_row[j+1] = big_boundary_conn.at<float>(i,j+1);;
      }

      //Diagonals
      if (i > 0 && j > 0 && !big_occluded.at<uint8_t>(i-1,j-1)) {
        // neighbor is known
        big_height.at<float>(i-1,j-1)= big_boundary_conn.at<float>(i-1,j-1);
      }
      if (i+1 < Hb_ && j > 0 && !big_occluded.at<uint8_t>(i+1,j-1)) {
        // neighbor is known
        big_height.at<float>(i+1,j-1) = big_boundary_conn.at<float>(i+1,j-1);
      }
      if (i > 0 && j+1 <  Wb_ && !big_occluded.at<uint8_t>(i-1,j+1)) {
        // neighbor is known
        big_height.at<float>(i-1,j+1) = big_boundary_conn.at<float>(i-1,j+1);
      }
      if (i+1 < Hb_ && j+1 <  Wb_ && !big_occluded.at<uint8_t>(i+1,j+1)) {
        // neighbor is known
        big_height.at<float>(i+1,j+1) = big_boundary_conn.at<float>(i+1,j+1);
      }
    }
  }
  PROFILE_END(update_boundary_conditions);

  PROFILE_SAVE_PNG(big_boundary_conn, "big_boundary_conn1" + run_suffix + ".png");
  PROFILE_SAVE_PNG(big_occluded * 255, "big_occluded2" + run_suffix + ".png");
  PROFILE_SAVE_PNG(big_height, "big_height2" + run_suffix + ".png");

  // ---------- Warm start & clamp ----------
  PROFILE_START(laplace_solve);
  // If we have a previous full-grid solution matching the current size, warm-start the interior
  if (!prev_fill_full_.empty() &&
      prev_fill_full_.rows == Hb_ && prev_fill_full_.cols == Wb_) {
    prev_fill_full_.copyTo(big_height, big_occluded);
    harmonicFillROI_(big_height, big_occluded,
                     -std::numeric_limits<float>::infinity(),
                      std::numeric_limits<float>::infinity(),
                     /*iters=*/80, /*eps=*/1e-3f, /*init_interior=*/false);
  } else {
    // Conservative clamp from boundary (observed) values
    cv::Mat invMask; cv::bitwise_not(big_occluded, invMask);
    double min_b = 0.0, max_b = 0.0;
    cv::minMaxLoc(big_height, &min_b, &max_b, nullptr, nullptr, invMask);
    const float lo = std::isfinite(min_b) ? static_cast<float>(min_b) : 0.0f;
    const float hi = std::isfinite(max_b) ? static_cast<float>(max_b) : static_cast<float>(max_h_ + robot_z_);
    harmonicFillROI_(big_height, big_occluded, lo, hi, /*iters=*/80, /*eps=*/1e-3f, /*init_interior=*/true);
  }
  PROFILE_END(laplace_solve);

  // ---------- Build final field: keep observed cells as raw heights ----------
  PROFILE_SAVE_PNG(big_height, "big_height3" + run_suffix + ".png");
  PROFILE_SAVE_PNG(big_occluded * 255, "big_occluded3" + run_suffix + ".png");

  // ---------- Write back if the grid didn't shift while we solved ----------
  PROFILE_START(copy_to_ringbuffer);
  {
    std::lock_guard<std::mutex> lk(m_);
    if (version_snapshot != rb_version_) {
      // Grid moved (recenter/shift) during solve; discard this stale result.
      // TODO: maybe just move/shift this result instead of doing nothing?
      PROFILE_END(copy_to_ringbuffer);
      PROFILE_END(total_solve);
      return;
    }

    const size_t N = static_cast<size_t>(Hb_) * static_cast<size_t>(Wb_);
    if (filled_b_.size() != N) filled_b_.assign(N, static_cast<float>(max_h_ + robot_z_));
    for (int i = 0; i < Hb_; ++i) {
      const float* h_row = big_height.ptr<float>(i);
      for (int j = 0; j < Wb_; ++j) {
        filled_b_[idxRB(i, j)] = h_row[j];  // map to current ring-buffer storage
      }
    }
  }
  PROFILE_END(copy_to_ringbuffer);

  have_prev_fill_ = true;
  PROFILE_END(total_solve);
}

void HeightMap::ingestPoints(const std::vector<Point3f>& pts) {
  if (!have_origin_) return;

  static int run_counter = 0;
  const std::string run_suffix = "_" + std::to_string(run_counter++);

  // TODO: Debugging
  cv::Mat bh(Hb_, Wb_, CV_32FC1);
  cv::Mat bk(Hb_, Wb_, CV_32FC1); // delta-connected max
  cv::Mat bo(Hb_, Wb_, CV_8UC1);       // 1=occluded, 0=observed
  {
    std::lock_guard<std::mutex> lk(m_);
    for (int i = 0; i < Hb_; ++i) {
      float*  h_row  = bh.ptr<float>(i);
      float*  k_row  = bk.ptr<float>(i);
      uint8_t* o_row = bo.ptr<uint8_t>(i);
      for (int j = 0; j < Wb_; ++j) {
        const size_t id = idxRB(i, j);
        h_row[j]  = height_b_[id];
        k_row[j]  = known_b_[id];
        o_row[j]  = occ_b_[id];
      }
    }
  }
  PROFILE_SAVE_PNG(bo, "pre_ingest_occ" + run_suffix + ".png");
  PROFILE_SAVE_PNG(bh, "pre_ingest_h" + run_suffix + ".png");
  PROFILE_SAVE_PNG(bk, "pre_ingest_known" + run_suffix + ".png");

  // Reset per-scan min buckets
  const size_t Nb = static_cast<size_t>(Wb_) * static_cast<size_t>(Hb_);
  std::fill(temp_min_.begin(), temp_min_.end(), std::numeric_limits<float>::infinity());
  std::fill(temp_cnt_.begin(), temp_cnt_.end(), 0);

  // Bin points: update temp_min_ and z-histograms together
  for (const auto& p : pts) {
    if (p.z < robot_z_ + params_.z_min || p.z > robot_z_ + params_.z_max) continue;
    const int j = static_cast<int>(std::floor((p.x - origin_x_) / res_));
    const int i = static_cast<int>(std::floor((p.y - origin_y_) / res_));
    if (i < 0 || i >= Hb_ || j < 0 || j >= Wb_) continue;
    const size_t id = idxRB(i, j);

    float &cell_min = temp_min_[id];
    if (p.z < cell_min) cell_min = p.z;
    ++temp_cnt_[id];

    // Update per-cell histogram (no lock; ingestPoints is single-threaded)
    ZAgg& agg = zagg_b_[id];
    zaggInsert_(agg, p.z);
    hconn_b_[id] = agg.h_conn_max; // keep cache fresh
  }

  // Fuse per-scan min into height & masks (lock during write)
  const float tnow = 0.0f;
  {
    for (int i = 0; i < Hb_; ++i) {
      for (int j = 0; j < Wb_; ++j) {
        const size_t id = idxRB(i,j);
        const int cnt = temp_cnt_[id];
        if (cnt == 0) continue;

        const float zmin_scan = temp_min_[id];
        float   &h = height_b_[id];
        uint8_t &k = known_b_[id];
        uint8_t &o = occ_b_[id];

        if (!k) { h = zmin_scan; k = 1; o = 0; stamp_b_[id] = tnow; continue; }
        if (zmin_scan < h - static_cast<float>(drop_thresh_) && cnt >= min_support_) {
          h = zmin_scan; o = 0; stamp_b_[id] = tnow;
        } else if (zmin_scan < h) {
          h = 0.9f*h + 0.1f*zmin_scan; o = 0; stamp_b_[id] = tnow;
        }
      }
    }

    // TODO: Debugging
  {
    for (int i = 0; i < Hb_; ++i) {
      float*  h_row  = bh.ptr<float>(i);
      float*  k_row  = bk.ptr<float>(i);
      uint8_t* o_row = bo.ptr<uint8_t>(i);
      for (int j = 0; j < Wb_; ++j) {
        const size_t id = idxRB(i, j);
        h_row[j]  = height_b_[id];
        k_row[j]  = known_b_[id];
        o_row[j]  = occ_b_[id];
      }
    }
  }
  PROFILE_SAVE_PNG(bo, "post_ingest_occ" + run_suffix + ".png");
  PROFILE_SAVE_PNG(bh, "post_ingest_h" + run_suffix + ".png");
  PROFILE_SAVE_PNG(bk, "post_ingest_known" + run_suffix + ".png");
  } // lock released
  //TODO: is this more efficient than locking per-cell inside the loop? And doing all updates in one pass?

  // Global occlusion solve (also updates filled_b_ for fast sampling)
  solveGlobalFill_();
}


void HeightMap::generateSubgrid(double rx, double ry, double rYaw,
                                cv::Mat& sub_raw, cv::Mat& sub_filled, SubgridMeta& meta) const
{
  // ---- Snapshot everything we need under a single short lock
  const int Wb = Wb_, Hb = Hb_;
  double origin_x, origin_y, res;
  int start_i, start_j;
  const double max_h = max_h_ + robot_z_;
  const double min_h = -max_h_ + robot_z_;

  // Local copies of ring-buffered arrays (flat, row-major)
  std::vector<float> snap_height;
  std::vector<float> snap_hconn;
  std::vector<uint8_t> snap_known;
  std::vector<uint8_t> snap_occ;
  std::vector<float> snap_filled;  // may be empty; we’ll fall back to height

  {
    std::lock_guard<std::mutex> lk(m_);

    origin_x = origin_x_;
    origin_y = origin_y_;
    res      = res_;
    start_i  = start_i_;
    start_j  = start_j_;

    snap_height = height_b_;   // float per cell
    snap_hconn  = hconn_b_;    // float per cell (delta-connected max)
    snap_known  = known_b_;    // 0/1
    snap_occ    = occ_b_;      // 0/1

    // If we have a filled field, copy; else leave empty and we’ll fall back to snap_height.
    snap_filled = (!filled_b_.empty()) ? filled_b_ : height_b_;
  }

  auto idxRB_local = [&](int i, int j) -> size_t {
    // Map logical (i,j) into ring-buffered linear index using the snapped offsets
    int ri = i + start_i; if (ri >= Hb) ri -= Hb; else if (ri < 0) ri += Hb;
    int rj = j + start_j; if (rj >= Wb) rj -= Wb; else if (rj < 0) rj += Wb;
    return static_cast<size_t>(ri) * Wb + static_cast<size_t>(rj);
  };

  auto sampleBilinear = [&](double wx, double wy, float& h_raw, float& h_fill) {
    const double uf = (wx - origin_x) / res;
    const double vf = (wy - origin_y) / res;

    const int j0 = static_cast<int>(std::floor(uf));
    const int i0 = static_cast<int>(std::floor(vf));
    const double du = uf - j0;
    const double dv = vf - i0;

    if (i0 < 0 || i0 + 1 >= Hb || j0 < 0 || j0 + 1 >= Wb) {
      h_raw = static_cast<float>(max_h);
      h_fill = static_cast<float>(max_h);
      return;
    }

    const size_t id00 = idxRB_local(i0,   j0  );
    const size_t id10 = idxRB_local(i0,   j0+1);
    const size_t id01 = idxRB_local(i0+1, j0  );
    const size_t id11 = idxRB_local(i0+1, j0+1);

    // Known flags from the snapshot
    const uint8_t k00 = snap_known[id00];
    const uint8_t k10 = snap_known[id10];
    const uint8_t k01 = snap_known[id01];
    const uint8_t k11 = snap_known[id11];

    const int known_cnt = (k00!=0) + (k10!=0) + (k01!=0) + (k11!=0);

    const double w00 = (1.0 - du)*(1.0 - dv);
    const double w10 = (du)      *(1.0 - dv);
    const double w01 = (1.0 - du)*(dv);
    const double w11 = (du)      *(dv);

    if (known_cnt > 0) {
      // Use the SAME known-aware bilinear as the raw view -> identical outside occlusion.
      const double kw00 = k00 ? w00 : 0.0;
      const double kw10 = k10 ? w10 : 0.0;
      const double kw01 = k01 ? w01 : 0.0;
      const double kw11 = k11 ? w11 : 0.0;
      const double kwsum = w00 + w10 + w01 + w11;

      // Use the raw heights here (observed corners)
      const float h00 = snap_height[id00], h10 = snap_height[id10];
      const float h01 = snap_height[id01], h11 = snap_height[id11];
      
      h_raw = static_cast<float>((kw00*h00 + kw10*h10 + kw01*h01 + kw11*h11) / kwsum);
    } else {
      h_raw = static_cast<float>(max_h);
    }

    // Show the PDE solution
    const float f00 = snap_filled[id00], f10 = snap_filled[id10];
    const float f01 = snap_filled[id01], f11 = snap_filled[id11];

    h_fill = static_cast<float>(w00*f00 + w10*f10 + w01*f01 + w11*f11);
    h_fill = std::min(std::max(h_fill, static_cast<float>(min_h)), static_cast<float>(max_h));
  };

  // ---- Now do the usual subgrid sampling with NO locks
  const int Wq = params_.Wq, Hq = params_.Hq;
  const double rq = params_.res_q;

  sub_raw.create(Hq, Wq, CV_32FC1);
  sub_filled.create(Hq, Wq, CV_32FC1);

  const double c = std::cos(rYaw), s = std::sin(rYaw);
  const double halfW = 0.5 * Wq * rq;
  const double halfH = 0.5 * Hq * rq;

  for (int i = 0; i < Hq; ++i) {
    float* raw_row    = sub_raw.ptr<float>(i);
    float* filled_row = sub_filled.ptr<float>(i);
    const double gy = (static_cast<double>(i) + 0.5) * rq - halfH;
    for (int j = 0; j < Wq; ++j) {
      const double gx = (static_cast<double>(j) + 0.5) * rq - halfW;

      const double wx = rx + c*gx - s*gy;
      const double wy = ry + s*gx + c*gy;

      float hraw, hfill;
      sampleBilinear(wx, wy, hraw, hfill);
      raw_row[j] = hraw;
      filled_row[j] = hfill;
    }
  }

  // ---- Meta
  meta.width = Wq; meta.height = Hq;
  meta.resolution = static_cast<float>(rq);
  meta.yaw = rYaw;

  const double ox = rx + c*(-halfW) - s*(-halfH);
  const double oy = ry + s*(-halfW) + c*(-halfH);
  meta.origin_x = ox;
  meta.origin_y = oy;
}


void HeightMap::snapshotBig(cv::Mat& outHeight, cv::Mat& outKnown, cv::Mat& outOcc) const {
  // Copies big grid into snapshot
  outHeight.create(Hb_, Wb_, CV_32FC1);
  outKnown.create(Hb_, Wb_, CV_8UC1);
  outOcc.create(Hb_, Wb_, CV_8UC1);
  std::lock_guard<std::mutex> lk(m_);
  for (int i = 0; i < Hb_; ++i) {
    float*   hr = outHeight.ptr<float>(i);
    uint8_t* kr = outKnown.ptr<uint8_t>(i);
    uint8_t* orw= outOcc.ptr<uint8_t>(i);
    for (int j = 0; j < Wb_; ++j) {
      const size_t id = idxRB(i,j);
      hr[j] = height_b_[id];
      kr[j] = known_b_[id];
      orw[j]= occ_b_[id];
    }
  }
}

} // namespace height_mapping
