#include "HeightMapping.hpp"
#include <iostream>
#include <algorithm>

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
  B_              = std::max(1, static_cast<int>(std::ceil((zmax_ - zmin_) / bin_size_)));
  max_empty_bins_ = std::max(0, static_cast<int>(std::floor(params_.z_connect_delta / bin_size_)) - 1);
  bin_min_count_  = std::max(1, params_.z_bin_min_count);

  reset();
}

void HeightMap::reset() {
  const size_t Nb = static_cast<size_t>(Wb_) * static_cast<size_t>(Hb_);
  height_b_.assign(Nb, static_cast<float>(max_h_));
  known_b_.assign(Nb, 0);
  occ_b_.assign(Nb, 1);
  stamp_b_.assign(Nb, 0.0f);

  // Per-cell z histograms and delta-connected max height
  zagg_b_.resize(Nb);
  hconn_b_.assign(Nb, static_cast<float>(max_h_));
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

void HeightMap::ensureOrigin(double robot_x, double robot_y) {
  if (have_origin_) return;
  origin_x_ = robot_x - 0.5 * Wb_ * res_;
  origin_y_ = robot_y - 0.5 * Hb_ * res_;
  have_origin_ = true;
}

void HeightMap::shiftRingBuffer_(int si, int sj) {
  std::lock_guard<std::mutex> lk(m_);

  auto wipe_col = [&](int col) {
    for (int row = 0; row < Hb_; ++row) {
      size_t id = static_cast<size_t>(row) * Wb_ + static_cast<size_t>(col);
      height_b_[id] = static_cast<float>(max_h_);
      known_b_[id]  = 0;
      occ_b_[id]    = 1;
      stamp_b_[id]  = 0.0f;
      zaggInit_(zagg_b_[id]);
      hconn_b_[id]  = static_cast<float>(max_h_);
    }
  };
  auto wipe_row = [&](int row) {
    size_t row_off = static_cast<size_t>(row) * Wb_;
    for (int col = 0; col < Wb_; ++col) {
      size_t id = row_off + static_cast<size_t>(col);
      height_b_[id] = static_cast<float>(max_h_);
      known_b_[id]  = 0;
      occ_b_[id]    = 1;
      stamp_b_[id]  = 0.0f;
      zaggInit_(zagg_b_[id]);
      hconn_b_[id]  = static_cast<float>(max_h_);
    }
  };

  if (sj != 0) {
    start_j_ = (start_j_ - sj) % Wb_; if (start_j_ < 0) start_j_ += Wb_;
    const int ncols = std::abs(sj);
    const int dir   = (sj > 0) ? 1 : -1;
    for (int k = 0; k < ncols; ++k) {
      int col = (dir > 0) ? ((Wb_ - 1 - k + start_j_) % Wb_) : ((k + start_j_) % Wb_);
      wipe_col(col);
    }
  }
  if (si != 0) {
    start_i_ = (start_i_ - si) % Hb_; if (start_i_ < 0) start_i_ += Hb_;
    const int nrows = std::abs(si);
    const int dir   = (si > 0) ? 1 : -1;
    for (int k = 0; k < nrows; ++k) {
      int row = (dir > 0) ? ((Hb_ - 1 - k + start_i_) % Hb_) : ((k + start_i_) % Hb_);
      wipe_row(row);
    }
  }
  ++rb_version_;
}

void HeightMap::recenterIfNeeded(double robot_x, double robot_y) {
  if (!have_origin_) return;
  const double cx = origin_x_ + 0.5 * Wb_ * res_;
  const double cy = origin_y_ + 0.5 * Hb_ * res_;
  const double dx = robot_x - cx;
  const double dy = robot_y - cy;
  if (std::abs(dx) < shift_thresh_ && std::abs(dy) < shift_thresh_) return;

  const int sj = static_cast<int>(std::floor(dx / res_));
  const int si = static_cast<int>(std::floor(dy / res_));
  if (si == 0 && sj == 0) return;

  origin_x_ += sj * res_;
  origin_y_ += si * res_;
  shiftRingBuffer_(si, sj);
}

// --- per-cell histogram update ---
inline void HeightMap::zaggInsert_(ZAgg& agg, float z) {
  int bin = static_cast<int>(std::floor((z - static_cast<float>(zmin_)) / static_cast<float>(bin_size_)));
  if (bin < 0 || bin >= B_) return;   // z is outside histogram range

  uint8_t& count = agg.bins[bin];     // Increment the appropriate count, if bin not saturated
  if (count < 255) ++count;
  // if (count < bin_min_count_) return; // Not enough points yet in this bin, not considered full yet.

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
    agg.top_conn_from_min = top;  // This updates to the index which is the top connected from the minimum bin
    //TODO: can this be more efficient? We are doing a linear scan from min_bin every time we insert a new min_bin
    // it might be possible - if the new bin is the minimum, we only need to scan up to the previous min_bin. If this scan is connected, 
    // top is unchanged. Otherwise, top is in (min_bin, previous_min_bin - max_empty_bins_])
  } else if (agg.top_conn_from_min >= 0 && bin <= agg.top_conn_from_min + max_empty_bins_) {
    int top = agg.top_conn_from_min, gaps = 0;
    for (int t = top + 1; t < B_; ++t) {
      if (bin_present(t)) { top = t; gaps = 0; }
      else if (++gaps > max_empty_bins_) break;
    }
    agg.top_conn_from_min = top;
    // Here we do have to scan all the way up, since the point could have plugged a hole. 
  }

  if (agg.min_bin < B_) {
    agg.h_min = static_cast<float>(zmin_ + (agg.min_bin + 0.5) * bin_size_);
    int top = (agg.top_conn_from_min >= 0) ? agg.top_conn_from_min : agg.min_bin;
    agg.h_conn_max = static_cast<float>(zmin_ + (top + 0.5) * bin_size_);
  } else {
    agg.h_min = agg.h_conn_max = static_cast<float>(max_h_);
  }
}

void HeightMap::solveGlobalFill_()
{
  // ---------- Snapshot under one short lock ----------
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

  save_png(big_height, "big_height0.png");
  save_png(big_boundary_conn, "big_boundary_conn0.png");
  save_png(big_occluded, "big_occluded0.png");

  // ---------- Build occlusion mask & edge clusters ----------
  cv::Mat occ_mask = (big_occluded > 0); // CV_8U, 0/255 interior=in mask  TODO: why not just use big_occluded directly?
  cv::Mat occluded_labels;
  const int n_labels = cv::connectedComponents(occ_mask, occluded_labels, 4, CV_32S);
  std::vector<uint8_t> cluster_touches_edge(std::max(1, n_labels), 0);

  auto mark_if_edge = [&](int i, int j) {
    if (!occ_mask.at<uint8_t>(i,j)) return;
    const int lbl = occluded_labels.at<int>(i,j);
    cluster_touches_edge[lbl] = 1;
  };
  // loop over edge squares, mark which clusters touch the edge
  for (int j = 0; j < Wb_; ++j) { mark_if_edge(0, j); mark_if_edge(Hb_-1, j); }
  for (int i = 0; i < Hb_; ++i) { mark_if_edge(i, 0); mark_if_edge(i, Wb_-1); }

  // Any occluded cell whose component touches the outer border: set to max and remove from mask
  // TODO: do we need the additional clones? we have already coppied height_b
  cv::Mat working_mask = occ_mask.clone(); // we’ll edit this
  cv::Mat changed_working_mask(working_mask.rows, working_mask.cols, CV_8UC1, cv::Scalar(0));       // 1=occluded, 0=observed
  cv::Mat solver_field = big_height.clone(); // used by the PDE
  for (int i = 0; i < Hb_; ++i) {
    uint8_t* working_mask_row = working_mask.ptr<uint8_t>(i);
    float*   solver_field_row = solver_field.ptr<float>(i);
    for (int j = 0; j < Wb_; ++j) {
      if (!working_mask_row[j]) continue;
      const int lbl = occluded_labels.at<int>(i,j);
      if (cluster_touches_edge[lbl]) {
        solver_field_row[j] = static_cast<float>(max_h_);
        working_mask_row[j] = 0; // don’t solve there TOOO: we can just use occ_mask here?
      }
    }
  }
  save_png(working_mask * 255, "working_mask0.png");
  save_png(solver_field, "solver_field0.png");

  // ---------- Dirichlet boundary only for the PDE (do NOT persist into output) ----------
  // Boundary = observed cells that are 4-neighbors of currently occluded cells
  for (int i = 0; i < Hb_; ++i) {
    uint8_t* mrow = working_mask.ptr<uint8_t>(i);
    float* sf_row = solver_field.ptr<float>(i);
    uint8_t* change_row = changed_working_mask.ptr<uint8_t>(i);
    for (int j = 0; j < Wb_; ++j) {
      if (!mrow[j]) {
        continue;
      }
      bool updated = false;
      if (i > 0     && !working_mask.at<uint8_t>(i-1,j)) {
        // neighbor is known
        float neighbor_conn = big_boundary_conn.at<float>(i-1,j);
        sf_row[j] = updated ? std::max(sf_row[j], neighbor_conn) : neighbor_conn;
        updated = true;
      }
      if (i+1 < Hb_ && !working_mask.at<uint8_t>(i+1,j)) {
        // neighbor is known
        float neighbor_conn = big_boundary_conn.at<float>(i+1,j);
        sf_row[j] = updated ? std::max(sf_row[j], neighbor_conn) : neighbor_conn;
        updated = true;
      }
      if (j > 0     && !working_mask.at<uint8_t>(i,j-1)) {
        // neighbor is known
        float neighbor_conn = big_boundary_conn.at<float>(i,j-1);
        sf_row[j] = updated ? std::max(sf_row[j], neighbor_conn) : neighbor_conn;
        updated = true;
      }
      if (j+1 < Wb_ && !working_mask.at<uint8_t>(i,j+1)) {
        // neighbor is known
        float neighbor_conn = big_boundary_conn.at<float>(i,j+1);
        sf_row[j] = updated ? std::max(sf_row[j], neighbor_conn) : neighbor_conn;
        updated = true;
      }
      if (updated) {change_row[j] = 1;}
      // else: keep raw height as boundary
    }
  }

  save_png(big_boundary_conn, "big_boundary_conn.png");
  working_mask &= changed_working_mask == 0; // add these new boundary pixels to the PDE mask
  save_png(changed_working_mask * 255, "changed_working_mask.png");
  save_png(working_mask * 255, "working_mask1.png");
  save_png(solver_field, "solver_field1.png");

  // ---------- Warm start & clamp ----------
  // If we have a previous full-grid solution matching the current size, warm-start the interior
  if (!prev_fill_full_.empty() &&
      prev_fill_full_.rows == Hb_ && prev_fill_full_.cols == Wb_) {
    prev_fill_full_.copyTo(solver_field, working_mask);
    harmonicFillROI_(solver_field, working_mask,
                     -std::numeric_limits<float>::infinity(),
                      std::numeric_limits<float>::infinity(),
                     /*iters=*/80, /*eps=*/1e-3f, /*init_interior=*/false);
  } else {
    // Conservative clamp from boundary (observed) values
    cv::Mat invMask; cv::bitwise_not(working_mask, invMask);
    double min_b = 0.0, max_b = 0.0;
    cv::minMaxLoc(solver_field, &min_b, &max_b, nullptr, nullptr, invMask);
    const float lo = std::isfinite(min_b) ? static_cast<float>(min_b) : 0.0f;
    const float hi = std::isfinite(max_b) ? static_cast<float>(max_b) : static_cast<float>(max_h_);
    harmonicFillROI_(solver_field, working_mask, lo, hi, /*iters=*/80, /*eps=*/1e-3f, /*init_interior=*/true);
  }

  // ---------- Build final field: keep observed cells as raw heights ----------
  //  TODO: is this copying necessary
  cv::Mat final_field = big_height.clone();           // observed stay raw
  working_mask |= changed_working_mask;
  solver_field.copyTo(final_field, working_mask);     // only interior pixels get the PDE result
  prev_fill_full_ = final_field;                      // cache for warm start next time

  save_png(final_field, "final_field.png");
  save_png(working_mask * 255, "working_mask2.png");
  save_png(solver_field, "solver_field2.png");

  // ---------- Write back if the grid didn’t shift while we solved ----------
  {
    std::lock_guard<std::mutex> lk(m_);
    if (version_snapshot != rb_version_) {
      // Grid moved (recenter/shift) during solve; discard this stale result.
      // TODO: maybe just move/shift this result instead of doing nothing?
      return;
    }

    const size_t N = static_cast<size_t>(Hb_) * static_cast<size_t>(Wb_);
    if (filled_b_.size() != N) filled_b_.assign(N, static_cast<float>(max_h_));
    for (int i = 0; i < Hb_; ++i) {
      const float* row = final_field.ptr<float>(i);
      for (int j = 0; j < Wb_; ++j) {
        filled_b_[idxRB(i, j)] = row[j];  // map to current ring-buffer storage
      }
    }
  }

  have_prev_fill_ = true;
}

void HeightMap::ingestPoints(const std::vector<Point3f>& pts) {
  if (!have_origin_) return;

  // Reset per-scan min buckets
  const size_t Nb = static_cast<size_t>(Wb_) * static_cast<size_t>(Hb_);
  std::fill(temp_min_.begin(), temp_min_.end(), std::numeric_limits<float>::infinity());
  std::fill(temp_cnt_.begin(), temp_cnt_.end(), 0);

  // Bin points: update temp_min_ and z-histograms together
  for (const auto& p : pts) {
    if (p.z < zmin_ || p.z > zmax_) continue;
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
    std::lock_guard<std::mutex> lk(m_);
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
  } // lock released
  //TODO: is this more efficient than locking per-cell inside the loop? And doing all updates in one pass?

  // Global occlusion solve (also updates filled_b_ for fast sampling)
  solveGlobalFill_();
}

inline void HeightMap::sampleFilled_(double wx, double wy, float& h_fill) const
{
  const double uf = (wx - origin_x_) / res_;
  const double vf = (wy - origin_y_) / res_;
  const int j0 = static_cast<int>(std::floor(uf));
  const int i0 = static_cast<int>(std::floor(vf));
  const double du = uf - j0;
  const double dv = vf - i0;

  if (i0 < 0 || i0 + 1 >= Hb_ || j0 < 0 || j0 + 1 >= Wb_) {
    h_fill = static_cast<float>(max_h_);
    return;
  }

  float f00, f10, f01, f11;
  {
    std::lock_guard<std::mutex> lk(m_);
    size_t id00 = idxRB(i0,   j0  );
    size_t id10 = idxRB(i0,   j0+1);
    size_t id01 = idxRB(i0+1, j0  );
    size_t id11 = idxRB(i0+1, j0+1);
    const std::vector<float>& src = filled_b_.empty() ? height_b_ : filled_b_;
    f00 = src[id00]; f10 = src[id10];
    f01 = src[id01]; f11 = src[id11];
  }
  const double w00 = (1.0 - du)*(1.0 - dv);
  const double w10 = (du)      *(1.0 - dv);
  const double w01 = (1.0 - du)*(dv);
  const double w11 = (du)      *(dv);
  h_fill = static_cast<float>(w00*f00 + w10*f10 + w01*f01 + w11*f11);
}

void HeightMap::sampleBilinearBoth_(double wx, double wy,
                                    float& h_raw, float& h_bound,
                                    uint8_t& occ_val, uint8_t& edge_touch) const {
  const double uf = (wx - origin_x_) / res_;
  const double vf = (wy - origin_y_) / res_;

  const int j0 = static_cast<int>(std::floor(uf));
  const int i0 = static_cast<int>(std::floor(vf));
  const double du = uf - j0;
  const double dv = vf - i0;

  if (i0 < 0 || i0 + 1 >= Hb_ || j0 < 0 || j0 + 1 >= Wb_) {
    h_raw = static_cast<float>(max_h_);
    h_bound = static_cast<float>(max_h_);
    occ_val = 1;
    edge_touch = 1;
    return;
  }

  float h00,h10,h01,h11;
  float b00,b10,b01,b11;
  uint8_t k00,k10,k01,k11;
  uint8_t o00,o10,o01,o11;
  {
    std::lock_guard<std::mutex> lk(m_);
    size_t id00 = idxRB(i0,   j0  );
    size_t id10 = idxRB(i0,   j0+1);
    size_t id01 = idxRB(i0+1, j0  );
    size_t id11 = idxRB(i0+1, j0+1);
    h00 = height_b_[id00]; h10 = height_b_[id10];
    h01 = height_b_[id01]; h11 = height_b_[id11];

    b00 = hconn_b_[id00];  b10 = hconn_b_[id10];
    b01 = hconn_b_[id01];  b11 = hconn_b_[id11];

    k00 = known_b_[id00];  k10 = known_b_[id10];
    k01 = known_b_[id01];  k11 = known_b_[id11];

    o00 = occ_b_[id00];    o10 = occ_b_[id10];
    o01 = occ_b_[id01];    o11 = occ_b_[id11];
  }

  const double w00 = k00 ? (1.0 - du)*(1.0 - dv) : 0.0;
  const double w10 = k10 ? (du)      *(1.0 - dv) : 0.0;
  const double w01 = k01 ? (1.0 - du)*(dv)       : 0.0;
  const double w11 = k11 ? (du)      *(dv)       : 0.0;
  const double wsum = w00 + w10 + w01 + w11;

  if (k00 || k10 || k01 || k11) {
    const double h_interp = (w00*h00 + w10*h10 + w01*h01 + w11*h11) / wsum;
    const double b_interp = (w00*b00 + w10*b10 + w01*b01 + w11*b11) / wsum;
    h_raw   = static_cast<float>(h_interp);
    h_bound = static_cast<float>(b_interp);
    occ_val = 0;
  } else {
    h_raw   = static_cast<float>(max_h_);
    h_bound = static_cast<float>(max_h_);
    occ_val = 1;
  }
  edge_touch = 0;
}

// (Kept for parity; no longer used when solving globally before sampling)
void HeightMap::brushfireFill_(const cv::Mat& sub_raw,
                               const cv::Mat& sub_bound,
                               const cv::Mat& sub_occ,
                               const cv::Mat& sub_edge,
                               cv::Mat& sub_filled) const {
  CV_Assert(sub_raw.type()   == CV_32FC1);
  CV_Assert(sub_bound.type() == CV_32FC1);
  CV_Assert(sub_occ.type()   == CV_8UC1);
  CV_Assert(sub_edge.type()  == CV_8UC1);

  // Start from sub_raw
  sub_filled = sub_raw.clone();

  // Mask of all occluded cells
  cv::Mat mask = (sub_occ > 0);

  // Edge-touching occlusions: force to max and remove from mask
  for (int i = 0; i < sub_edge.rows; ++i) {
    const uint8_t* e = sub_edge.ptr<uint8_t>(i);
    const uint8_t* m = mask.ptr<uint8_t>(i);
    float* f         = sub_filled.ptr<float>(i);
    for (int j = 0; j < sub_edge.cols; ++j) {
      if (m[j] && e[j]) {
        f[j] = static_cast<float>(max_h_);
        mask.at<uint8_t>(i,j) = 0;
      }
    }
  }

  // Dirichlet boundary = boundary-connected heights on observed cells
  sub_bound.copyTo(sub_filled, ~mask);

  // Clamp range from boundary
  double minb, maxb;
  cv::minMaxLoc(sub_bound, &minb, &maxb, nullptr, nullptr, ~mask);
  const float lo = std::isfinite(minb) ? static_cast<float>(minb) : 0.0f;
  const float hi = std::isfinite(maxb) ? static_cast<float>(maxb) : static_cast<float>(max_h_);

  // Solve Laplace in occluded mask
  harmonicFillROI_(sub_filled, mask, lo, hi, /*iters=*/80, /*eps=*/1e-3f);
}

void HeightMap::generateSubgrid(double rx, double ry, double rYaw,
                                cv::Mat& sub_raw, cv::Mat& sub_filled, SubgridMeta& meta) const
{
  // ---- Snapshot everything we need under a single short lock
  const int Wb = Wb_, Hb = Hb_;
  double origin_x, origin_y, res;
  int start_i, start_j;
  double max_h = max_h_;

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
    if (!filled_b_.empty()) snap_filled = filled_b_;
  }

  auto idxRB_local = [&](int i, int j) -> size_t {
    // Map logical (i,j) into ring-buffered linear index using the snapped offsets
    int ri = i + start_i; if (ri >= Hb) ri -= Hb; else if (ri < 0) ri += Hb;
    int rj = j + start_j; if (rj >= Wb) rj -= Wb; else if (rj < 0) rj += Wb;
    return static_cast<size_t>(ri) * Wb + static_cast<size_t>(rj);
  };

  auto sampleBilinearRawBoundUnlocked = [&](double wx, double wy,
                                            float& h_raw, float& h_bound,
                                            uint8_t& occ_val) {
    // world -> big-grid fractional indices (column = x, row = y)
    const double uf = (wx - origin_x) / res;
    const double vf = (wy - origin_y) / res;

    const int j0 = static_cast<int>(std::floor(uf));
    const int i0 = static_cast<int>(std::floor(vf));
    const double du = uf - j0;
    const double dv = vf - i0;

    if (i0 < 0 || i0 + 1 >= Hb || j0 < 0 || j0 + 1 >= Wb) {
      h_raw   = static_cast<float>(max_h);
      h_bound = static_cast<float>(max_h);
      occ_val = 1;
      return;
    }

    const size_t id00 = idxRB_local(i0,   j0  );
    const size_t id10 = idxRB_local(i0,   j0+1);
    const size_t id01 = idxRB_local(i0+1, j0  );
    const size_t id11 = idxRB_local(i0+1, j0+1);

    const float h00 = snap_height[id00], h10 = snap_height[id10];
    const float h01 = snap_height[id01], h11 = snap_height[id11];

    const float b00 = snap_hconn[id00],  b10 = snap_hconn[id10];
    const float b01 = snap_hconn[id01],  b11 = snap_hconn[id11];

    const uint8_t k00 = snap_known[id00], k10 = snap_known[id10];
    const uint8_t k01 = snap_known[id01], k11 = snap_known[id11];

    const double w00 = k00 ? (1.0 - du)*(1.0 - dv) : 0.0;
    const double w10 = k10 ? (du)      *(1.0 - dv) : 0.0;
    const double w01 = k01 ? (1.0 - du)*(dv)       : 0.0;
    const double w11 = k11 ? (du)      *(dv)       : 0.0;
    const double wsum = w00 + w10 + w01 + w11;

    if (wsum > 0.0) {
      h_raw   = static_cast<float>((w00*h00 + w10*h10 + w01*h01 + w11*h11) / wsum);
      h_bound = static_cast<float>((w00*b00 + w10*b10 + w01*b01 + w11*b11) / wsum);
      occ_val = 0;
    } else {
      h_raw   = static_cast<float>(max_h);
      h_bound = static_cast<float>(max_h);
      occ_val = 1;
    }
  };

  auto sampleBilinearFilledUnlocked = [&](double wx, double wy, float& h_fill) {
    const double uf = (wx - origin_x) / res;
    const double vf = (wy - origin_y) / res;

    const int j0 = static_cast<int>(std::floor(uf));
    const int i0 = static_cast<int>(std::floor(vf));
    const double du = uf - j0;
    const double dv = vf - i0;

    if (i0 < 0 || i0 + 1 >= Hb || j0 < 0 || j0 + 1 >= Wb) {
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

    if (known_cnt > 0) {
      // Use the SAME known-aware bilinear as the raw view -> identical outside occlusion.
      const double w00 = k00 ? (1.0 - du)*(1.0 - dv) : 0.0;
      const double w10 = k10 ? (du)      *(1.0 - dv) : 0.0;
      const double w01 = k01 ? (1.0 - du)*(dv)       : 0.0;
      const double w11 = k11 ? (du)      *(dv)       : 0.0;
      const double wsum = w00 + w10 + w01 + w11;

      // Use the raw heights here (observed corners)
      const float h00 = snap_height[id00], h10 = snap_height[id10];
      const float h01 = snap_height[id01], h11 = snap_height[id11];

      if (wsum > 0.0) {
        h_fill = static_cast<float>((w00*h00 + w10*h10 + w01*h01 + w11*h11) / wsum);
      } else {
        // Shouldn’t happen if known_cnt > 0, but be safe:
        h_fill = static_cast<float>(max_h);
      }
      return;
    }

    // All four corners are unknown -> show the PDE solution
    const std::vector<float>& src = snap_filled.empty() ? snap_height : snap_filled;
    const float f00 = src[id00], f10 = src[id10];
    const float f01 = src[id01], f11 = src[id11];

    const double w00 = (1.0 - du)*(1.0 - dv);
    const double w10 = (du)      *(1.0 - dv);
    const double w01 = (1.0 - du)*(dv);
    const double w11 = (du)      *(dv);

    h_fill = static_cast<float>(w00*f00 + w10*f10 + w01*f01 + w11*f11);
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

      float hraw, hbound; uint8_t occ;
      sampleBilinearRawBoundUnlocked(wx, wy, hraw, hbound, occ);
      raw_row[j] = hraw;

      float hfill;
      sampleBilinearFilledUnlocked(wx, wy, hfill);
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
