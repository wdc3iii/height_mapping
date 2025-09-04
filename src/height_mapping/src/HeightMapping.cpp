#include "height_mapping.hpp"

namespace height_mapping {

HeightMap::HeightMap(const Params& p)
: params_(p),
  Wb_(p.Wb), Hb_(p.Hb), res_(p.res),
  max_h_(p.max_h), zmin_(p.z_min), zmax_(p.z_max),
  drop_thresh_(p.drop_thresh), min_support_(p.min_support),
  shift_thresh_(p.shift_thresh)
{
  reset();
}

void HeightMap::reset() {
  const size_t Nb = static_cast<size_t>(Wb_) * static_cast<size_t>(Hb_);
  height_b_.assign(Nb, static_cast<float>(max_h_));
  known_b_.assign(Nb, 0);
  occ_b_.assign(Nb, 1);
  stamp_b_.assign(Nb, 0.0f);
  temp_min_.assign(Nb, std::numeric_limits<float>::infinity());
  temp_cnt_.assign(Nb, 0);
  start_i_ = start_j_ = 0;
  origin_x_ = origin_y_ = 0.0;
  have_origin_ = false;
}

void HeightMap::ensureOrigin(double robot_x, double robot_y) {
  if (have_origin_) return;
  origin_x_ = robot_x - 0.5 * Wb_ * res_;
  origin_y_ = robot_y - 0.5 * Hb_ * res_;
  have_origin_ = true;
}

void HeightMap::shiftRingBuffer_(int si, int sj) {
  std::lock_guard<std::mutex> lk(m_);

  if (sj != 0) {
    start_j_ = (start_j_ - sj) % Wb_; if (start_j_ < 0) start_j_ += Wb_;
    const int ncols = std::abs(sj);
    const int dir   = (sj > 0) ? 1 : -1;
    for (int k = 0; k < ncols; ++k) {
      int col = (dir > 0) ? ((Wb_ - 1 - k + start_j_) % Wb_) : ((k + start_j_) % Wb_);
      for (int i = 0; i < Hb_; ++i) {
        size_t id = static_cast<size_t>(i) * Wb_ + static_cast<size_t>(col);
        height_b_[id] = static_cast<float>(max_h_);
        known_b_[id]  = 0;
        occ_b_[id]    = 1;
        stamp_b_[id]  = 0.0f;
      }
    }
  }
  if (si != 0) {
    start_i_ = (start_i_ - si) % Hb_; if (start_i_ < 0) start_i_ += Hb_;
    const int nrows = std::abs(si);
    const int dir   = (si > 0) ? 1 : -1;
    for (int k = 0; k < nrows; ++k) {
      int row = (dir > 0) ? ((Hb_ - 1 - k + start_i_) % Hb_) : ((k + start_i_) % Hb_);
      size_t row_off = static_cast<size_t>(row) * Wb_;
      for (int j = 0; j < Wb_; ++j) {
        size_t id = row_off + static_cast<size_t>(j);
        height_b_[id] = static_cast<float>(max_h_);
        known_b_[id]  = 0;
        occ_b_[id]    = 1;
        stamp_b_[id]  = 0.0f;
      }
    }
  }
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

void HeightMap::ingestPoints(const std::vector<Point3f>& pts) {
  if (!have_origin_) return;

  // reset temp buckets
  const size_t Nb = static_cast<size_t>(Wb_) * static_cast<size_t>(Hb_);
  std::fill(temp_min_.begin(), temp_min_.end(), std::numeric_limits<float>::infinity());
  std::fill(temp_cnt_.begin(), temp_cnt_.end(), 0);

  // bin points
  for (const auto& p : pts) {
    if (p.z < zmin_ || p.z > zmax_) continue;
    const int j = static_cast<int>(std::floor((p.x - origin_x_) / res_));
    const int i = static_cast<int>(std::floor((p.y - origin_y_) / res_));
    if (i < 0 || i >= Hb_ || j < 0 || j >= Wb_) continue;
    const size_t id = idxRB(i, j);
    float &cell_min = temp_min_[id];
    if (p.z < cell_min) cell_min = p.z;
    ++temp_cnt_[id];
  }

  // fuse
  const float tnow = 0.0f; // (could pass in if you want aging)
  std::lock_guard<std::mutex> lk(m_);
  for (int i = 0; i < Hb_; ++i) {
    for (int j = 0; j < Wb_; ++j) {
      const size_t id = idxRB(i, j);
      const int cnt = temp_cnt_[id];
      if (cnt == 0) continue;

      const float zmin = temp_min_[id];
      float &h   = height_b_[id];
      uint8_t &k = known_b_[id];
      uint8_t &o = occ_b_[id];

      if (!k) { h = zmin; k = 1; o = 0; stamp_b_[id] = tnow; continue; }
      if (zmin < h - static_cast<float>(drop_thresh_) && cnt >= min_support_) {
        h = zmin; o = 0; stamp_b_[id] = tnow;
      } else if (zmin < h) {
        h = 0.9f*h + 0.1f*zmin; o = 0; stamp_b_[id] = tnow;
      }
    }
  }
}

void HeightMap::sampleBilinear_(double wx, double wy,
                                            float& h_val, uint8_t& occ_val, uint8_t& edge_touch) const {
  // world -> big-grid fractional indices (column = x, row = y)
  const double uf = (wx - origin_x_) / res_;
  const double vf = (wy - origin_y_) / res_;

  const int j0 = static_cast<int>(std::floor(uf));
  const int i0 = static_cast<int>(std::floor(vf));
  const double du = uf - j0;
  const double dv = vf - i0;

  // outside big grid footprint?
  if (i0 < 0 || i0 + 1 >= Hb_ || j0 < 0 || j0 + 1 >= Wb_) {
    h_val = static_cast<float>(max_h_);
    occ_val = 1;
    edge_touch = 1;
    return;
  }

  float h00,h10,h01,h11; uint8_t k00,k10,k01,k11; uint8_t o00,o10,o01,o11;
  {
    std::lock_guard<std::mutex> lk(m_);
    size_t id00 = idxRB(i0,   j0  );
    size_t id10 = idxRB(i0,   j0+1);
    size_t id01 = idxRB(i0+1, j0  );
    size_t id11 = idxRB(i0+1, j0+1);
    h00 = height_b_[id00]; h10 = height_b_[id10];
    h01 = height_b_[id01]; h11 = height_b_[id11];
    k00 = known_b_[id00];  k10 = known_b_[id10];
    k01 = known_b_[id01];  k11 = known_b_[id11];
    o00 = occ_b_[id00];    o10 = occ_b_[id10];
    o01 = occ_b_[id01];    o11 = occ_b_[id11];
  }

  // known-weighted bilinear height
  const double w00 = k00 ? (1.0 - du)*(1.0 - dv) : 0.0;
  const double w10 = k10 ? (du)      *(1.0 - dv) : 0.0;
  const double w01 = k01 ? (1.0 - du)*(dv)       : 0.0;
  const double w11 = k11 ? (du)      *(dv)       : 0.0;
  const double wsum = w00 + w10 + w01 + w11;

  if (k00 || k10 || k01 || k11){
    const double h_interp = (w00*h00 + w10*h10 + w01*h01 + w11*h11) / wsum;
    h_val = static_cast<float>(h_interp);
  } else {
    h_val = static_cast<float>(max_h_);
  }

  // occlusion fraction (no need to lock again; small race is okay for viz)
  const double occ_avg =
      (1.0 - du)*(1.0 - dv) * o00 +
      (du)      *(1.0 - dv) * o10 +
      (1.0 - du)*(dv)       * o01 +
      (du)      *(dv)       * o11;

  occ_val = static_cast<uint8_t>(occ_avg > params_.occluded_fraction_threshold);
  edge_touch = 0;
}

void HeightMap::brushfireFill_(const cv::Mat& sub_raw, const cv::Mat& sub_occ,
                                           const cv::Mat& sub_edge, cv::Mat& sub_filled) const {
  CV_Assert(sub_raw.type() == CV_32FC1);
  CV_Assert(sub_occ.type() == CV_8UC1);
  CV_Assert(sub_edge.type()== CV_8UC1);

  cv::Mat labels;
  int nLabels = cv::connectedComponents(sub_occ, labels, 4, CV_32S);

  struct Cluster { bool touches_edge=false; float min_boundary=std::numeric_limits<float>::infinity(); };
  std::vector<Cluster> clusters(std::max(1, nLabels));

  auto inb = [&](int i, int j){ return (i>=0 && i<sub_raw.rows && j>=0 && j<sub_raw.cols); };

  // accumulate
  for (int i = 0; i < sub_raw.rows; ++i) {
    const int* L = labels.ptr<int>(i);
    const uint8_t* O = sub_occ.ptr<uint8_t>(i);
    for (int j = 0; j < sub_raw.cols; ++j) {
      if (!O[j]) continue;
      Cluster &C = clusters[L[j]];
      if (sub_edge.at<uint8_t>(i,j)) C.touches_edge = true;

      static const int di[4] = {-1,1,0,0};
      static const int dj[4] = {0,0,-1,1};
      for (int k = 0; k < 4; ++k) {
        int ni = i + di[k], nj = j + dj[k];
        if (!inb(ni,nj)) { C.touches_edge = true; continue; }
        if (sub_occ.at<uint8_t>(ni,nj) == 0) {
          float nh = sub_raw.at<float>(ni,nj);
          if (nh < C.min_boundary) C.min_boundary = nh;
        }
      }
    }
  }

  sub_filled = sub_raw.clone();
  for (int i = 0; i < sub_raw.rows; ++i) {
    const int* L = labels.ptr<int>(i);
    const uint8_t* O = sub_occ.ptr<uint8_t>(i);
    float* F = sub_filled.ptr<float>(i);
    for (int j = 0; j < sub_raw.cols; ++j) {
      if (!O[j]) continue;
      const Cluster &C = clusters[L[j]];
      if (C.touches_edge || !std::isfinite(C.min_boundary)) {
        F[j] = static_cast<float>(max_h_);
      } else {
        F[j] = C.min_boundary;
      }
    }
  }
}

void HeightMap::generateSubgrid(double rx, double ry, double rYaw,
                                            cv::Mat& sub_raw, cv::Mat& sub_filled, SubgridMeta& meta) const {
  const int Wq = params_.Wq, Hq = params_.Hq;
  const double rq = params_.res_q;

  sub_raw.create(Hq, Wq, CV_32FC1);
  cv::Mat sub_occ(Hq, Wq, CV_8UC1);
  cv::Mat sub_edge(Hq, Wq, CV_8UC1, cv::Scalar(0));

  const double c = std::cos(rYaw), s = std::sin(rYaw);
  const double halfW = 0.5 * Wq * rq;
  const double halfH = 0.5 * Hq * rq;

  for (int i = 0; i < Hq; ++i) {
    float* hr = sub_raw.ptr<float>(i);
    uint8_t* ho = sub_occ.ptr<uint8_t>(i);
    uint8_t* he = sub_edge.ptr<uint8_t>(i);

    const double gy = (static_cast<double>(i) + 0.5) * rq - halfH;
    for (int j = 0; j < Wq; ++j) {
      const double gx = (static_cast<double>(j) + 0.5) * rq - halfW;

      const double wx = rx + c*gx - s*gy;
      const double wy = ry + s*gx + c*gy;

      float h; uint8_t occ, edge;
      sampleBilinear_(wx, wy, h, occ, edge);

      hr[j] = h;
      ho[j] = occ;
      he[j] = edge;
    }
  }

  brushfireFill_(sub_raw, sub_occ, sub_edge, sub_filled);

  // meta
  meta.width = Wq; meta.height = Hq;
  meta.resolution = static_cast<float>(rq);
  meta.yaw = rYaw;

  // bottom-left of subgrid in world
  const double ox = rx + c*(-halfW) - s*(-halfH);
  const double oy = ry + s*(-halfW) + c*(-halfH);
  meta.origin_x = ox;
  meta.origin_y = oy;
}

void HeightMap::snapshotBig(cv::Mat& H, cv::Mat& K, cv::Mat& O) const {
  H.create(Hb_, Wb_, CV_32FC1);
  K.create(Hb_, Wb_, CV_8UC1);
  O.create(Hb_, Wb_, CV_8UC1);
  std::lock_guard<std::mutex> lk(m_);
  for (int i = 0; i < Hb_; ++i) {
    float* hr = H.ptr<float>(i);
    uint8_t* kr = K.ptr<uint8_t>(i);
    uint8_t* orw = O.ptr<uint8_t>(i);
    for (int j = 0; j < Wb_; ++j) {
      const size_t id = idxRB(i,j);
      hr[j] = height_b_[id];
      kr[j] = known_b_[id];
      orw[j]= occ_b_[id];
    }
  }
}

} // namespace height_mapping
