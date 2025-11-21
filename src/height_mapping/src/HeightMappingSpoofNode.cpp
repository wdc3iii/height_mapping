#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <sensor_msgs/point_cloud2_iterator.hpp>
#include <geometry_msgs/msg/transform_stamped.hpp>

#include <tf2_ros/buffer.h>
#include <tf2/LinearMath/Matrix3x3.h>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2_ros/transform_listener.h>

#include <opencv2/core.hpp>

#include <pcl/point_types.h>
#include <pcl/ModelCoefficients.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/segmentation/sac_segmentation.h>

#include <mujoco/mujoco.h>
#include <ament_index_cpp/get_package_share_directory.hpp>

#include "HeightMapping.hpp"

class HeightMapSpoofNode : public rclcpp::Node {
public:
  HeightMapSpoofNode() : Node("height_mapping_spoof_node"),
              tf_buffer_(this->get_clock()), tf_listener_(tf_buffer_),
              mujoco_model_(nullptr), mujoco_data_(nullptr) {

    // --- params ---
    height_mapping::Params P;
    map_frame_    = declare_parameter<std::string>("map_frame", "odom");
    base_frame_   = declare_parameter<std::string>("base_frame", "body");
    topic_cloud_  = declare_parameter<std::string>("topic_cloud", "/cloud_registered");
    lidar_frame_  = declare_parameter<std::string>("livox_frame", "");
    mujoco_xml_file_ = declare_parameter<std::string>("mujoco_xml_file", "stepping_stones.xml");
    z_offset_     = declare_parameter<double>("z_offset", 0.0);

    // MuJoCo scan parameters (for raycasting grid)
    scan_width_      = declare_parameter<int>("scan_width", 50);
    scan_height_     = declare_parameter<int>("scan_height", 50);
    scan_resolution_ = declare_parameter<double>("scan_resolution", 0.05);

    P.res         = declare_parameter<double>("resolution", 0.05);
    P.Wb          = declare_parameter<int>("big_width", 400);
    P.Hb          = declare_parameter<int>("big_height", 400);
    P.max_h       = declare_parameter<double>("max_height", 2.0);
    max_height_   = P.max_h;
    P.z_min       = declare_parameter<double>("z_clip_low", -4.0);
    P.z_max       = declare_parameter<double>("z_clip_high",  4.0);
    P.drop_thresh = declare_parameter<double>("drop_thresh", 0.25);
    P.min_support = declare_parameter<int>("min_support", 2);
    P.shift_thresh= declare_parameter<double>("shift_thresh_m", 0.5);

    P.Wq          = declare_parameter<int>("sub_width", 200);
    P.Hq          = declare_parameter<int>("sub_height", 200);
    P.res_q       = declare_parameter<double>("sub_resolution", P.res);
    P.subgrid_bilinear_interp = declare_parameter<bool>("subgrid_bilinear_interp", true);

    publish_rate_ = declare_parameter<double>("publish_rate_hz", 10.0);
    use_voxel_ds_ = declare_parameter<bool>("voxel_downsample", false);
    voxel_leaf_   = declare_parameter<double>("voxel_leaf", 0.05);
    transform_cloud_if_needed_ = declare_parameter<bool>("transform_cloud", true);

    mapper_ = std::make_shared<height_mapping::HeightMap>(P);

    // Initialize MuJoCo
    initializeMujoco();

    // Create callback groups for multi-threaded execution
    subscriber_cb_group_ = create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);
    timer_cb_group_ = create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);

    // pubs/subs
    pub_raw_  = create_publisher<sensor_msgs::msg::PointCloud2>("height_grid/pub_raw", 1);
    pub_fill_ = create_publisher<sensor_msgs::msg::PointCloud2>("height_grid/pub_filled", 1);
    pub_big_ = create_publisher<sensor_msgs::msg::PointCloud2>("height_grid/pub_big", 1);
    pub_hm_  = create_publisher<sensor_msgs::msg::Image>("height_grid/height_map", 1);
    pub_ground_inliers_ = create_publisher<sensor_msgs::msg::PointCloud2>("ground_plane/inliers", 1);
    enable_pub_big_ = declare_parameter<bool>("enable_pub_big", false);
    enable_pub_raw_ = declare_parameter<bool>("enable_pub_raw", false);
    enable_pub_fill_ = declare_parameter<bool>("enable_pub_fill", true);

    // Subscribe to LiDAR cloud for initial ground-plane estimation
    rclcpp::SubscriptionOptions sub_options;
    sub_options.callback_group = subscriber_cb_group_;

    sub_cloud_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
      topic_cloud_,
      rclcpp::SensorDataQoS(),
      std::bind(&HeightMapSpoofNode::onFirstCloud, this, std::placeholders::_1),
      sub_options);

    // Raycast timer with dedicated callback group (runs in separate thread)
    // This generates the synthetic point cloud via MuJoCo raycasting
    const double raycast_rate = 50.0;  // Hz - can be different from publish rate
    raycast_timer_ = rclcpp::create_timer(
        this,
        this->get_clock(),
        rclcpp::Duration::from_seconds(1.0 / raycast_rate),
        std::bind(&HeightMapSpoofNode::onRaycast, this),
        subscriber_cb_group_);  // Use subscriber callback group for raycasting

    // Publish timer with dedicated callback group
    timer_ = rclcpp::create_timer(
        this,
        this->get_clock(),
        rclcpp::Duration::from_seconds(1.0 / publish_rate_),
        std::bind(&HeightMapSpoofNode::onPublish, this),
        timer_cb_group_);

    RCLCPP_INFO(get_logger(), "height_mapping node up: big %dx%d @%.2f, sub %dx%d @%.2f",
                P.Wb, P.Hb, P.res, P.Wq, P.Hq, P.res_q);
    RCLCPP_INFO(get_logger(), "Constructor completed successfully");
    RCLCPP_INFO(get_logger(), "Waiting for transforms: %s -> %s", map_frame_.c_str(), base_frame_.c_str());
    RCLCPP_INFO(get_logger(), "Subscribed to point cloud topic: %s", topic_cloud_.c_str());
  }

  ~HeightMapSpoofNode() {
    if (mujoco_data_) mj_deleteData(mujoco_data_);
    if (mujoco_model_) mj_deleteModel(mujoco_model_);
  }

private:
  void initializeMujoco() {
    // Construct full path to XML file
    std::string package_share_dir;
    try {
      package_share_dir = ament_index_cpp::get_package_share_directory("height_mapping");
    } catch (const std::exception& e) {
      RCLCPP_ERROR(get_logger(), "Failed to get package share directory: %s", e.what());
      return;
    }

    // Try multiple possible paths
    std::vector<std::string> possible_paths = {
      package_share_dir + "/../../src/height_mapping/rsc/" + mujoco_xml_file_,
      package_share_dir + "/rsc/" + mujoco_xml_file_,
      mujoco_xml_file_  // absolute path
    };

    char error[1000] = "Could not load XML";
    bool loaded = false;

    for (const auto& xml_path : possible_paths) {
      RCLCPP_INFO(get_logger(), "Attempting to load MuJoCo XML from: %s", xml_path.c_str());
      mujoco_model_ = mj_loadXML(xml_path.c_str(), nullptr, error, 1000);
      if (mujoco_model_) {
        RCLCPP_INFO(get_logger(), "Successfully loaded MuJoCo model from: %s", xml_path.c_str());
        loaded = true;
        break;
      }
    }

    if (!loaded) {
      RCLCPP_ERROR(get_logger(), "Failed to load MuJoCo XML file: %s", error);
      return;
    }

    // Create data structure
    mujoco_data_ = mj_makeData(mujoco_model_);
    if (!mujoco_data_) {
      RCLCPP_ERROR(get_logger(), "Failed to create MuJoCo data structure");
      mj_deleteModel(mujoco_model_);
      mujoco_model_ = nullptr;
      return;
    }

    // Forward simulation to initialize
    mj_forward(mujoco_model_, mujoco_data_);

    RCLCPP_INFO(get_logger(), "MuJoCo initialized successfully with z_offset: %.3f", z_offset_);
  }

  double raycastMujoco(double x, double y) {
    if (!mujoco_model_ || !mujoco_data_) {
      return std::numeric_limits<double>::quiet_NaN();
    }

    // Ray origin: start well above the scene
    mjtNum pnt[3] = {x, y, 10.0};

    // Ray direction: straight down
    mjtNum vec[3] = {0.0, 0.0, -1.0};

    // Perform raycast
    int geomid = -1;
    mjtNum dist = mj_ray(mujoco_model_, mujoco_data_, pnt, vec, nullptr, 1, -1, &geomid);

    if (geomid >= 0 && dist >= 0) {
      // Hit point z coordinate
      double z_hit = pnt[2] + dist * vec[2];
      return z_hit + z_offset_;
    }

    // No hit - return NaN
    return std::numeric_limits<double>::quiet_NaN();
  }

  void onRaycast() {
    // Get current robot pose
    double rx, ry, rz, rYaw;
    if (!getRobotPose(rclcpp::Time(0), rx, ry, rz, rYaw)) {
      return;  // No pose available yet
    }

    // Initialize map origin if needed and recenter if robot has moved
    {
      std::lock_guard<std::mutex> lock(mapper_mutex_);
      if (!mapper_->haveOrigin()) {
        mapper_->ensureOrigin(rx, ry, rz);
        RCLCPP_INFO(get_logger(), "Map origin initialized at (%.2f, %.2f, %.2f)", rx, ry, rz);
      }
      mapper_->recenterIfNeeded(rx, ry, rz);
    }

    // Build point cloud using scan parameters (finer discretization)
    std::vector<height_mapping::Point3f> raycast_points;
    raycast_points.reserve(scan_width_ * scan_height_);

    const double half_width = 0.5 * scan_width_ * scan_resolution_;
    const double half_height = 0.5 * scan_height_ * scan_resolution_;

    const double c = std::cos(rYaw);
    const double s = std::sin(rYaw);

    int hit_count = 0;
    for (int i = 0; i < scan_height_; ++i) {
      for (int j = 0; j < scan_width_; ++j) {
        // Grid coordinates in robot frame (centered on robot)
        const double local_x = (static_cast<double>(j) + 0.5) * scan_resolution_ - half_width;
        const double local_y = (static_cast<double>(i) + 0.5) * scan_resolution_ - half_height;

        // World coordinates (rotate by robot yaw and translate by robot position)
        double world_x = rx + c * local_x - s * local_y;
        double world_y = ry + s * local_x + c * local_y;

        // Perform MuJoCo raycast to get z height
        double world_z = raycastMujoco(world_x, world_y);

        if (!std::isnan(world_z)) {
          raycast_points.push_back(height_mapping::Point3f{
            static_cast<float>(world_x),
            static_cast<float>(world_y),
            static_cast<float>(world_z)
          });
          hit_count++;
        }
      }
    }

    // Ingest the raycasted points into the mapper
    if (!raycast_points.empty()) {
      std::lock_guard<std::mutex> lock(mapper_mutex_);
      mapper_->ingestPoints(raycast_points);
      RCLCPP_DEBUG(get_logger(), "Raycasted %d/%d points", hit_count, scan_width_ * scan_height_);
    }
  }

  sensor_msgs::msg::PointCloud2::SharedPtr createPointCloud(
      const cv::Mat& height_mat,
      const height_mapping::SubgridMeta& meta,
      const rclcpp::Time& stamp) {
    auto cloud = std::make_shared<sensor_msgs::msg::PointCloud2>();
    cloud->header.stamp = stamp;
    cloud->header.frame_id = map_frame_;

    // Set up point cloud structure: XYZ + RGB
    cloud->width = meta.width * meta.height;
    cloud->height = 1;
    cloud->is_dense = false;

    // Point cloud fields: x, y, z, rgb
    cloud->fields.resize(4);
    cloud->fields[0].name = "x";
    cloud->fields[0].offset = 0;
    cloud->fields[0].datatype = sensor_msgs::msg::PointField::FLOAT32;
    cloud->fields[0].count = 1;

    cloud->fields[1].name = "y";
    cloud->fields[1].offset = 4;
    cloud->fields[1].datatype = sensor_msgs::msg::PointField::FLOAT32;
    cloud->fields[1].count = 1;

    cloud->fields[2].name = "z";
    cloud->fields[2].offset = 8;
    cloud->fields[2].datatype = sensor_msgs::msg::PointField::FLOAT32;
    cloud->fields[2].count = 1;

    cloud->fields[3].name = "rgb";
    cloud->fields[3].offset = 12;
    cloud->fields[3].datatype = sensor_msgs::msg::PointField::UINT32;
    cloud->fields[3].count = 1;

    cloud->point_step = 16; // 4 floats * 4 bytes
    cloud->row_step = cloud->point_step * cloud->width;
    cloud->data.resize(cloud->row_step);

    // First pass: find min/max heights in valid data
    float min_height = std::numeric_limits<float>::infinity();
    float max_height_data = -std::numeric_limits<float>::infinity();

    for (int i = 0; i < meta.height; ++i) {
      const float* row = height_mat.ptr<float>(i);
      for (int j = 0; j < meta.width; ++j) {
        float height = row[j];
        if (!std::isnan(height) && !std::isinf(height)) {
          min_height = std::min(min_height, height);
          max_height_data = std::max(max_height_data, height);
        }
      }
    }

    // Handle case where no valid data exists
    if (std::isinf(min_height)) {
      min_height = 0.0f;
      max_height_data = 1.0f;
    }

    // Avoid division by zero
    float height_range = max_height_data - min_height;
    if (height_range < 1e-6f) {
      height_range = 1.0f;
    }

    // Convert height map to 3D points with height-based coloring
    int point_idx = 0;
    uint8_t* data_ptr = cloud->data.data();

    for (int i = 0; i < meta.height; ++i) {
      const float* row = height_mat.ptr<float>(i);
      for (int j = 0; j < meta.width; ++j) {
        float height = row[j];

        if (std::isnan(height) || std::isinf(height)) {
          // Skip invalid points
          continue;
        }

        // World coordinates for this grid cell - use same transform as generateSubgrid
        const double halfW = 0.5 * meta.width * meta.resolution;
        const double halfH = 0.5 * meta.height * meta.resolution;
        const double gx = (static_cast<double>(j) + 0.5) * meta.resolution - halfW;
        const double gy = (static_cast<double>(i) + 0.5) * meta.resolution - halfH;

        // Apply rotation (same as generateSubgrid)
        const double c = std::cos(meta.yaw), s = std::sin(meta.yaw);
        // Extract robot position from meta.origin (which is bottom-left corner)
        const double rx = meta.origin_x - c*(-halfW) + s*(-halfH);
        const double ry = meta.origin_y - s*(-halfW) - c*(-halfH);

        double world_x = rx + c*gx - s*gy;
        double world_y = ry + s*gx + c*gy;

        // Get pointer to this point's data
        uint8_t* point_data = data_ptr + point_idx * cloud->point_step;
        float* xyz_ptr = reinterpret_cast<float*>(point_data);
        uint32_t* rgb_ptr = reinterpret_cast<uint32_t*>(point_data + 12);

        // Set XYZ
        xyz_ptr[0] = static_cast<float>(world_x);
        xyz_ptr[1] = static_cast<float>(world_y);
        xyz_ptr[2] = height;

        // Height-based coloring using actual data range: blue (low) to red (high)
        float normalized_height = (height - min_height) / height_range;
        normalized_height = std::max(0.0f, std::min(1.0f, normalized_height));

        uint8_t r = static_cast<uint8_t>(normalized_height * 255);
        uint8_t g = static_cast<uint8_t>((1.0f - normalized_height) * normalized_height * 4 * 255);
        uint8_t b = static_cast<uint8_t>((1.0f - normalized_height) * 255);
        uint32_t rgb = (static_cast<uint32_t>(r) << 16) |
                      (static_cast<uint32_t>(g) << 8) |
                      static_cast<uint32_t>(b);

        *rgb_ptr = rgb;
        point_idx++;
      }
    }

    // Update actual point count
    cloud->width = point_idx;
    cloud->data.resize(point_idx * cloud->point_step);

    return cloud;
  }

  bool getRobotPose(const rclcpp::Time& t, double& x, double& y, double& z, double& yaw) {
    try {
      // RCLCPP_DEBUG(get_logger(), "Attempting TF lookup: %s -> %s", map_frame_.c_str(), base_frame_.c_str());
      // Try with exact timestamp first
      auto T = tf_buffer_.lookupTransform(map_frame_, base_frame_, t, tf2::durationFromSec(0.1));
      x = T.transform.translation.x;
      y = T.transform.translation.y;
      z = T.transform.translation.z;
      const auto &q = T.transform.rotation;
      tf2::Quaternion qq(q.x, q.y, q.z, q.w);
      tf2::Matrix3x3 R(qq);
      double roll, pitch; R.getRPY(roll, pitch, yaw);
      // RCLCPP_DEBUG(get_logger(), "TF lookup success: pose (%.2f, %.2f, %.2f, %.2f)", x, y, z, yaw);
      return true;
    } catch (const tf2::ExtrapolationException& e) {
      // If extrapolation fails, try with latest available transform
      try {
        RCLCPP_DEBUG(get_logger(), "Extrapolation failed, trying latest transform");
        auto T = tf_buffer_.lookupTransform(map_frame_, base_frame_, tf2::TimePointZero);
        x = T.transform.translation.x;
        y = T.transform.translation.y;
        z = T.transform.translation.z;
        const auto &q = T.transform.rotation;
        tf2::Quaternion qq(q.x, q.y, q.z, q.w);
        tf2::Matrix3x3 R(qq);
        double roll, pitch; R.getRPY(roll, pitch, yaw);
        RCLCPP_DEBUG(get_logger(), "TF lookup success with latest: pose (%.2f, %.2f, %.2f, %.2f)", x, y, z, yaw);
        return true;
      } catch (const std::exception& e2) {
        RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000, "TF lookup failed (latest): %s", e2.what());
        return false;
      }
    } catch (const std::exception& e) { 
      RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000, "TF lookup failed: %s", e.what());
      return false; 
    }
  }

  void fillHeightImageMsg(const cv::Mat& map,
                        float rz,
                        const rclcpp::Time& stamp,
                        const std::string& frame_id,
                        sensor_msgs::msg::Image& msg)
  {
    using sensor_msgs::msg::Image;

    if (map.type() != CV_32FC1) {
      throw std::runtime_error("fillHeightImageMsg: map must be CV_32FC1");
    }

    msg.header.stamp = stamp;
    msg.header.frame_id = frame_id;

    msg.height = map.rows;
    msg.width  = map.cols;
    msg.encoding = "32FC1";   // 1-channel float32
    msg.is_bigendian = false;
    msg.step = msg.width * sizeof(float);

    msg.data.resize(msg.height * msg.step);
    float* dst = reinterpret_cast<float*>(msg.data.data());

    // if (map.isContinuous()) {
    //   const float* src = map.ptr<float>(0);
    //   int N = map.rows * map.cols;
    //   for (int i = 0; i < N; ++i) {
    //     dst[i] = rz - src[i];  // h = rz - map
    //     RCLCPP_INFO(get_logger(), "%0.4f, %0.4f, %0.4f", dst[i], rz, src[i]);
    //   }
    // } else {
    int idx = 0;
    for (int r = 0; r < map.rows; ++r) {
      const float* row = map.ptr<float>(r);
      for (int c = 0; c < map.cols; ++c, ++idx) {
        dst[idx] = rz - row[c];
        // RCLCPP_INFO(get_logger(), "%0.4f, %0.4f, %0.4f", dst[idx], rz, row[c]);
      }
    }
    // }
  }

  void onPublish() {
    RCLCPP_DEBUG(get_logger(), "Publish timer callback triggered");
    auto t0 = now();

    // Get current robot pose
    double rx, ry, rz, rYaw;
    if (!getRobotPose(rclcpp::Time(0), rx, ry, rz, rYaw)) {
      RCLCPP_DEBUG(get_logger(), "Skipping publish - no robot pose");
      return;
    }

    // Generate subgrid and big map with mutex protection
    cv::Mat pub_raw_map, pub_filled_map;
    height_mapping::SubgridMeta meta, big_meta;
    cv::Mat pub_big_map, tmp1, tmp2;

    {
      std::lock_guard<std::mutex> lock(mapper_mutex_);

      // Check if map is initialized
      if (!mapper_->haveOrigin()) {
        RCLCPP_DEBUG(get_logger(), "Map not initialized yet");
        return;
      }

      // Generate subgrid for current pose
      mapper_->generateSubgrid(rx, ry, rYaw, pub_raw_map, pub_filled_map, meta);

      // Get big map if needed
      if (enable_pub_big_) {
        big_meta = mapper_->getBigMapMeta();
        mapper_->snapshotBig(pub_big_map, tmp1, tmp2);
      }
    }

    // Publish all messages (outside of mutex lock)
    auto stamp = now();

    // Publish height image
    sensor_msgs::msg::Image img_msg;
    fillHeightImageMsg(pub_filled_map, static_cast<float>(rz), stamp, base_frame_, img_msg);
    pub_hm_->publish(img_msg);

    // Publish subgrid point clouds
    if (enable_pub_fill_) {
      pub_fill_->publish(*createPointCloud(pub_filled_map, meta, stamp));
    }
    if (enable_pub_raw_) {
      pub_raw_->publish(*createPointCloud(pub_raw_map, meta, stamp));
    }

    // Publish big map point cloud
    if (enable_pub_big_) {
      pub_big_->publish(*createPointCloud(pub_big_map, big_meta, stamp));
    }

    double elapsed_ms = (now() - t0).seconds() * 1000.0;
    RCLCPP_DEBUG(get_logger(), "Published height maps in %.6f ms", elapsed_ms);
  }

  void onFirstCloud(const sensor_msgs::msg::PointCloud2::SharedPtr msg)
  {
    if (have_ground_plane_) {
      return;
    }

    // --- 1. Figure out source frame and lookup TF map <- src ---
    geometry_msgs::msg::TransformStamped T_map_src;
    bool do_tf = false;

    const std::string src_frame =
        msg->header.frame_id.empty() ? lidar_frame_ : msg->header.frame_id;

    if (transform_cloud_if_needed_ && !src_frame.empty() && src_frame != map_frame_) {
      try {
        T_map_src = tf_buffer_.lookupTransform(
            map_frame_, src_frame, msg->header.stamp, tf2::durationFromSec(0.1));
        do_tf = true;
      } catch (const std::exception& e) {
        RCLCPP_WARN(get_logger(),
                    "Ground plane: TF missing %s->%s: %s",
                    map_frame_.c_str(), src_frame.c_str(), e.what());
        return;
      }
    }

    // --- 2. Build rotation + translation (map <- src) ---
    double r00 = 1, r01 = 0, r02 = 0, tx = 0;
    double r10 = 0, r11 = 1, r12 = 0, ty = 0;
    double r20 = 0, r21 = 0, r22 = 1, tz = 0;

    if (do_tf) {
      const auto &t = T_map_src.transform.translation;
      const auto &q = T_map_src.transform.rotation;
      tf2::Quaternion qq(q.x, q.y, q.z, q.w);
      tf2::Matrix3x3 R(qq);
      r00 = R[0][0]; r01 = R[0][1]; r02 = R[0][2]; tx = t.x;
      r10 = R[1][0]; r11 = R[1][1]; r12 = R[1][2]; ty = t.y;
      r20 = R[2][0]; r21 = R[2][1]; r22 = R[2][2]; tz = t.z;
    }

    // --- 3. Iterate PointCloud2, transform to map frame, keep z <= 0 ---
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
    cloud->reserve(msg->width * msg->height);

    sensor_msgs::PointCloud2ConstIterator<float> it_x(*msg, "x");
    sensor_msgs::PointCloud2ConstIterator<float> it_y(*msg, "y");
    sensor_msgs::PointCloud2ConstIterator<float> it_z(*msg, "z");

    for (; it_x != it_x.end(); ++it_x, ++it_y, ++it_z) {
      double x = *it_x;
      double y = *it_y;
      double z = *it_z;

      if (do_tf) {
        const double xx = r00 * x + r01 * y + r02 * z + tx;
        const double yy = r10 * x + r11 * y + r12 * z + ty;
        const double zz = r20 * x + r21 * y + r22 * z + tz;
        x = xx; y = yy; z = zz;
      }

      // Reject points above z = 0 in odom/map frame
      if (z <= -0.5 && std::sqrt(x*x + y*y) < 2.0) {  // also limit to 3m radius
        cloud->push_back(pcl::PointXYZ(
            static_cast<float>(x),
            static_cast<float>(y),
            static_cast<float>(z)));
      }
    }

    cloud->width = cloud->size();
    cloud->height = 1;
    cloud->is_dense = false;

    if (cloud->empty()) {
      RCLCPP_WARN(get_logger(),
                  "Ground plane: no points left after TF + z<=0 filtering.");
      return;
    }

    // --- 4. Optional voxel downsample before RANSAC ---
    if (use_voxel_ds_) {
      pcl::VoxelGrid<pcl::PointXYZ> vg;
      vg.setInputCloud(cloud);
      vg.setLeafSize(voxel_leaf_, voxel_leaf_, voxel_leaf_);
      pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_ds(new pcl::PointCloud<pcl::PointXYZ>);
      vg.filter(*cloud_ds);
      cloud = cloud_ds;
    }

    // --- 5. RANSAC plane fit ---
    pcl::SACSegmentation<pcl::PointXYZ> seg;
    seg.setOptimizeCoefficients(true);
    seg.setModelType(pcl::SACMODEL_PLANE);
    seg.setMethodType(pcl::SAC_RANSAC);
    seg.setDistanceThreshold(0.02);  // tune as needed
    seg.setMaxIterations(1000);

    pcl::ModelCoefficients::Ptr coeff(new pcl::ModelCoefficients);
    pcl::PointIndices::Ptr inliers(new pcl::PointIndices);
    seg.setInputCloud(cloud);
    seg.segment(*inliers, *coeff);

    if (coeff->values.size() != 4) {
      RCLCPP_WARN(get_logger(),
                  "Ground plane: failed to estimate plane, coeff size = %zu",
                  coeff->values.size());
      return;
    }

    // --- DEBUG: publish inlier points as a separate cloud ---
    if (!inliers->indices.empty()) {
      pcl::PointCloud<pcl::PointXYZ>::Ptr inlier_cloud(new pcl::PointCloud<pcl::PointXYZ>);
      inlier_cloud->reserve(inliers->indices.size());
      for (int idx : inliers->indices) {
        if (idx >= 0 && static_cast<size_t>(idx) < cloud->size()) {
          inlier_cloud->push_back((*cloud)[idx]);
        }
      }
      inlier_cloud->width  = inlier_cloud->size();
      inlier_cloud->height = 1;
      inlier_cloud->is_dense = false;

      sensor_msgs::msg::PointCloud2 inlier_msg;
      pcl::toROSMsg(*inlier_cloud, inlier_msg);
      inlier_msg.header.frame_id = map_frame_;          // we're already in odom/map frame
      inlier_msg.header.stamp    = msg->header.stamp;
      pub_ground_inliers_->publish(inlier_msg);

      RCLCPP_INFO(get_logger(),
                  "Ground plane: publishing %zu inliers on 'ground_plane/inliers'",
                  inlier_cloud->size());
    } else {
      RCLCPP_WARN(get_logger(), "Ground plane: RANSAC returned 0 inliers.");
    }


    const float a = coeff->values[0];
    const float b = coeff->values[1];
    const float c = coeff->values[2];
    const float d = coeff->values[3];

    if (std::abs(c) < 1e-3f) {
      RCLCPP_WARN(get_logger(),
                  "Ground plane: plane normal too vertical (c = %f).", c);
      return;
    }

    // Ground height at (x,y) = (0,0) in map_frame_/odom
    const double ground_z0 = -d / c;

    // Also check base height relative to that ground
    double rx0, ry0, rz0, rYaw0;
    if (getRobotPose(msg->header.stamp, rx0, ry0, rz0, rYaw0)) {
      const double base_above_ground = rz0 - ground_z0;
      RCLCPP_INFO(get_logger(),
                  "Ground plane: z0=%.3f, base z=%.3f, base above ground=%.3f",
                  ground_z0, rz0, base_above_ground);
    } else {
      RCLCPP_INFO(get_logger(),
                  "Ground plane: z0=%.3f (base pose unavailable)", ground_z0);
    }

    // --- 6. Set MuJoCo z offset and mark as done ---
    z_offset_ = ground_z0;
    have_ground_plane_ = true;

    RCLCPP_INFO(get_logger(),
                "Ground plane estimated: %.3f x + %.3f y + %.3f z + %.3f = 0, "
                "setting z_offset_ = %.3f",
                a, b, c, d, z_offset_);

    // Drop subscription; we only needed the first cloud
    sub_cloud_.reset();
  }

private:
  // ROS
  std::string map_frame_, base_frame_, topic_cloud_, lidar_frame_;
  bool enable_pub_big_, enable_pub_raw_, enable_pub_fill_;
  double publish_rate_{10.0};
  bool use_voxel_ds_{false}, transform_cloud_if_needed_{true};
  double voxel_leaf_{0.05};
  double max_height_{2.0};
  bool have_ground_plane_{false};

  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pub_raw_, pub_fill_, pub_big_;
  rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr pub_hm_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pub_ground_inliers_;

  rclcpp::TimerBase::SharedPtr timer_;
  rclcpp::TimerBase::SharedPtr raycast_timer_;
  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr sub_cloud_;

  // Callback groups for multi-threaded execution
  rclcpp::CallbackGroup::SharedPtr subscriber_cb_group_;
  rclcpp::CallbackGroup::SharedPtr timer_cb_group_;

  tf2_ros::Buffer tf_buffer_;
  tf2_ros::TransformListener tf_listener_;

  // Core mapper (thread-safe access via mutex)
  std::shared_ptr<height_mapping::HeightMap> mapper_;
  std::mutex mapper_mutex_;

  // MuJoCo
  mjModel* mujoco_model_;
  mjData* mujoco_data_;
  std::string mujoco_xml_file_;
  double z_offset_;

  // MuJoCo scan parameters
  int scan_width_;
  int scan_height_;
  double scan_resolution_;
};

int main(int argc, char** argv) {
  rclcpp::init(argc, argv);

  // Create node
  auto node = std::make_shared<HeightMapSpoofNode>();

  // Create multi-threaded executor with 2 threads
  // This allows subscriber and timer callbacks to run concurrently
  rclcpp::executors::MultiThreadedExecutor executor(rclcpp::ExecutorOptions(), 2);
  executor.add_node(node);

  RCLCPP_INFO(node->get_logger(), "Starting multi-threaded executor with 2 threads");
  executor.spin();

  rclcpp::shutdown();
  return 0;
}
