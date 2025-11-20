#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <sensor_msgs/point_cloud2_iterator.hpp>
#include <geometry_msgs/msg/transform_stamped.hpp>

#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2/LinearMath/Matrix3x3.h>

#include <opencv2/core.hpp>

#include <pcl/point_types.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl_conversions/pcl_conversions.h>

#include "HeightMapping.hpp"

class HeightMapNode : public rclcpp::Node {
public:
  HeightMapNode() : Node("height_mapping_node"),
              tf_buffer_(this->get_clock()), tf_listener_(tf_buffer_) {

    // --- params ---
    height_mapping::Params P;
    map_frame_    = declare_parameter<std::string>("map_frame", "odom");
    base_frame_   = declare_parameter<std::string>("base_frame", "body");
    topic_cloud_  = declare_parameter<std::string>("topic_cloud", "/cloud_registered");
    lidar_frame_  = declare_parameter<std::string>("livox_frame", "");

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

    // Create callback groups for multi-threaded execution
    subscriber_cb_group_ = create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);
    timer_cb_group_ = create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);

    // pubs/subs
    pub_raw_  = create_publisher<sensor_msgs::msg::PointCloud2>("height_grid/pub_raw", 1);
    pub_fill_ = create_publisher<sensor_msgs::msg::PointCloud2>("height_grid/pub_filled", 1);
    pub_big_ = create_publisher<sensor_msgs::msg::PointCloud2>("height_grid/pub_big", 1);
    pub_hm_  = create_publisher<sensor_msgs::msg::Image>("height_grid/height_map", 1);
    enable_pub_big_ = declare_parameter<bool>("enable_pub_big", false);
    enable_pub_raw_ = declare_parameter<bool>("enable_pub_raw", false);
    enable_pub_fill_ = declare_parameter<bool>("enable_pub_fill", true);

    // Subscription options with dedicated callback group
    auto sub_options = rclcpp::SubscriptionOptions();
    sub_options.callback_group = subscriber_cb_group_;

    sub_cloud_ = create_subscription<sensor_msgs::msg::PointCloud2>(
      topic_cloud_, rclcpp::SensorDataQoS(),
      std::bind(&HeightMapNode::cloudCb, this, std::placeholders::_1),
      sub_options);

    // timer with dedicated callback group
    // Use rclcpp::create_timer to respect sim time for bag playback
    const auto period = std::chrono::duration<double>(1.0 / publish_rate_);
    timer_ = rclcpp::create_timer(
        this,
        this->get_clock(),
        rclcpp::Duration::from_seconds(1.0 / publish_rate_),
        std::bind(&HeightMapNode::onPublish, this),
        timer_cb_group_);

    RCLCPP_INFO(get_logger(), "height_mapping node up: big %dx%d @%.2f, sub %dx%d @%.2f",
                P.Wb, P.Hb, P.res, P.Wq, P.Hq, P.res_q);
    RCLCPP_INFO(get_logger(), "Constructor completed successfully");
    RCLCPP_INFO(get_logger(), "Waiting for transforms: %s -> %s", map_frame_.c_str(), base_frame_.c_str());
    RCLCPP_INFO(get_logger(), "Subscribed to point cloud topic: %s", topic_cloud_.c_str());
  }

private:
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

  void cloudCb(const sensor_msgs::msg::PointCloud2::SharedPtr msg) {
    RCLCPP_DEBUG_THROTTLE(get_logger(), *get_clock(), 2000, "Received point cloud with %d points", 
                         msg->width * msg->height);
    auto t0 = now();
    // TF: ensure origin & recenter
    double rx, ry, rz, rYaw;
    if (!getRobotPose(msg->header.stamp, rx, ry, rz, rYaw)) {
      RCLCPP_DEBUG(get_logger(), "Skipping cloud processing - no robot pose");
      return;
    }
    // RCLCPP_DEBUG(get_logger(), "Processing cloud at robot pose (%.2f, %.2f, %.2f, %.2f)", rx, ry, rz, rYaw);
    
    mapper_->ensureOrigin(rx, ry, rz);
    mapper_->recenterIfNeeded(rx, ry, rz);

    // Optional voxel DS (still in ROS layer)
    const sensor_msgs::msg::PointCloud2 *cloud_in = msg.get();
    sensor_msgs::msg::PointCloud2 cloud_ds;

    if (use_voxel_ds_) {
        pcl::PCLPointCloud2 pcl_pc2;
        pcl_conversions::toPCL(*msg, pcl_pc2);

        pcl::PCLPointCloud2::Ptr pcl_in(new pcl::PCLPointCloud2(pcl_pc2));  // <- PCL-compatible Ptr
        pcl::PCLPointCloud2 pcl_out;

        pcl::VoxelGrid<pcl::PCLPointCloud2> vg;
        vg.setInputCloud(pcl_in);
        vg.setLeafSize(voxel_leaf_, voxel_leaf_, voxel_leaf_);
        vg.filter(pcl_out);

        pcl_conversions::fromPCL(pcl_out, cloud_ds);
        cloud_in = &cloud_ds;
    }

    // Transform to map if required (on-the-fly numeric TF)
    geometry_msgs::msg::TransformStamped T_map_src;
    bool do_tf = false;
    if (transform_cloud_if_needed_) {
      const std::string src_frame = cloud_in->header.frame_id.empty() ? lidar_frame_ : cloud_in->header.frame_id;
      if (!src_frame.empty() && src_frame != map_frame_) {
        try {
          T_map_src = tf_buffer_.lookupTransform(map_frame_, src_frame, msg->header.stamp, tf2::durationFromSec(0.1));
          do_tf = true;
        } catch (...) {
          RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 1000, "TF missing %s->%s", map_frame_.c_str(), src_frame.c_str());
        }
      }
    }

    double r00=1, r01=0, r02=0, tx=0;
    double r10=0, r11=1, r12=0, ty=0;
    double r20=0, r21=0, r22=1, tz=0;
    if (do_tf) {
      const auto &t = T_map_src.transform.translation;
      const auto &q = T_map_src.transform.rotation;
      tf2::Quaternion qq(q.x, q.y, q.z, q.w);
      tf2::Matrix3x3 R(qq);
      r00 = R[0][0]; r01 = R[0][1]; r02 = R[0][2]; tx = t.x;
      r10 = R[1][0]; r11 = R[1][1]; r12 = R[1][2]; ty = t.y;
      r20 = R[2][0]; r21 = R[2][1]; r22 = R[2][2]; tz = t.z;
    }

    // Convert cloud to vector<Point3f> in map frame
    std::vector<height_mapping::Point3f> pts;
    pts.reserve(cloud_in->width * cloud_in->height);

    sensor_msgs::PointCloud2ConstIterator<float> it_x(*cloud_in, "x");
    sensor_msgs::PointCloud2ConstIterator<float> it_y(*cloud_in, "y");
    sensor_msgs::PointCloud2ConstIterator<float> it_z(*cloud_in, "z");

    for (; it_x != it_x.end(); ++it_x, ++it_y, ++it_z) {
      double x = *it_x, y = *it_y, z = *it_z;
      if (do_tf) {
        const double xx = r00*x + r01*y + r02*z + tx;
        const double yy = r10*x + r11*y + r12*z + ty;
        const double zz = r20*x + r21*y + r22*z + tz;
        x = xx; y = yy; z = zz;
      }
      pts.push_back(height_mapping::Point3f{static_cast<float>(x), static_cast<float>(y), static_cast<float>(z)});
    }

    mapper_->ingestPoints(pts);
    RCLCPP_DEBUG(get_logger(), "Ingesting %zu points completed in %0.6f ms", pts.size(), (now() - t0).seconds() * 1000.0);
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
        RCLCPP_INFO(get_logger(), "%0.4f, %0.4f, %0.4f", dst[idx], rz, row[c]);
      }
    }
    // }
  }

  void onPublish() {
    RCLCPP_DEBUG(get_logger(), "Publish timer callback triggered");
    auto t0 = now();

    // Pose at publish time - use latest available transform (Time(0))
    // This avoids extrapolation issues when timer drifts ahead of data
    double rx, ry, rz, rYaw;
    if (!getRobotPose(rclcpp::Time(0), rx, ry, rz, rYaw)) {
      RCLCPP_INFO(get_logger(), "Skipping publish - no robot pose");
      return;
    }
    if (!mapper_->haveOrigin()) {
      RCLCPP_INFO(get_logger(), "Skipping publish - no map origin set");
      return;
    }

    // RCLCPP_DEBUG(get_logger(), "Generating subgrid at pose (%.2f, %.2f, %.2f, %.2f)", rx, ry, rz, rYaw);
    cv::Mat pub_raw_map, pub_filled_map;
    height_mapping::SubgridMeta meta;
    mapper_->generateSubgrid(rx, ry, rYaw, pub_raw_map, pub_filled_map, meta);
    // RCLCPP_DEBUG(get_logger(), "Subgrid generated: %dx%d", meta.width, meta.height);

    // Publish height point clouds
    auto stamp = now();
    // Publish rz - pub_filled_map as a 2D array using pub_hm_
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

    // Publish big map point cloud with its own metadata
    if (enable_pub_big_) {
      cv::Mat pub_big_map, tmp1, tmp2;
      height_mapping::SubgridMeta big_meta = mapper_->getBigMapMeta();
      mapper_->snapshotBig(pub_big_map, tmp1, tmp2);
      pub_big_->publish(*createPointCloud(pub_big_map, big_meta, stamp));
    }

    double elapsed_ms = (now() - t0).seconds() * 1000.0;
    RCLCPP_DEBUG(get_logger(), "Published height maps in %.6f ms", elapsed_ms);
  }

private:
  // ROS
  std::string map_frame_, base_frame_, topic_cloud_, lidar_frame_;
  bool enable_pub_big_, enable_pub_raw_, enable_pub_fill_;
  double publish_rate_{10.0};
  bool use_voxel_ds_{false}, transform_cloud_if_needed_{true};
  double voxel_leaf_{0.05};
  double max_height_{2.0};

  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr sub_cloud_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pub_raw_, pub_fill_, pub_big_;
  rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr pub_hm_;
  rclcpp::TimerBase::SharedPtr timer_;

  // Callback groups for multi-threaded execution
  rclcpp::CallbackGroup::SharedPtr subscriber_cb_group_;
  rclcpp::CallbackGroup::SharedPtr timer_cb_group_;

  tf2_ros::Buffer tf_buffer_;
  tf2_ros::TransformListener tf_listener_;

  // Core mapper
  std::shared_ptr<height_mapping::HeightMap> mapper_;
};

int main(int argc, char** argv) {
  rclcpp::init(argc, argv);

  // Create node
  auto node = std::make_shared<HeightMapNode>();

  // Create multi-threaded executor with 2 threads
  // This allows subscriber and timer callbacks to run concurrently
  rclcpp::executors::MultiThreadedExecutor executor(rclcpp::ExecutorOptions(), 2);
  executor.add_node(node);

  RCLCPP_INFO(node->get_logger(), "Starting multi-threaded executor with 2 threads");
  executor.spin();

  rclcpp::shutdown();
  return 0;
}
