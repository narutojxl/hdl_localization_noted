#ifndef ODOM_SYSTEM_HPP
#define ODOM_SYSTEM_HPP

#include <kkl/alg/unscented_kalman_filter.hpp>

namespace hdl_localization {

/**
 * @brief This class models the sensor pose estimation based on robot odometry
 * @note state = [px, py, pz, qw, qx, qy, qz]
 *       observation = [px, py, pz, qw, qx, qy, qz]
 *       maybe better to use expmap
 */
class OdomSystem {
public:
  typedef float T;
  typedef Eigen::Matrix<T, 3, 1> Vector3t;
  typedef Eigen::Matrix<T, 4, 1> Vector4t;
  typedef Eigen::Matrix<T, 4, 4> Matrix4t;
  typedef Eigen::Matrix<T, Eigen::Dynamic, 1> VectorXt;
  typedef Eigen::Quaternion<T> Quaterniont;

public:
  // system equation 
  VectorXt f(const VectorXt& state, //从k时刻状态中提取的每个sigma points， 因为此时odom_ukf的mean是用常量速度模型或者imu meas预测出来当前帧时刻, 即k状态的均值
             const VectorXt& control) const {//laser k-1时刻到 laser k时刻的delta变换
    Matrix4t pt = Matrix4t::Identity();
    pt.block<3, 1>(0, 3) = Vector3t(state[0], state[1], state[2]);
    pt.block<3, 3>(0, 0) = Quaterniont(state[3], state[4], state[5], state[6]).normalized().toRotationMatrix();

    Matrix4t delta = Matrix4t::Identity();
    delta.block<3, 1>(0, 3) = Vector3t(control[0], control[1], control[2]);
    delta.block<3, 3>(0, 0) = Quaterniont(control[3], control[4], control[5], control[6]).normalized().toRotationMatrix();

    Matrix4t pt_ = pt * delta; //TODO(jxl): 感觉这不对。创建odom_ukf的时候，state应该为ukf k-1时刻的mean，而不是已经经过常量速度模型或者imu meas预测的k时刻状态
    Quaterniont quat_(pt_.block<3, 3>(0, 0));

    VectorXt next_state(7);
    next_state.head<3>() = pt_.block<3, 1>(0, 3);
    next_state.tail<4>() = Vector4t(quat_.w(), quat_.x(), quat_.y(), quat_.z());

    return next_state;
  }

  // observation equation
  VectorXt h(const VectorXt& state) const {
    return state;
  }
};

}  // namespace hdl_localization

#endif  // POSE_SYSTEM_HPP
