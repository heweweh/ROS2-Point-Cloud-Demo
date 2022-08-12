import sys
import os

import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, QoSReliabilityPolicy, qos_profile_sensor_data
from tf_transformations import quaternion_from_euler
import sensor_msgs.msg as sensor_msgs
import numpy as np
from autoware_auto_perception_msgs.msg import BoundingBoxArray
from autoware_auto_perception_msgs.msg import BoundingBox
from ament_index_python.packages import get_package_share_directory

import ros2_numpy as rnp

import torch
from det3d import torchie
from det3d.models import build_detector
from det3d.torchie.parallel import MegDataParallel
from det3d.torchie.trainer import load_checkpoint

from det3d.datasets.pipelines import Compose
from det3d.torchie.parallel import collate_kitti
from det3d.torchie.trainer.trainer import example_to_device

REFERENCE_LIDAR_HEIGHT = 1.73 # the height of velydone HDL-64E

class PCDListener(Node):

    def __init__(self):
        super().__init__('pcd_subsriber_node')

        config_dir = os.path.join(get_package_share_directory(
            'pcd_demo'), 'config', 'config.py')
        cfg = torchie.Config.fromfile(config_dir)

        self.preprocess = Compose(cfg.online_execute_pipeline)
        checkpoint_path = os.path.join(get_package_share_directory(
            'pcd_demo'), 'checkpoint', 'latest_ema.pth')
        model_ = build_detector(cfg.model, train_cfg=None, test_cfg=cfg.test_cfg)
        checkpoint = load_checkpoint(model_, checkpoint_path, map_location="cpu")
        self.model = MegDataParallel(model_, device_ids=[0])

        # Set up a subscription to the 'pcd' topic with a callback to the
        # function `listener_callback`
        self.pcd_subscriber = self.create_subscription(
            sensor_msgs.PointCloud2,    # Msg type
            'pcd_in',                      # topic
            self.listener_callback,      # Function to call
            10                          # QoS
        )

        self.bbox_publisher = self.create_publisher(BoundingBoxArray,
                                                    'pcd_bbox_out',
                                                    QoSProfile(
                                                        depth=1,
                                                        reliability=QoSReliabilityPolicy.RELIABLE,
                                                        durability=DurabilityPolicy.VOLATILE,
                                                        history=HistoryPolicy.KEEP_LAST
                                                    ))

        self.pcd_publisher = self.create_publisher(sensor_msgs.PointCloud2,
                                                    'pcd_converted_twice',
                                                    QoSProfile(
                                                        depth=1,
                                                        reliability=QoSReliabilityPolicy.RELIABLE,
                                                        durability=DurabilityPolicy.VOLATILE,
                                                        history=HistoryPolicy.KEEP_LAST
                                                    ))


    def listener_callback(self, msg):
        # Here we convert the 'msg', which is of the type PointCloud2.
        # I ported the function read_points2 from
        # the ROS1 package.
        # https://github.com/ros/common_msgs/blob/noetic-devel/sensor_msgs/src/sensor_msgs/point_cloud2.py

        structed_array = rnp.numpify(msg)
        structed_array['z'] -= REFERENCE_LIDAR_HEIGHT
        unstructed_array = structed_array.view('f4').reshape(-1,4)

        pcd_c_tw = rnp.msgify(sensor_msgs.PointCloud2, structed_array)
        pcd_c_tw.header.frame_id = "ego_vehicle"
        self.pcd_publisher.publish(pcd_c_tw)

        res = {
            "lidar": {
                "type": "lidar",
                "points": unstructed_array,
                "targets": None,      # include cls_labels & reg_targets
            },
            "mode": "test",
            "metadata": None
        }
        preproceed_data, _ = self.preprocess(res, None)
        example = example_to_device(collate_kitti([preproceed_data]), device=torch.device('cuda'))

        with torch.no_grad():
            # outputs: predicted results in lidar coord.
            outputs = self.model(example, return_loss=False, rescale=True)

        output = outputs[0]
        bboxes = output['box3d_lidar'].cpu().numpy()
        scores = output['scores'].cpu().numpy()
        label_preds = output['label_preds'].cpu().numpy()

        temp_index = np.argsort(-scores)
        #print(f"scores: {scores[temp_index]} bboxes: {bboxes[temp_index]}")

        filters = scores > 0.35
        filtered_bboxes = bboxes[filters]
        filtered_bboxes[:,3] += REFERENCE_LIDAR_HEIGHT # adjust z of bounding box
        filtered_scores = scores[filters]
        filtered_labels = label_preds[filters]

        print(filtered_bboxes, filtered_scores, filtered_labels)

        self.publish_bboxes(msg,
                            zip(filtered_scores.astype(float),
                                filtered_bboxes.astype(float),
                                filtered_labels.tolist()))


    def publish_bboxes(self, msg, lst):

        bboxarr = BoundingBoxArray()
        bboxarr.header = msg.header
        for pscore, pbbox, plabel in lst:
            bbox = BoundingBox()
            bbox.centroid.x = pbbox[0]
            bbox.centroid.y = pbbox[1]
            bbox.centroid.z = pbbox[2]
            bbox.size.x = pbbox[3]
            bbox.size.y = pbbox[4]
            bbox.size.z = pbbox[5]
            q = quaternion_from_euler(0, 0, pbbox[6])  # prevent the data from being overwritten
            bbox.orientation.x = q[0]
            bbox.orientation.y = q[1]
            bbox.orientation.z = q[2]
            bbox.orientation.w = q[3]
            bbox.variance = [0., 0., 0., 0., 0., 0., 0., 0.]
            bbox.value = pscore
            bbox.vehicle_label = plabel+1
            bbox.class_likelihood = pscore
            bboxarr.boxes.append(bbox)
        self.bbox_publisher.publish(bboxarr)


def main(args=None):
    # Boilerplate code.
    rclpy.init(args=args)
    pcd_listener = PCDListener()
    rclpy.spin(pcd_listener)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    pcd_listener.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
