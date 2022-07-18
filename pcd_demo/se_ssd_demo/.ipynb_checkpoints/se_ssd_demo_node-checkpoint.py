import sys
import os

import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, QoSReliabilityPolicy, qos_profile_sensor_data
import sensor_msgs.msg as sensor_msgs
import numpy as np
from autoware_auto_perception_msgs.msg import BoundingBoxArray
from autoware_auto_perception_msgs.msg import BoundingBox
from ament_index_python.packages import get_package_share_directory

import torch
from det3d import torchie
from det3d.models import build_detector
from det3d.torchie.parallel import MegDataParallel
from det3d.torchie.trainer import load_checkpoint

from det3d.datasets.pipelines import Compose
from det3d.torchie.parallel import collate_kitti
from det3d.torchie.trainer.trainer import example_to_device

class PCDListener(Node):

    def __init__(self):
        super().__init__('pcd_subsriber_node')

        config_dir = os.path.join(get_package_share_directory(
            'pcd_demo'), 'config', 'config.py')
        cfg = torchie.Config.fromfile(config_dir)

        self.preprocess = Compose(cfg.online_execute_pipeline)
        checkpoint_path = os.path.join(get_package_share_directory(
            'pcd_demo'), 'checkpoint', 'se-ssd-model.pth')
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



    def listener_callback(self, msg):
        # Here we convert the 'msg', which is of the type PointCloud2.
        # I ported the function read_points2 from
        # the ROS1 package.
        # https://github.com/ros/common_msgs/blob/noetic-devel/sensor_msgs/src/sensor_msgs/point_cloud2.py

        pcd_as_numpy_array = np.array(list(read_points(msg)), dtype=np.float32)
        res = {
            "lidar": {
                "type": "lidar",
                "points": pcd_as_numpy_array,
                "targets": None,      # include cls_labels & reg_targets
            },
            "mode": "test",
            "metadata": None
        }
        preproceed_data, _ = self.preprocess(res, None)
        example = collate_kitti([preproceed_data])

        with torch.no_grad():
            # outputs: predicted results in lidar coord.
            outputs = self.model(example, return_loss=False, rescale=True)

        output = outputs[0]
        bboxes = output['box3d_lidar'].cpu().numpy()
        scores = output['scores'].cpu().numpy()
        label_preds = output['label_preds'].cpu().numpy()
        
        temp_index = np.argsort(-scores)
        print(f"scores: {scores[temp_index]} bboxes: {bboxes[temp_index]}")

        filters = scores > 0.25
        self.publish_bboxes(msg,
                            zip(scores[filters].astype(float),
                            bboxes[filters].astype(float),
                            label_preds[filters].astype(int)))


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
            bbox.variance = [0., 0., 0., 0., 0., 0., 0., 0.]
            bbox.value = pscore
            bbox.vehicle_label = 1
            bbox.class_likelihood = pscore
            bboxarr.boxes.append(bbox)
        self.bbox_publisher.publish(bboxarr)



## The code below is "ported" from
# https://github.com/ros/common_msgs/tree/noetic-devel/sensor_msgs/src/sensor_msgs
import sys
from collections import namedtuple
import ctypes
import math
import struct
from sensor_msgs.msg import PointCloud2, PointField

_DATATYPES = {}
_DATATYPES[PointField.INT8]    = ('b', 1)
_DATATYPES[PointField.UINT8]   = ('B', 1)
_DATATYPES[PointField.INT16]   = ('h', 2)
_DATATYPES[PointField.UINT16]  = ('H', 2)
_DATATYPES[PointField.INT32]   = ('i', 4)
_DATATYPES[PointField.UINT32]  = ('I', 4)
_DATATYPES[PointField.FLOAT32] = ('f', 4)
_DATATYPES[PointField.FLOAT64] = ('d', 8)

def read_points(cloud, field_names=None, skip_nans=False, uvs=[]):
    """
    Read points from a L{sensor_msgs.PointCloud2} message.

    @param cloud: The point cloud to read from.
    @type  cloud: L{sensor_msgs.PointCloud2}
    @param field_names: The names of fields to read. If None, read all fields. [default: None]
    @type  field_names: iterable
    @param skip_nans: If True, then don't return any point with a NaN value.
    @type  skip_nans: bool [default: False]
    @param uvs: If specified, then only return the points at the given coordinates. [default: empty list]
    @type  uvs: iterable
    @return: Generator which yields a list of values for each point.
    @rtype:  generator
    """
    assert isinstance(cloud, PointCloud2), 'cloud is not a sensor_msgs.msg.PointCloud2'
    fmt = _get_struct_fmt(cloud.is_bigendian, cloud.fields, field_names)
    width, height, point_step, row_step, data, isnan = cloud.width, cloud.height, cloud.point_step, cloud.row_step, cloud.data, math.isnan
    unpack_from = struct.Struct(fmt).unpack_from

    if skip_nans:
        if uvs:
            for u, v in uvs:
                p = unpack_from(data, (row_step * v) + (point_step * u))
                has_nan = False
                for pv in p:
                    if isnan(pv):
                        has_nan = True
                        break
                if not has_nan:
                    yield p
        else:
            for v in range(height):
                offset = row_step * v
                for u in range(width):
                    p = unpack_from(data, offset)
                    has_nan = False
                    for pv in p:
                        if isnan(pv):
                            has_nan = True
                            break
                    if not has_nan:
                        yield p
                    offset += point_step
    else:
        if uvs:
            for u, v in uvs:
                yield unpack_from(data, (row_step * v) + (point_step * u))
        else:
            for v in range(height):
                offset = row_step * v
                for u in range(width):
                    yield unpack_from(data, offset)
                    offset += point_step

def _get_struct_fmt(is_bigendian, fields, field_names=None):
    fmt = '>' if is_bigendian else '<'

    offset = 0
    for field in (f for f in sorted(fields, key=lambda f: f.offset) if field_names is None or f.name in field_names):
        if offset < field.offset:
            fmt += 'x' * (field.offset - offset)
            offset = field.offset
        if field.datatype not in _DATATYPES:
            print('Skipping unknown PointField datatype [%d]' % field.datatype, file=sys.stderr)
        else:
            datatype_fmt, datatype_length = _DATATYPES[field.datatype]
            fmt    += field.count * datatype_fmt
            offset += field.count * datatype_length

    return fmt



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
