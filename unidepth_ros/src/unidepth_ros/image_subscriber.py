from threading import Lock

import rospy
from sensor_msgs.msg import Image


class LatestImageSubscriber:
    """Keep only the latest camera frame so inference never queues stale data."""

    def __init__(self, image_topic: str):
        self._lock = Lock()
        self._latest = None
        self._subscriber = rospy.Subscriber(
            image_topic,
            Image,
            self._callback,
            queue_size=1,
            buff_size=2 ** 24,
        )

    def _callback(self, message: Image) -> None:
        with self._lock:
            self._latest = message

    def take_latest(self):
        with self._lock:
            message = self._latest
            self._latest = None
        return message
