import os
import cv2
import argparse

if __name__ == "__main__":
    args = argparse.ArgumentParser("object detection")
    args.add_argument("-c", "--config", default=None, type=str, help="config file path (default: None)")
    args.add_argument("-r", "--resume", default=None, type=str, help="path to latest checkpoint (default: None)")
