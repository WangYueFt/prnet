#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author: Yue Wang
@Contact: yuewangx@mit.edu
@File: data.py
@Time: 2/12/19 6:21 PM
"""

import os
import sys
import glob
import h5py
import numpy as np
from open3d import read_point_cloud
from scipy.spatial.transform import Rotation
from torch.utils.data import Dataset
from sklearn.neighbors import NearestNeighbors
from scipy.spatial.distance import minkowski


def download():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE_DIR, 'data')
    if not os.path.exists(DATA_DIR):
        os.mkdir(DATA_DIR)
    if not os.path.exists(os.path.join(DATA_DIR, 'modelnet40_ply_hdf5_2048')):
        www = 'https://shapenet.cs.stanford.edu/media/modelnet40_ply_hdf5_2048.zip'
        zipfile = os.path.basename(www)
        os.system('wget %s; unzip %s' % (www, zipfile))
        os.system('mv %s %s' % (zipfile[:-4], DATA_DIR))
        os.system('rm %s' % (zipfile))


def load_data(partition):
    download()
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE_DIR, 'data')
    all_data = []
    all_label = []
    for h5_name in glob.glob(os.path.join(DATA_DIR, 'modelnet40_ply_hdf5_2048', 'ply_data_%s*.h5' % partition)):
        f = h5py.File(h5_name)
        data = f['data'][:].astype('float32')
        label = f['label'][:].astype('int64')
        f.close()
        all_data.append(data)
        all_label.append(label)
    all_data = np.concatenate(all_data, axis=0)
    all_label = np.concatenate(all_label, axis=0)
    return all_data, all_label


def translate_pointcloud(pointcloud):
    xyz1 = np.random.uniform(low=2. / 3., high=3. / 2., size=[3])
    xyz2 = np.random.uniform(low=-0.2, high=0.2, size=[3])

    translated_pointcloud = np.add(np.multiply(pointcloud, xyz1), xyz2).astype('float32')
    return translated_pointcloud


def jitter_pointcloud(pointcloud, sigma=0.01, clip=0.05):
    N, C = pointcloud.shape
    pointcloud += np.clip(sigma * np.random.randn(N, C), -1 * clip, clip)
    return pointcloud


def farthest_subsample_one_points(pointcloud, num_subsampled_points=768, num_views=8):
    pointclouds = []
    nbrs = NearestNeighbors(n_neighbors=num_subsampled_points, algorithm='auto',
                            metric=lambda x, y: minkowski(x, y)).fit(pointcloud)
    for i in range(num_views):
        view = np.array([[i//4, i%4//2, i%2]])
        view[view==0] = -1
        view = view * 500
        idx = nbrs.kneighbors(view, return_distance=False).reshape((num_subsampled_points,))
        pointclouds.append(pointcloud[idx, :])
    pointclouds = np.array(pointclouds)
    return pointclouds.astype('float32')


def farthest_subsample_points(pointcloud1, pointcloud2, num_subsampled_points=768):
    pointcloud1 = pointcloud1.T
    pointcloud2 = pointcloud2.T
    num_points = pointcloud1.shape[0]
    nbrs1 = NearestNeighbors(n_neighbors=num_subsampled_points, algorithm='auto',
                             metric=lambda x, y: minkowski(x, y)).fit(pointcloud1)
 #   random_p1 = pointcloud1[np.random.randint(0, num_points), :].reshape((1, -1))
    random_p1 = np.random.random(size=(1, 3)) + np.array([[500, 500, 500]]) * np.random.choice([1, -1, 1, -1])
    idx1 = nbrs1.kneighbors(random_p1, return_distance=False).reshape((num_subsampled_points,))
    nbrs2 = NearestNeighbors(n_neighbors=num_subsampled_points, algorithm='auto',
                             metric=lambda x, y: minkowski(x, y)).fit(pointcloud2)
    random_p2 = random_p1 #np.random.random(size=(1, 3)) + np.array([[500, 500, 500]]) * np.random.choice([1, -1, 2, -2])
 #   random_p2 = pointcloud2[np.random.randint(0, num_points), :].reshape((1, -1))
    idx2 = nbrs2.kneighbors(random_p2, return_distance=False).reshape((num_subsampled_points,))
    return pointcloud1[idx1, :].T, pointcloud2[idx2, :].T


class ModelNet40(Dataset):
    def __init__(self, num_points, num_subsampled_points=768, partition='train',
                 gaussian_noise=False, unseen=False, rot_factor=4, category=None):
        super(ModelNet40, self).__init__()
        self.data, self.label = load_data(partition)
        if category is not None:
            self.data = self.data[self.label==category]
            self.label = self.label[self.label==category]
        self.num_points = num_points
        self.num_subsampled_points = num_subsampled_points
        self.partition = partition
        self.gaussian_noise = gaussian_noise
        self.unseen = unseen
        self.label = self.label.squeeze()
        self.rot_factor = rot_factor
        if num_points != num_subsampled_points:
            self.subsampled = True
        else:
            self.subsampled = False
        if self.unseen:
            ######## simulate testing on first 20 categories while training on last 20 categories
            if self.partition == 'test':
                self.data = self.data[self.label>=20]
                self.label = self.label[self.label>=20]
            elif self.partition == 'train':
                self.data = self.data[self.label<20]
                self.label = self.label[self.label<20]

    def __getitem__(self, item):
        pointcloud = self.data[item][:self.num_points]
        # if self.gaussian_noise:
        #    pointcloud = jitter_pointcloud(pointcloud)
        if self.partition != 'train':
            np.random.seed(item)
        anglex = np.random.uniform() * np.pi / self.rot_factor
        angley = np.random.uniform() * np.pi / self.rot_factor
        anglez = np.random.uniform() * np.pi / self.rot_factor
        cosx = np.cos(anglex)
        cosy = np.cos(angley)
        cosz = np.cos(anglez)
        sinx = np.sin(anglex)
        siny = np.sin(angley)
        sinz = np.sin(anglez)
        Rx = np.array([[1, 0, 0],
                        [0, cosx, -sinx],
                        [0, sinx, cosx]])
        Ry = np.array([[cosy, 0, siny],
                        [0, 1, 0],
                        [-siny, 0, cosy]])
        Rz = np.array([[cosz, -sinz, 0],
                        [sinz, cosz, 0],
                        [0, 0, 1]])
        R_ab = Rx.dot(Ry).dot(Rz)
        R_ba = R_ab.T
        translation_ab = np.array([np.random.uniform(-0.5, 0.5), np.random.uniform(-0.5, 0.5),
                                   np.random.uniform(-0.5, 0.5)])
        translation_ba = -R_ba.T.dot(translation_ab)

        pointcloud1 = pointcloud.T

        rotation_ab = Rotation.from_euler('zyx', [anglez, angley, anglex])
        pointcloud2 = rotation_ab.apply(pointcloud1.T).T + np.expand_dims(translation_ab, axis=1)

        euler_ab = np.asarray([anglez, angley, anglex])
        euler_ba = -euler_ab[::-1]

        pointcloud1 = np.random.permutation(pointcloud1.T).T
        pointcloud2 = np.random.permutation(pointcloud2.T).T

        if self.gaussian_noise:
            pointcloud1 = jitter_pointcloud(pointcloud1)
            pointcloud2 = jitter_pointcloud(pointcloud2)

        if self.subsampled:
            pointcloud1, pointcloud2 = farthest_subsample_points(pointcloud1, pointcloud2,
                                                                 num_subsampled_points=self.num_subsampled_points)

        return pointcloud1.astype('float32'), pointcloud2.astype('float32'), R_ab.astype('float32'), \
               translation_ab.astype('float32'), R_ba.astype('float32'), translation_ba.astype('float32'), \
               euler_ab.astype('float32'), euler_ba.astype('float32')

    def __len__(self):
        return self.data.shape[0]


def calc_distances(p0, points):
    return ((p0 - points)**2).sum(axis=1)


def graipher(pts, K):
    farthest_pts = np.zeros((K, 3))
    farthest_pts[0] = pts[np.random.randint(len(pts))]
    distances = calc_distances(farthest_pts[0], pts)
    for i in range(1, K):
        farthest_pts[i] = pts[np.argmax(distances)]
        distances = np.minimum(distances, calc_distances(farthest_pts[i], pts))
    return farthest_pts


def load_bunny():
    point_clouds = []
    for f in glob.glob('data/bunny/*.ply'):
        points = np.array(read_point_cloud(f).points)
        points = graipher(points, 2048) 
        points = (points-points.mean())/np.abs(points).max()       
        point_clouds.append(points)
    return np.array(point_clouds).astype('float32'), None


class Bunny(Dataset):
    def __init__(self, num_points, num_subsampled_points=768, partition='train',
                 gaussian_noise=False, unseen=False, rot_factor=4, category=None):
        super(Bunny, self).__init__()
        self.data, self.label = load_bunny()
        if category is not None:
            self.data = self.data[self.label==category]
            self.label = self.label[self.label==category]
        self.num_points = num_points
        self.num_subsampled_points = num_subsampled_points
        self.partition = partition
        self.gaussian_noise = gaussian_noise
        self.unseen = unseen
        self.rot_factor = rot_factor
        if num_points != num_subsampled_points:
            self.subsampled = True
        else:
            self.subsampled = False

    def __getitem__(self, item):
        pointcloud = self.data[item][:self.num_points]
        # if self.gaussian_noise:
        #    pointcloud = jitter_pointcloud(pointcloud)
        if self.partition != 'train':
            np.random.seed(item)
        anglex = np.random.uniform() * np.pi / self.rot_factor
        angley = np.random.uniform() * np.pi / self.rot_factor
        anglez = np.random.uniform() * np.pi / self.rot_factor
        cosx = np.cos(anglex)
        cosy = np.cos(angley)
        cosz = np.cos(anglez)
        sinx = np.sin(anglex)
        siny = np.sin(angley)
        sinz = np.sin(anglez)
        Rx = np.array([[1, 0, 0],
                        [0, cosx, -sinx],
                        [0, sinx, cosx]])
        Ry = np.array([[cosy, 0, siny],
                        [0, 1, 0],
                        [-siny, 0, cosy]])
        Rz = np.array([[cosz, -sinz, 0],
                        [sinz, cosz, 0],
                        [0, 0, 1]])
        R_ab = Rx.dot(Ry).dot(Rz)
        R_ba = R_ab.T
        translation_ab = np.array([np.random.uniform(-0.5, 0.5), np.random.uniform(-0.5, 0.5),
                                   np.random.uniform(-0.5, 0.5)])
        translation_ba = -R_ba.T.dot(translation_ab)

        pointcloud1 = pointcloud.T

        rotation_ab = Rotation.from_euler('zyx', [anglez, angley, anglex])
        pointcloud2 = rotation_ab.apply(pointcloud1.T).T + np.expand_dims(translation_ab, axis=1)

        euler_ab = np.asarray([anglez, angley, anglex])
        euler_ba = -euler_ab[::-1]

        pointcloud1 = np.random.permutation(pointcloud1.T).T
        pointcloud2 = np.random.permutation(pointcloud2.T).T

        if self.gaussian_noise:
            pointcloud1 = jitter_pointcloud(pointcloud1)
            pointcloud2 = jitter_pointcloud(pointcloud2)

        if self.subsampled:
            pointcloud1, pointcloud2 = farthest_subsample_points(pointcloud1, pointcloud2,
                                                                 num_subsampled_points=self.num_subsampled_points)

        return pointcloud1.astype('float32'), pointcloud2.astype('float32'), R_ab.astype('float32'), \
               translation_ab.astype('float32'), R_ba.astype('float32'), translation_ba.astype('float32'), \
               euler_ab.astype('float32'), euler_ba.astype('float32')

    def __len__(self):
        return self.data.shape[0]


class ShapeNet(Dataset):
    def __init__(self, num_points, num_subsampled_points=768, partition='train',
                 gaussian_noise=False, unseen=False, rot_factor=4):
        super(ShapeNet, self).__init__()
        if not os.path.exists('data/shapenet57448.h5'):
            url = 'https://ucb4a2e1dae1e8855ad2641e152f.dl.dropboxusercontent.com/cd/0/get/AhUCISdroTPnMFhMAVM7WCbgd3zgGyHonOc1tGgtL7pzVxbx35yPDzsdvLOZRFZnszKluahwFcfrv3HsPbu9ahgMwXSlqSgMmw8dSlWiPhB6Yg/file?dl=1#'
            os.system('wget %s' % url)
            os.system('mv file?dl=1 data/shapenet57448.h5')
        with h5py.File('data/shapenet57448.h5') as f:
            self.data = f['data'][:]
        self.num_points = num_points
        self.num_subsampled_points = num_subsampled_points
        self.partition = partition
        self.gaussian_noise = gaussian_noise
        self.unseen = unseen
        self.rot_factor = rot_factor
        if num_points != num_subsampled_points:
            self.subsampled = True
        else:
            self.subsampled = False

    def __getitem__(self, item):
        pointcloud = self.data[item][:self.num_points]
        if self.partition != 'train':
            np.random.seed(item)
        anglex = np.random.uniform() * np.pi / self.rot_factor
        angley = np.random.uniform() * np.pi / self.rot_factor
        anglez = np.random.uniform() * np.pi / self.rot_factor
        cosx = np.cos(anglex)
        cosy = np.cos(angley)
        cosz = np.cos(anglez)
        sinx = np.sin(anglex)
        siny = np.sin(angley)
        sinz = np.sin(anglez)
        Rx = np.array([[1, 0, 0],
                        [0, cosx, -sinx],
                        [0, sinx, cosx]])
        Ry = np.array([[cosy, 0, siny],
                        [0, 1, 0],
                        [-siny, 0, cosy]])
        Rz = np.array([[cosz, -sinz, 0],
                        [sinz, cosz, 0],
                        [0, 0, 1]])
        R_ab = Rx.dot(Ry).dot(Rz)
        R_ba = R_ab.T
        translation_ab = np.array([np.random.uniform(-0.5, 0.5), np.random.uniform(-0.5, 0.5),
                                   np.random.uniform(-0.5, 0.5)])
        translation_ba = -R_ba.T.dot(translation_ab)

        pointcloud1 = pointcloud.T

        rotation_ab = Rotation.from_euler('zyx', [anglez, angley, anglex])
        pointcloud2 = rotation_ab.apply(pointcloud1.T).T + np.expand_dims(translation_ab, axis=1)

        euler_ab = np.asarray([anglez, angley, anglex])
        euler_ba = -euler_ab[::-1]

        pointcloud1 = np.random.permutation(pointcloud1.T).T
        pointcloud2 = np.random.permutation(pointcloud2.T).T

        if self.gaussian_noise:
            pointcloud1 = jitter_pointcloud(pointcloud1)
            pointcloud2 = jitter_pointcloud(pointcloud2)

        if self.subsampled:
            pointcloud1, pointcloud2 = farthest_subsample_points(pointcloud1, pointcloud2,
                                                                 num_subsampled_points=self.num_subsampled_points)

        return pointcloud1.astype('float32'), pointcloud2.astype('float32'), R_ab.astype('float32'), \
               translation_ab.astype('float32'), R_ba.astype('float32'), translation_ba.astype('float32'), \
               euler_ab.astype('float32'), euler_ba.astype('float32')

    def __len__(self):
        return self.data.shape[0]


class ModelNet40CLS(Dataset):
    def __init__(self, num_points, num_subsampled_points=768, partition='train', so3=False, num_views=8):
        super(ModelNet40CLS, self).__init__()
        self.data, self.label = load_data(partition)
        self.num_points = num_points
        self.num_subsampled_points = num_subsampled_points
        self.partition = partition
        self.so3 = so3
        self.num_views = num_views

    def __getitem__(self, item):
        pointcloud = self.data[item][:self.num_points]
        label = self.label[item]
        # if self.partition == 'train':
        #    pointcloud = translate_pointcloud(pointcloud)
        if self.partition != 'train':
            np.random.seed(item)
        anglex = np.random.uniform() * np.pi / 0.5 # self.rot_factor
        angley = np.random.uniform() * np.pi / 0.5 # self.rot_factor
        anglez = np.random.uniform() * np.pi / 0.5 # self.rot_factor
        cosx = np.cos(anglex)
        cosy = np.cos(angley)
        cosz = np.cos(anglez)
        sinx = np.sin(anglex)
        siny = np.sin(angley)
        sinz = np.sin(anglez)
        Rx = np.array([[1, 0, 0],
                       [0, cosx, -sinx],
                       [0, sinx, cosx]])
        Ry = np.array([[cosy, 0, siny],
                       [0, 1, 0],
                       [-siny, 0, cosy]])
        Rz = np.array([[cosz, -sinz, 0],
                       [sinz, cosz, 0],
                       [0, 0, 1]])
        R_ab = Rx.dot(Ry).dot(Rz)
        translation_ab = np.array([np.random.uniform(-0.0, 0.0), np.random.uniform(-0.0, 0.0),
                                   np.random.uniform(-0.0, 0.0)])

        if self.so3:
            rotation_ab = Rotation.from_euler('zyx', [anglez, angley, anglex])
            pointcloud = rotation_ab.apply(pointcloud) + np.expand_dims(translation_ab, axis=1).T
       
        pointclouds = farthest_subsample_one_points(pointcloud,
                                                    self.num_subsampled_points,
                                                    num_views=self.num_views)
        return pointclouds, label

    def __len__(self):
        return self.data.shape[0]


if __name__ == '__main__':
    train = ModelNet40(1024)
    test = ModelNet40(1024, 'test')
    for data in train:
        print(len(data))
        break
