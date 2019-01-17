# -*- coding: utf-8 -*-
###############################################
#created by :  lxy
#Time:  2018/07/5 10:09
#project: Face detect
#company: Senscape
#rversion: 0.1
#tool:   python 3
#modified:
#description  crop face align
####################################################
import numpy as np
import cv2
import math
from skimage import transform as trans

class Align_img(object):
    def __init__(self,desired_size,padding=0):
        self.h,self.w = desired_size
        self.padding = padding

    def list2colmatrix(self, pts_list):
        """
            convert list to column matrix
        Parameters:
        ----------
            pts_list:
                input list
        Retures:
        -------
            colMat:
        """
        assert len(pts_list) > 0
        colMat = []
        for i in range(len(pts_list)):
            colMat.append(pts_list[i][0])
            colMat.append(pts_list[i][1])
        #print("the colMat shape before mat ",np.shape(colMat))
        colMat = np.matrix(colMat).transpose()
        #print("the colMat shape after mat ",np.shape(colMat))
        return colMat

    def find_tfrom_between_shapes(self, from_shape, to_shape):
        """
            find transform between shapes
        Parameters:
        ----------
            from_shape:
            to_shape:
        Retures:
        -------
            tran_m:
            tran_b:
        """
        assert from_shape.shape[0] == to_shape.shape[0] and from_shape.shape[0] % 2 == 0

        sigma_from = 0.0
        sigma_to = 0.0
        cov = np.matrix([[0.0, 0.0], [0.0, 0.0]])

        # compute the mean and cov
        from_shape_points = from_shape.reshape(from_shape.shape[0]/2, 2)
        to_shape_points = to_shape.reshape(to_shape.shape[0]/2, 2)
        mean_from = from_shape_points.mean(axis=0)
        mean_to = to_shape_points.mean(axis=0)

        for i in range(from_shape_points.shape[0]):
            temp_dis = np.linalg.norm(from_shape_points[i] - mean_from)
            sigma_from += temp_dis * temp_dis
            temp_dis = np.linalg.norm(to_shape_points[i] - mean_to)
            sigma_to += temp_dis * temp_dis
            cov += (to_shape_points[i].transpose() - mean_to.transpose()) * (from_shape_points[i] - mean_from)

        sigma_from = sigma_from / to_shape_points.shape[0]
        sigma_to = sigma_to / to_shape_points.shape[0]
        cov = cov / to_shape_points.shape[0]

        # compute the affine matrix
        s = np.matrix([[1.0, 0.0], [0.0, 1.0]])
        u, d, vt = np.linalg.svd(cov)

        if np.linalg.det(cov) < 0:
            if d[1] < d[0]:
                s[1, 1] = -1
            else:
                s[0, 0] = -1
        r = u * s * vt
        c = 1.0
        if sigma_from != 0:
            c = 1.0 / sigma_from * np.trace(np.diag(d) * s)

        tran_b = mean_to.transpose() - c * r * mean_from.transpose()
        tran_m = c * r

        return tran_m, tran_b

    def extract_image_chips(self, img, points):
        """
            crop and align face
        Parameters:
        ----------
            img: numpy array, bgr order of shape (1, 3, n, m)
                input image
            points: numpy array, n x 10 (x1,y1, x2,y2... x5,y5)
            desired_size: default 256
            padding: default 0
        Retures:
        -------
            crop_imgs: list, n
                cropped and aligned faces
        """
        crop_imgs = []
        for p in points:
            shape = p
            '''
            for k in range(len(p)/2):
                shape.append(p[k])
                shape.append(p[k+5])
            '''
            if self.padding > 0:
                padding = self.padding
            else:
                padding = 0
            # average positions of face points
            mean_face_shape_x = [0.224152, 0.75610125, 0.490127, 0.254149, 0.726104]
            mean_face_shape_y = [0.2119465, 0.2119465, 0.628106, 0.780233, 0.780233]

            from_points = []
            to_points = []

            for i in range(len(shape)/2):
                x = (padding + mean_face_shape_x[i]) / (2 * padding + 1) * self.w
                y = (padding + mean_face_shape_y[i]) / (2 * padding + 1) * self.h
                to_points.append([x, y])
                from_points.append([shape[2*i], shape[2*i+1]])

            # convert the points to Mat
            #print("the points from and to shape ",np.shape(from_points),np.shape(to_points))
            from_mat = self.list2colmatrix(from_points)
            to_mat = self.list2colmatrix(to_points)
            #print("the points mat from and to shape ",np.shape(from_mat),np.shape(to_mat))
            # compute the similar transfrom
            tran_m, tran_b = self.find_tfrom_between_shapes(from_mat, to_mat)

            probe_vec = np.matrix([1.0, 0.0]).transpose()
            probe_vec = tran_m * probe_vec

            scale = np.linalg.norm(probe_vec)
            angle = 180.0 / math.pi * math.atan2(probe_vec[1, 0], probe_vec[0, 0])
            #print("the angle is ",angle)
            from_center = [(shape[0]+shape[2])/2.0, (shape[1]+shape[3])/2.0]
            to_center = [0, 0]
            to_center[1] = self.h * 0.4
            to_center[0] = self.w * 0.5

            ex = to_center[0] - from_center[0]
            ey = to_center[1] - from_center[1]

            rot_mat = cv2.getRotationMatrix2D((from_center[0], from_center[1]), -1*angle, scale)
            rot_mat[0][2] += ex
            rot_mat[1][2] += ey

            chips = cv2.warpAffine(img, rot_mat, (self.w, self.h))
            crop_imgs.append(chips)
        return crop_imgs

def alignImg(img,image_size,points):
    src = np.array([
      [30.2946, 51.6963],
      [65.5318, 51.5014],
      [48.0252, 71.7366],
      [33.5493, 92.3655],
      [62.7299, 92.2041] ], dtype=np.float32 )
    if image_size[1]==112:
        src[:,0] += 8.0
    tform = trans.SimilarityTransform()
    crop_imgs = []
    cropsize = (image_size[1],image_size[0])
    for p in points:
        dst = np.reshape(p,(2,5)).T
        tform.estimate(dst, src)
        M = tform.params[0:2,:]
        warped = cv2.warpAffine(img,M,(image_size[1],image_size[0]), borderMode=1)
        #tform = trans.estimate_transform('affine', dst, src) # Assume square
        #warped = cv2.warpPerspective(img, tform.params, cropsize, borderMode=1)
        if warped is not None:
            crop_imgs.append(warped)
    return crop_imgs