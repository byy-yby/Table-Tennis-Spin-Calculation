"""多图处理: 张正友棋盘格标定 + 亚像素角点提取 + 重投影误差评估"""
import cv2, glob, numpy as np

DEFAULT_CHECKERBOARD = (10, 7)
DEFAULT_SQUARE_SIZE = 28.9


class CameraCalibrator:
    """张正友相机内外参标定器"""
    def __init__(self, checkerboard=DEFAULT_CHECKERBOARD, square_size=DEFAULT_SQUARE_SIZE):
        self.checkerboard = checkerboard
        self.square_size = square_size
        
        # 亚像素角点提取终止条件: 最大迭代30次 或 移动距离<0.001像素
        self.criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

        # 预设世界坐标系下的 3D 物理网格点: (0,0,0), (1,0,0) ...
        cb_w, cb_h = self.checkerboard
        self.objp = np.zeros((cb_w * cb_h, 3), np.float32)
        self.objp[:, :2] = np.mgrid[0:cb_w, 0:cb_h].T.reshape(-1, 2)
        self.objp *= self.square_size

    def calibrate(self, image_path='images/*.jpg', show_process=False):
        """处理标定图集: 提取角点 → 相机标定。返回 (mtx, dist) 或 (None, None)"""
        images = glob.glob(image_path)
        if not images:
            print(f"[Calibration] 错误: 未在 {image_path} 找到图片")
            return None, None

        objpoints = []
        imgpoints = []

        for fname in images:
            img = cv2.imread(fname)
            if img is None: continue
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            ret, corners = cv2.findChessboardCorners(gray, self.checkerboard, None)
            
            if ret:
                objpoints.append(self.objp)
                corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), self.criteria)
                imgpoints.append(corners2)

                if show_process:
                    cv2.drawChessboardCorners(img, self.checkerboard, corners2, ret)
                    cv2.imshow('Calibration - Press any key to skip', img)
                    cv2.waitKey(100)
            else:
                print(f"[Calibration] 警告: {fname} 未找到完整角点，已跳过")

        if show_process:
            cv2.destroyAllWindows()

        if not objpoints:
            return None, None

        img_shape = cv2.imread(images[0]).shape[:2][::-1]
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
            objpoints, imgpoints, img_shape, None, None
        )

        total_error = self._evaluate_error(objpoints, imgpoints, rvecs, tvecs, mtx, dist)

        if not hasattr(self, '_logged'):
            self._logged = True
            print(f"[Calibration] 提取成功图片数: {len(objpoints)}/{len(images)}")
            print(f"[Calibration] RMS={ret:.4f} px, Mean_Reproj_Error={total_error:.4f} px")
            print(f"[Calibration] Camera Matrix:\n{mtx}")
            print(f"[Calibration] Distortion:\n{dist}")

        return mtx, dist

    def _evaluate_error(self, objpoints, imgpoints, rvecs, tvecs, mtx, dist):
        """计算实际 2D 点与 3D 点重投影后的平均像素误差"""
        mean_error = 0
        for i in range(len(objpoints)):
            imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
            error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
            mean_error += error
        return mean_error / len(objpoints)


if __name__ == "__main__":
    calibrator = CameraCalibrator()
    calibrator.calibrate('images/*.jpg', show_process=True)
