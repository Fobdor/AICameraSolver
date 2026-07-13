import os
import numpy as np
from PySide6.QtCore import QThread, Signal

class SolveRefinementWorker(QThread):
    progress = Signal(int, str)
    finished = Signal(bool, str)

    def __init__(self, project_dir, constraints, camera_data):
        super().__init__()
        self.project_dir = project_dir
        self.constraints = constraints
        self.camera_data = camera_data

    def run(self):
        try:
            import torch
            from scipy.spatial.transform import Rotation
            
            raw_data_path = os.path.join(self.project_dir, 'solve_data_raw.npz')
            tracks_path = os.path.join(self.project_dir, 'tracks.npz')
            
            if not os.path.exists(raw_data_path) or not os.path.exists(tracks_path):
                self.finished.emit(False, "Missing raw solve data or tracks data.")
                return
                
            self.progress.emit(5, "Loading PyTorch tensors...")
            data = dict(np.load(raw_data_path))
            tracks_2d = data['tracks_2d'] # (S, N, 2)
            visibility = data['visibility'] # (S, N)
            
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            
            pts_3d = torch.tensor(data['points_3d'], dtype=torch.float32, device=device, requires_grad=True)
            
            rot_mats = data['cameras_rot'] # (S, 3, 3)
            translations = data['cameras_trans'] # (S, 3)
            
            S = rot_mats.shape[0]
            rot_vecs = np.zeros((S, 3), dtype=np.float32)
            for i in range(S):
                r = Rotation.from_matrix(rot_mats[i])
                rot_vecs[i] = r.as_rotvec()
                    
            r_vecs_t = torch.tensor(rot_vecs, dtype=torch.float32, device=device, requires_grad=True)
            t_vecs_t = torch.tensor(translations, dtype=torch.float32, device=device, requires_grad=True)
            
            plate_w = self.camera_data.get('plate_width', 1920)
            plate_h = self.camera_data.get('plate_height', 1080)
            cx, cy = plate_w / 2.0, plate_h / 2.0
            f = float(data['focal_px'])
            
            intrinsics = np.zeros((S, 3, 3), dtype=np.float32)
            for i in range(S):
                intrinsics[i] = np.array([[f, 0, cx],
                                          [0, f, cy],
                                          [0, 0,  1]], dtype=np.float32)
            
            optimizer = torch.optim.Adam([pts_3d, r_vecs_t, t_vecs_t], lr=0.01)
            
            def axis_angle_to_matrix(rot_vecs):
                theta = torch.norm(rot_vecs, dim=-1, keepdim=True)
                omega = rot_vecs / (theta + 1e-8)
                x, y, z = omega[..., 0], omega[..., 1], omega[..., 2]
                K = torch.zeros(*rot_vecs.shape[:-1], 3, 3, device=rot_vecs.device)
                K[..., 0, 1] = -z
                K[..., 0, 2] = y
                K[..., 1, 0] = z
                K[..., 1, 2] = -x
                K[..., 2, 0] = -y
                K[..., 2, 1] = x
                I = torch.eye(3, device=rot_vecs.device).expand_as(K)
                sin_theta = torch.sin(theta).unsqueeze(-1)
                cos_theta = torch.cos(theta).unsqueeze(-1)
                R = I + sin_theta * K + (1 - cos_theta) * torch.bmm(K.view(-1, 3, 3), K.view(-1, 3, 3)).view(K.shape)
                return R

            self.progress.emit(10, "Optimizing...")
            
            vis_mask = torch.tensor(visibility, dtype=torch.bool, device=device)
            tracks_t = torch.tensor(tracks_2d, dtype=torch.float32, device=device)
            K_mats = torch.tensor(intrinsics, dtype=torch.float32, device=device)
            
            coplanarity_weight = 1000.0
            
            plane_constraints = []
            for c in self.constraints:
                ctype = c['type']
                if "Plane" in ctype or "Coplanar Group" in ctype:
                    indices = [i for i in c['tracks'] if i < len(pts_3d)]
                    if len(indices) >= 3:
                        plane_constraints.append(torch.tensor(indices, dtype=torch.long, device=device))
            
            num_iters = 250
            for step in range(num_iters):
                optimizer.zero_grad()
                
                R = axis_angle_to_matrix(r_vecs_t) # (S, 3, 3)
                
                X_expanded = pts_3d.unsqueeze(0).expand(S, -1, -1) # (S, N, 3)
                P_cam = torch.bmm(X_expanded, R.transpose(1, 2)) + t_vecs_t.unsqueeze(1) # (S, N, 3)
                P_px = torch.bmm(P_cam, K_mats.transpose(1, 2)) # (S, N, 3)
                
                Z = P_px[..., 2]
                uv_proj = P_px[..., :2] / (Z.unsqueeze(-1) + 1e-8) # (S, N, 2)
                
                reproj_diff = uv_proj[vis_mask] - tracks_t[vis_mask]
                loss_reproj = torch.nn.functional.smooth_l1_loss(reproj_diff, torch.zeros_like(reproj_diff), beta=1.0)
                
                loss_coplanar = torch.tensor(0.0, device=device)
                for idxs in plane_constraints:
                    group_pts = pts_3d[idxs] # (K, 3)
                    centroid = group_pts.mean(dim=0, keepdim=True)
                    centered = group_pts - centroid
                    cov = torch.matmul(centered.T, centered) / (len(group_pts) - 1 + 1e-8)
                    L, V = torch.linalg.eigh(cov)
                    variance = L[0]
                    loss_coplanar = loss_coplanar + variance
                    
                total_loss = loss_reproj + coplanarity_weight * loss_coplanar
                total_loss.backward()
                optimizer.step()
                
                if step % 25 == 0:
                    self.progress.emit(10 + int(80 * step / num_iters), f"Optimizing step {step}/{num_iters}...")
                    
            self.progress.emit(95, "Saving refined data...")
            
            data['points_3d'] = pts_3d.detach().cpu().numpy()
            
            final_r_vecs = r_vecs_t.detach().cpu().numpy()
            final_R = np.zeros((S, 3, 3), dtype=np.float32)
            for i in range(S):
                final_R[i] = Rotation.from_rotvec(final_r_vecs[i]).as_matrix()
            data['cameras_rot'] = final_R
            data['cameras_trans'] = t_vecs_t.detach().cpu().numpy()
            
            np.savez(raw_data_path, **data)
            
            self.finished.emit(True, "Refinement complete!")
            
        except Exception as e:
            self.finished.emit(False, str(e))
