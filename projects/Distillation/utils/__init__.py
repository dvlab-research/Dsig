from .feature_kd_loss import mse_loss_withmask
from .feature_kd_loss import cosine_similarity
from .feature_kd_loss import similarity_mse_loss
from .feature_kd_loss import kl_similarity
from .feature_kd_loss import similarity_l1_loss
from .feature_kd_loss import kl_similarity_loss
from .feature_kd_loss import normalized_loss
from .feature_kd_loss import similarity_func_dict
from .feature_kd_loss import l2_similarity


from .matrix_utils import generate_correlation_matrix, corr_mat_mse_loss, split_features_per_image
from .matrix_utils import fuse_bg_features, cat_fg_bg_features, select_topk_features_as_fg

from .matrix_utils import sim_dis_compute