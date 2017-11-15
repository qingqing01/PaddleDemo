import paddle.v2 as paddle
from paddle.utils.merge_model import merge_v2_model
import ocr_det

if __name__ == "__main__":
    class_prob, bbox_pred = ocr_det.network()
    param_file = "ocr_frcnn.tar.gz"
    output_file = "ocr_frcnn.paddle"
    merge_v2_model(class_prob, param_file, output_file)
