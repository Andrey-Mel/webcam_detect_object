# запускать из под wsl, из виндовс не запускается. На radxa не срабатывает. Рабочий код convert_ultralytics.py.
# Нужно понять почему не получается, в ultralytics в кодах практически тоже самое
from rknn.api import RKNN

ONNX_MODEL = 'best_sdu_v3_clamped.onnx'  #'yolo9out/yolo11m_9out.onnx'#'best_sdu_v3.onnx'  # 'best_sdu_v3.onnx'
# PT_MODEL = 'yolo11m.pt'
RKNN_MODEL = 'best_sdu_v3_clamped.rknn' #'yolo11m_9_out.rknn'#best_sdu_v3.rknn' #'best_sdu_v3.rknn' #'best_sdu_v3.rknn'
# EXPORT_MODEL = 'best_sdu_v3-rk3588_export.rknn'
DATASET = 'dataset.txt'
QUANTIZE_ON = True

if __name__ == '__main__':
    # create rknn object
    rknn = RKNN()

   
    # preprocess config
    print('--> Config model')
    rknn.config(mean_values=[[0, 0, 0]], 
                std_values=[[255, 255, 255]], 
                quant_img_RGB2BGR = True,
                target_platform='rk3588',
                quantized_algorithm='normal',
                quantized_method='channel',
                optimization_level = 3,
                quantized_dtype = 'asymmetric_quantized-8' #'w8a8'
                )
    print('done')

   

    # Load ONNX model
    print('--> Loading model onnx')
    ret = rknn.load_onnx(model=ONNX_MODEL,
                         input_size_list=[[1, 3, 640, 640]])
    if ret != 0:
        print("load model failed!")
        exit(ret)
    

    #Build model
    print('--> Building model')
    ret = rknn.build(do_quantization=QUANTIZE_ON, dataset=DATASET) #
    if ret != 0:
        print('Build model failed')
        exit(ret)
    print('Done')

    #Export RKNN model
    print('-->Export model rknn')
    ret = rknn.export_rknn(RKNN_MODEL)  # RKNN_MODEL
    if ret != 0:
        print('Export rknn model failed')
        exit(ret)
    print('Done')

rknn.release()