# загружаем обученную сверточную модель
# загружаем экг из датасета
# бьем экг на перекрывающиеся сегменты
# создаем из них батч
# предсказываем сетью на том батче разметку
# отрисовавыем ее очень прозрачно на всей экг (там где очень ярко, модель очень уверена, типа)
# под штриховой линией, как обучно, отрисовываем истинную разметку

def get_model():
    model_name = 'oxxy_quad_annotator_long.h5'

def get_ecg():
    dset_name = 'DSET_argentina.pkl'

def get_batch_ecg(ecg, overide, seg_len=512):
    pass

def draw(full_ecg, full_ann, anns, times):
    name = "DRAW_FULL_ANN.png"

def run():
    model = get_model()
    full_ecg, full_ann = get_ecg()
    times, ecgs = get_batch_ecg(full_ecg, overide=105, seg_len=512)
    anns = model.predict_ob_batch(ecgs)
    draw(full_ecg, full_ann, anns, times)


