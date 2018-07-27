import numpy as np
from dataset_getter import prepare_data
import pprint
from utils import open_pickle, get_addon_mask
import matplotlib.pyplot as plt
import BaselineWanderRemoval as bwr

SHRINK_FACTOR = 2

def shrink_dataset(dataset_in):
    """
    выкидывает каждый пиксель с шагом SHRINK_FACTOR (из экг-шки и из маск)
    :param dataset_in: датасет для разрезания, представляет собой мапу с 2 ключами- 'х' и 'ann'
    :return:
    """
    dset_shrinked = {}
    x = np.array(dataset_in['x'])
    ann = np.array(dataset_in['ann'])
    print ("dataset x has shape: " + str(x.shape))
    dset_shrinked['x'] = x[:,:,::SHRINK_FACTOR]
    print("dataset x has NOW shape: " + str(dset_shrinked['x'].shape))
    dset_shrinked['ann'] = ann[:,:,::SHRINK_FACTOR]
    return dset_shrinked

def annotated_generator(segment_len, batch_size, dataset_in):

    """
    батч-генератор для ЭКГ с аннотациями
    :param segment_len: длина (в тактах) кусков экг, которые будем вырезать
    :param batch_size: длина батча, возвращаемого ф-цией
    :param ecg_dataset: датасет для разрезания, представляет собой мапу с 2 ключами- 'х' и 'ann'
    """

    # open_pickle находится в utils для компактности

    ecg_dataset = np.array(dataset_in['x'])
    ecg_annotations = np.array(dataset_in['ann'])

    # отступ от начала и конца
    offset = 700

    ecg_dataset = np.swapaxes(ecg_dataset, 1, 2)
    ecg_annotations = np.swapaxes(ecg_annotations, 1, 2)

    starting_position = np.random.randint(offset, ecg_dataset.shape[1] - segment_len - offset)
    ending_position = starting_position + segment_len
    ecg_rand = np.random.randint(0, ecg_dataset.shape[0])

    while True:
        batch_x = ecg_dataset[ecg_rand: ecg_rand +1, starting_position:ending_position, :]
        batch_ann = np.array([ecg_annotations[ecg_rand, starting_position:ending_position, :]])
        for i in range(0, batch_size- 1):
            starting_position = np.random.randint(offset, ecg_dataset.shape[1] - segment_len - offset)
            ending_position = starting_position + segment_len
            ecg_rand = np.random.randint(0, ecg_dataset.shape[0])
            batch_x = np.concatenate(
                (batch_x, ecg_dataset[ecg_rand:ecg_rand + 1, starting_position:ending_position, :]), 0)
            batch_ann = np.concatenate(
                (batch_ann, ecg_annotations[ecg_rand:ecg_rand + 1, starting_position:ending_position, :]), 0)
        print(batch_x.shape)
        print(batch_ann.shape)
        yield (batch_x, batch_ann)

def annotated_generator_with_addon(segment_len, batch_size, dataset_in):
    """
    батч-генератор для ЭКГ с аннотациями + дополнительная маска
    :param segment_len: длина (в тактах) кусков экг, которые будем вырезать
    :param batch_size: длина батча, возвращаемого ф-цией
    :param ecg_dataset: датасет для разрезания, представляет собой мапу с 2 ключами- 'х' и 'ann'
    """

    ecg_dataset = np.array(dataset_in['x'])
    ecg_annotations = np.array(dataset_in['ann'])

    # отступ от начала и конца
    offset = 700

    ecg_dataset = np.swapaxes(ecg_dataset, 1, 2)
    ecg_annotations = np.swapaxes(ecg_annotations, 1, 2)

    starting_position = np.random.randint(offset, ecg_dataset.shape[1] - segment_len - offset)
    ending_position = starting_position + segment_len
    ecg_rand = np.random.randint(0, ecg_dataset.shape[0])
    while True:
        batch_x = ecg_dataset[ecg_rand: ecg_rand +1, starting_position:ending_position, :]
        batch_ann = np.array([ecg_annotations[ecg_rand, starting_position:ending_position, :]])
        
        addition = get_addon_mask(ecg_annotations[ecg_rand:ecg_rand + 1, starting_position:ending_position, :])
        batch_ann = np.concatenate((batch_ann, addition), 2)

        for i in range(0, batch_size- 1):
            starting_position = np.random.randint(offset, ecg_dataset.shape[1] - segment_len - offset)
            ending_position = starting_position + segment_len
            ecg_rand = np.random.randint(0, ecg_dataset.shape[0])            
            batch_x = np.concatenate(
                (batch_x, ecg_dataset[ecg_rand:ecg_rand + 1, starting_position:ending_position, :]), 0)
            
            mask = get_addon_mask(ecg_annotations[ecg_rand:ecg_rand+1, starting_position:ending_position, :])
            addition = np.concatenate((ecg_annotations[ecg_rand:ecg_rand + 1, starting_position:ending_position, :], mask), 2)
            batch_ann = np.concatenate((batch_ann, addition), 0)

        batch_ann = np.swapaxes(batch_ann, 1, 2)
        print(batch_x.shape)
        print(batch_ann.shape)
        yield (batch_x, batch_ann)

def extract_first_lines(dataset_in):
    """
    из каждой записи входного датасета берет первое отведение и первую маску
    :param dataset_in: датасет, представляет собой мапу с 2 ключами- 'х' и 'ann'
    :return: датасет, представляет собой мапу с 2 ключами- 'х' и 'ann'
    """
    x = dataset_in['x'][:,0:1,:]
    ann = dataset_in['ann'][:,2:3,:]
    return {'x':x, 'ann':ann}


def get_enhansed_generator(segment_len, batch_size, dataset_in):
    """
    генератор данных для аннотатора, внутри него произведена вся необходимая предобраотка экг/аннотиаций
    :param segment_len:
    :param batch_size:
    :param dataset_in: датасет, представляет собой мапу с 2 ключами- 'х' и 'ann'
    :return:
    """
    dataset_only_one_channel = extract_first_lines(dataset_in)
    print("One channel shape: " + str(dataset_only_one_channel['x'].shape))

    for i in range(0, dataset_only_one_channel['x'].shape[0]):
        print("Now smoothing: " + str(i))
        dataset_only_one_channel['x'][i, 0, :] = bwr.fix_baseline_wander(dataset_only_one_channel['x'][i, 0, :], 500)

    dataset_shrinked = shrink_dataset(dataset_only_one_channel)
    my_generator = annotated_generator(segment_len=segment_len, batch_size=batch_size, dataset_in=dataset_shrinked)
    return my_generator

def extract_first_leads(dataset_in):
    """
    из каждой записи входного датасета берет первое отведение
    :param dataset_in: датасет, представляет собой мапу с 2 ключами- 'х' и 'ann'
    :return: датасет, представляет собой мапу с 2 ключами- 'х' и 'ann'
    """
    x = dataset_in['x'][:,0:1,:]
    return {'x':x, 'ann':dataset_in['ann']}

def get_mulimask_generator(segment_len, batch_size, dataset_in):
    """
    генератор данных для аннотатора, внутри него произведена вся необходимая предобраотка экг/аннотиаций
    :param segment_len:
    :param batch_size:
    :param dataset_in: датасет, представляет собой мапу с 2 ключами- 'х' и 'ann'
    :return:
    """
    dataset_only_one_channel = extract_first_leads(dataset_in)
    dataset_shrinked = shrink_dataset(dataset_only_one_channel)
    my_generator = annotated_generator(segment_len=segment_len, batch_size=batch_size, dataset_in=dataset_shrinked)
    return my_generator

def get_mulimask_generator_addon(segment_len, batch_size, dataset_in):
    """
    генератор данных для аннотатора, внутри него произведена вся необходимая предобраотка экг/аннотиаций + доп. маска
    :param segment_len:
    :param batch_size:
    :param dataset_in: датасет, представляет собой мапу с 2 ключами- 'х' и 'ann'
    :return:
    """
    dataset_only_one_channel = extract_first_leads(dataset_in)
    dataset_shrinked = shrink_dataset(dataset_only_one_channel)
    my_generator = annotated_generator_with_addon(segment_len=segment_len, batch_size=batch_size, dataset_in=dataset_shrinked)
    return my_generator

def TEST_all():
    dataset_in = open_pickle('../datasets/DSET_argentina.pkl')
    my_generator = get_enhansed_generator(segment_len=512, batch_size=20, dataset_in=dataset_in)
    batch = next(my_generator)
    print ("shape of batch x= " + str(batch[0].shape))
    print("shape of batch y= " + str(batch[1].shape))

def draw_shrinked():
    dataset_in = open_pickle('../datasets/DSET_argentina.pkl')
    dataset_only_one_channel = extract_first_lines(dataset_in)
    dset_shrinked = shrink_dataset(dataset_only_one_channel)
    before_x = dataset_only_one_channel['x'][0,0,0:30]
    after_x = dset_shrinked['x'][0,0,0:30]

    figname = "shrinked_ecg.png"
    f, (ax1, ax2) = plt.subplots(1, 2, sharey=True, sharex=False)
    ax1.plot(before_x, 'k-', label="несжатый")
    ax2.plot(after_x, 'm-', label="сжатый в "+ str(SHRINK_FACTOR))

    plt.legend(loc=2)
    plt.savefig(figname)


if __name__ == "__main__":
    TEST_all()
    draw_shrinked()

