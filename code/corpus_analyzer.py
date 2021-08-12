import music21
import numpy as np
import matplotlib.pyplot as plt
import collections
import os


class SequenceMiner:
    @staticmethod
    def common_sequence(lst, len=2):
        pass


class Analyzer:
    @staticmethod
    def get_beat_list(xml_file):
        m21_file = music21.converter.parse(xml_file)
        flattened = m21_file.flat
        notes = flattened.getElementsByClass(music21.note.Note)
        total_beat = round(m21_file.duration.quarterLength)
        beat_division = 1
        beats = np.empty(shape=(int(total_beat * beat_division),4),dtype=str)
        beats.fill('')
        for note in notes:
            beat_of_note = int(note.offset * beat_division)
            note_index_in_beat = int((beat_of_note-beat_of_note)*4)
            beats[beat_of_note,note_index_in_beat] = note.nameWithOctave

        return beats

    @staticmethod
    def beat_level_similarity(xml_files):
        for xml_file in xml_files:
            beats = np.array(Analyzer.get_beat_list(xml_file))
            self_similarity_matrix = beats.reshape(1, -1) == beats.reshape(-1, 1)
            print(self_similarity_matrix)
            plt.imshow(self_similarity_matrix)
            name = xml_file.strip('../data/abc/')
            name = name.strip('/')
            name = name.strip('generated_template_for_polyphony.xml')
            name = name.strip('/')
            plt.show()
            # plt.savefig('/Users/zengren/Desktop/plots/self_similarity'+name+'.png')

    @staticmethod
    def form_analysis(xml_files):
        segmentation_dict = {}
        for xml_file in xml_files[:]:
            m21_file = music21.converter.parse(xml_file)
            repeats = list(m21_file.flat.getElementsByClass(music21.bar.Repeat))
            repeats_start = [repeat for repeat in repeats if repeat.direction == 'start']
            repeats_end = [repeat for repeat in repeats if repeat.direction == 'end']
            stats = collections.Counter([repeat.offset for repeat in repeats])
            repeat_offsets = [repeat.offset for repeat in repeats]

            segmentation_points = repeat_offsets + [0.0, m21_file.duration.quarterLength]
            unique_segmentation_points = sorted(list(set([int(round(x / 3)) for x in segmentation_points])))

            large_partition = list(zip(unique_segmentation_points, unique_segmentation_points[1:]))
            segmentation = tuple([y - x for (x, y) in large_partition])
            # print(segmentation)
            segmentation_dict[xml_file.strip('../data/abc/')] = segmentation
        print(collections.Counter(segmentation_dict.values()))

        # other_partitions = [x for x in segmentation_dict.values() if len(x)==1 or len(x)>3]

        def plot_binary_form_distribution():
            binary_partitions = [x for x in segmentation_dict.values() if len(x) == 2]
            binary_part_array = np.array(binary_partitions)
            print(binary_part_array)
            plt.title('distribution of binary forms (x bars + y bars), N=' + str(binary_part_array.shape[0]))
            H, xedges, yedges = np.histogram2d(binary_part_array[:, 0], binary_part_array[:, 1])
            plt.imshow(H)
            plt.xlim(0, 31)
            plt.ylim(0, 31)
            plt.xticks(np.arange(30))
            plt.yticks(np.arange(30))
            plt.show()

        def plot_ternary_partitions():
            ternary_partitions = [x for x in segmentation_dict.values() if len(x) == 3]
            ternary_part_array = np.array(ternary_partitions)
            fig = plt.figure()
            ax = fig.add_subplot(projection='3d')
            ax.scatter(ternary_part_array[:, 0], ternary_part_array[:, 1], ternary_part_array[:, 2], alpha=0.3)
            ax.set_title('distribution of ternary forms, N=' + str(ternary_part_array.shape[0]))
            ax.set_xlim(0, 15)
            ax.set_ylim(0, 15)
            ax.set_xticks(np.arange(15))
            ax.set_yticks(np.arange(15))
            ax.set_yticks(np.arange(15))
            plt.show()

        # plot_binary_form_distribution()

    @staticmethod
    def key_analysis(xml_files):
        keys = []
        for i, xml_file in enumerate(xml_files[:]):
            print(i)
            m21_file = music21.converter.parse(xml_file)
            key = m21_file.analyze('key')
            keys.append(key.tonicPitchNameWithCase)
            del m21_file

        print(collections.Counter(keys))
        plt.hist(np.array(keys))
        plt.show()

    @staticmethod
    def pattern_analysis(xml_files):
        concatenated_beat_list = np.empty(shape=(1,4),dtype=str)
        total_files = len(xml_files)
        for i,xml_file in enumerate(xml_files):
            print(str(i) + '/' + str(total_files))
            beat_list = Analyzer.get_beat_list(xml_file)
            concatenated_beat_list = np.append(concatenated_beat_list,beat_list,0)

        return concatenated_beat_list


def common_subsequence(lst):
    pass


if __name__ == '__main__':
    print('start')
    #xml_files = ['../data/xml/' + file_name for file_name in
    #             filter(lambda x: x.endswith('generated_template_for_polyphony.xml'), sorted(os.listdir('../data/xml')))]

    #beat_sequence = Analyzer.pattern_analysis(xml_files[:])

    #np.savez('beat_sequence',beat_sequence=beat_sequence)

    #beat_sequence = np.load('beat_sequence.npz',allow_pickle=True)['beat_sequence']
    #reshaped1 = beat_sequence.reshape((-1,1,4,1))
    #reshaped2 = beat_sequence.reshape((1,-1,1,4))
    #similarity_array = reshaped1==reshaped2
    #np.savez('similarity_array', similarity_array=similarity_array)

    similarity_array = np.load('similarity_array.npz',allow_pickle=True)['similarity_array']
    print(similarity_array)
    beat_similarity_array = np.sum(similarity_array,axis=(-1,-2))
    print(beat_similarity_array)
    np.savez('beat_similarity_array', beat_similarity_array=beat_similarity_array)
    #arg_sort = np.argsort(beat_occurances)
    #sorted_beat = beat_sequence[arg_sort]
    #print(list(zip(sorted_beat,beat_occurances[arg_sort])))
