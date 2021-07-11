import numpy as np
from IPython.display import display, HTML
import warnings
warnings.filterwarnings('ignore')
"""
Implemented by
Yonsei App.Stat. Sunwoo Kim
"""

class Musivisor() :
    def __init__(self, name, B, song_meta_data, music_id) :
        self.B = B
        self.song_meta_data = song_meta_data
        self.music_index = music_id
        self.name = name
        print('안녕하세요 %s님 당신만을 위한 음악추천. Musivisor입니다.'%(self.name))
        print('지금부터 Musivisor 사용 방법을 설명드리겠습니다.\n먼저, Musivisor는 4GB의 램을 필요로 합니다. 램이 부족하신경우, 다른 프로그램을 종료하여 충분한 램 공간을 확보해주세요.')
        print('1. 원하시는 곡 개수를 넣고 input_generator 함수를 시행해주세요. 이어 곡의 이름을 넣고 실행하시면, 가이드라인이 나옵니다.')
        print('2. input_generator 의 시행이 종료되면, prediction_generator를 시행해주세요!')

    def input_generator(self, n) :
        rec_list = []
        music_names = []
        for k in range(n) :
            your_answer = False
            while your_answer == False :
                search_key = input()
                first_step = self.song_meta_data[self.song_meta_data['song_name'].str.contains(search_key, regex = False)]
                can_rec = first_step[first_step.song_id.isin(self.music_index)]
                display(HTML(can_rec.to_html()))
                print('원하시는 곡이 있습니까? 있다면 y, 없다면 n')
                answer_ = input()
                if answer_ == "y" :
                    your_answer = True
                else :
                    print('다른 노래를 입력해주세요.')
            print('\n곡의 번호를 입력해주세요')
            music_code = input()
            rec_list.append(int(music_code))
            music_name = self.song_meta_data[self.song_meta_data.song_id == int(music_code)].song_name.values[0]
            music_names.append(music_name)
            print('{}이 당신의 리스트에 추가되었습니다'.format(music_name))
            print('다음 곡을 골라주세요.')
            print('\n')
        print('다음 음악들이 추천의 재료로 들어갑니다!', music_names)
        self.rec_list = rec_list
        return self.rec_list
    def prediction_generator(self, num) :
        print('노래가 생성중입니다.. 잠시만 기다려주세요..\n')
        x1 = np.repeat(0, self.B.shape[0])
        for name_ in self.rec_list :
            x1[np.where(self.music_index == name_)] = 1
        rec_vec = x1.dot(self.B)
        print('노래가 선택되었습니다! 추출중입니다..')
        rec_result = self.song_meta_data.iloc[self.music_index[np.where(rec_vec > np.sort(rec_vec)[::-1][num])], :]
        print('{}님. 오늘은 이런 음악 어떠세요?'.format(self.name))
        return rec_result