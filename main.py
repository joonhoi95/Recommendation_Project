import custom_based
import load_data
import sys
import time


if __name__ == '__main__':

    try:

        data = input("당신의 취향을 선택해주세요 :")

        df_book = load_data.load_excel()
        nm_reco = load_data.load_parameter(df_book)

        #nm_reco = 'Fahrenheit 451'

        custom_based.custom_recommender(nm_reco)

        time.sleep(20)

    except KeyboardInterrupt:

        sys.exit()  # 종료