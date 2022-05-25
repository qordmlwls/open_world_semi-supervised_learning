from modules.SNS_content_text_one_module_for_garbage_ossl import SnsContentClassifier
import timeit
import pandas as pd
import os


if __name__=='__main__':
    start_time = timeit.default_timer()
    ###############오늘 SNS contents 데이터 로드(하루에 3000건 정도)########################
    # dbgetter = getdb()
    # SNS_content_df = dbgetter.get_content_SNS()
    # SNS_content_df = pd.read_csv('../input/sns_content_fortest4_label23.csv')
    # SNS_content_df = pd.read_csv('../input/for_garbage_test.csv')
    SNS_content_df = pd.read_csv(os.path.join(os.getcwd(), 'input', 'jytest_ossl_50_revised.csv'))
    ################SNS contents 분류 #######################################################
    SNS_contents_classifier = SnsContentClassifier()
    result = SNS_contents_classifier.run(SNS_content_df)
    # print(len(result[0]['y_pred']))
    print(result)
    result.to_csv('../output/garbage_classification_result_ossl.csv',mode='w',encoding='utf-8-sig')
    terminate_time = timeit.default_timer() # 종료 시간 체크  
    print("%f초 걸렸습니다." % (terminate_time - start_time)) 
