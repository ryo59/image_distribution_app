# collect_images.py
import requests
import os
from urllib import parse
from dotenv import load_dotenv
from pathlib import Path

import config


def make_dir(path_to_member_images):
    """
    各メンバーの画像を保存するためのフォルダを作成
    :param path_to_member_images: ./images/neru_nagahama/
    :return: None
    """
    if not os.path.isdir(path_to_member_images):
        os.mkdir(path_to_member_images)


def create_image_path(path_to_images_dir, member_name, url):
    """
    画像を保存するためのパスを作成する
    :param path_to_images_dir: 保存先のディレクトリ ex. ./images
    :param member_name: メンバー名 ex. neru_nagahama
    :param url: 検索で返ってきた画像のURL 拡張子を得るために必要
    :return:画像の保存先へのパス ex. ./images/neru_nagahama/1_neru_nagahama.jpg
    """
    path_to_member_images_dir = os.path.join(path_to_images_dir, member_name)
    global serial_number_of_member_images
    serial_number_of_member_images += 1

    file_extension = os.path.splitext(url)[-1]
    if file_extension.lower() in (".jpg", ".jpeg", ".png"):
        path_to_member_images = os.path.join(path_to_member_images_dir,
                                             str(serial_number_of_member_images) +
                                             "_" + member_name + file_extension)
        return path_to_member_images
    else:
        raise ValueError("Not Applicable file extension")


def search_images_by_member_name_phrase(params, api_key):
    """
    「メンバー+キーワード」の検索フレーズで
    :param params: リクエストにわたすパラメーター "q": search_word ex. 長濱ねる かわいい
    :param api_key: azure_api_key
    :return: エンドポイントからのレスポンス
    """
    bing_image_search_endpoint = "https://api.cognitive.microsoft.com/bing/v7.0/images/search"

    response_from_endpoint = requests.get(bing_image_search_endpoint,
                                          headers={"Ocp-Apim-Subscription-Key": api_key},
                                          params=params,
                                          allow_redirects=True,
                                          timeout=10)

    if response_from_endpoint.status_code != 200:
        error = Exception("HTTP status code: " + str(response_from_endpoint.status_code))
        raise error

    return response_from_endpoint


def create_image_url_list_per_search_word(search_word, member_image_id_list, max_get_image_number=150, required_image_number=600):
    """
    検索ワード単位で画像のURLのリストを得る
    :param search_word: 検索ワード ex. "長濱ねる", "長濱ねる かわいい", "長濱ねる 顔"
    :param member_image_id_list: imageIdを要素にもつリスト 重複の検証に利用する
    :param max_get_image_number: エンドポイントへの一度のリクエストで得るページ数 最大150
    :param required_image_number: 欲しい画像の枚数 大きすぎると重複したデータが大量に来る ダメ
    :return: image_url_list_per_search_word: 検索ワードで得られた画像URLのリスト, member_image_id_list: imageIdのリスト,
             duplication_count_per_word: 重複したデータの個数,
    """
    offset_count = required_image_number // max_get_image_number

    image_url_list_per_search_word = []
    duplication_count_per_word = 0
    for offset in range(offset_count):
        params = parse.urlencode({'q': search_word,
                                  'count': max_get_image_number,
                                  'offset': offset * required_image_number})
        try:
            res = search_images_by_member_name_phrase(params=params, api_key=BING_API_KEY)
            response_json = res.json()
        except Exception as err:
            print("Error No.{}: {}".format(err.errno, err.strerror))
        else:
            for values in response_json['value']:
                # imageIdで重複データの検証
                image_id = values["imageId"]
                if image_id in member_image_id_list:
                    duplication_count_per_word += 1
                else:
                    member_image_id_list.append(image_id)
                    image_url_list_per_search_word.append(parse.unquote(values["contentUrl"]))

    return image_url_list_per_search_word, member_image_id_list, duplication_count_per_word


def get_member_image_binary_from_url(_image_url):
    """
    画像のURLにリクエストを送る
    content-typeがimageでないものはエラーを投げる
    :param _image_url: 検索フレーズで検索して返ってきた画像のurl
    :return: 画像URLからのレスポンスボディをバイナリ形式で返す
    """
    response_from_image_url = requests.get(_image_url, timeout=10)

    if response_from_image_url.status_code != 200:
        error = Exception("HTTP status code: " + str(response_from_image_url.status_code))
        raise error

    content_type = response_from_image_url.headers['content-type']
    if "image" not in content_type:
        error = Exception("Content-Type: " + content_type)
        raise error

    return response_from_image_url.content


def save_image(file_path, image):
    """
    画像を保存する
    :param file_path: 画像へのパス ex. ./images/neru_nagahama/1_neru_nagahama.jpg
    :param image: 画像のバイナリデータ
    :return: None
    """
    with open(file_path, "wb") as f:
        f.write(image)


if __name__=="__main__":
    env_path = Path('.') / '.env'
    load_dotenv(dotenv_path=env_path) # python3 only
    BING_API_KEY = os.getenv("AZURE_BING_SEARCH_API_KEY")
    SAVE_IMAGE_DIR = os.path.abspath(os.path.dirname(__file__)) + "\\images"

    MAX_GET_IMAGE_NUMBER = 150  # 最大が150 デフォルトは10
    REQUIRED_IMAGE_NUMBER = 600  # 必要な画像の枚数（なんだが、だいたい3~3.5倍くらいの画像が集まる）

    KEYAKI_MEMBER_NAMES_JA = config.keyaki_members_ja
    KEYAKI_MEMBER_NAMES_EN = config.keyaki_members_en

    for member_name_ja, member_name_en in zip(KEYAKI_MEMBER_NAMES_JA, KEYAKI_MEMBER_NAMES_EN):
        # メンバー名リストを回して、各メンバー名のフォルダを作成する
        path_to_member_image_dir = os.path.join(SAVE_IMAGE_DIR, member_name_en)
        # ex. C:\\Myprojects\\KeyakiApp\\images\\neru_nagahama
        make_dir(path_to_member_image_dir)

        image_id_list = []  # メンバー名単位で重複を検証する（"メンバー名 キーワード"のフレーズ単位ではない）
        total_duplication_count = 0  # メンバー名単位で重複したデータ数を知るため
        serial_number_of_member_images = 0  # 画像に連番をつけるため

        print("Member name: ", member_name_ja)
        for combined_word in config.combined_words_ja:
            # キーワードを結合して検索
            search_term = member_name_ja + " " + combined_word
            search_term = search_term.strip()

            image_url_list, image_id_list, duplication_count = create_image_url_list_per_search_word(search_term, image_id_list)
            total_duplication_count += duplication_count
            print("Search word: ", search_term)
            print("Value length: ", len(image_url_list))
            print("Duplication count per {}: {}".format(search_term, duplication_count))
            for image_url in image_url_list:  # URLリストでfor文を回して画像を保存する
                try:
                    image_content = get_member_image_binary_from_url(image_url)
                    full_path_to_member_image = create_image_path(SAVE_IMAGE_DIR, member_name_en, image_url)
                    save_image(file_path=full_path_to_member_image, image=image_content)
                    print("Saved image file: ", full_path_to_member_image)
                except KeyboardInterrupt:
                    break
                except Exception as err:
                    print(err)

        print("Total duplication count per {}: {}".format(member_name_ja, total_duplication_count))