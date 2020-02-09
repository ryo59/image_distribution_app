from flickrapi import FlickrAPI
from urllib.request import urlretrieve
from pprint import pprint
import os, time, sys

#APIキーの情報

key = "0f81e60000e5908c7bdf556aa6622c74"
secret = "a171be7b0876d668"
wait_time = 1

#保存フォルダの指定
animalname = sys.argv[1]
savedir = "./images/" + animalname

flickr = FlickrAPI(key, secret, format= "parsed-json")
result = flickr.photos.search(
    text = animalname,
    per_page = 400,
    media = "photos",
    sort = "relevance",
    safe_search = 1,
    extras = "url_q, licence"
)

photos = result["photos"]
#pprint(photos)

for i, photo in enumerate(photos["photo"]):
    url_q = photo["url_q"]
    filepath = savedir + "/" + photo["id"] + ".jpg"
    if os.path.exists(filepath): continue
    urlretrieve(url_q, filepath)
    time.sleep(wait_time)