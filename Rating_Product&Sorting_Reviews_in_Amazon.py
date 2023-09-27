###################################################
# PROJE: Rating Product & Sorting Reviews in Amazon
###################################################
import pandas as pd
import math
import scipy.stats as st
from sklearn.preprocessing import MinMaxScaler

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', 500)
pd.set_option('display.expand_frame_repr', False)
pd.set_option('display.float_format', lambda x: '%.5f' % x)

###################################################
# İş Problemi
###################################################

# E-ticaretteki en önemli problemlerden bir tanesi ürünlere satış sonrası verilen puanların doğru şekilde hesaplanmasıdır.
# Bu problemin çözümü e-ticaret sitesi için daha fazla müşteri memnuniyeti sağlamak, satıcılar için ürünün öne çıkması ve satın
# alanlar için sorunsuz bir alışveriş deneyimi demektir. Bir diğer problem ise ürünlere verilen yorumların doğru bir şekilde sıralanması
# olarak karşımıza çıkmaktadır. Yanıltıcı yorumların öne çıkması ürünün satışını doğrudan etkileyeceğinden dolayı hem maddi kayıp
# hem de müşteri kaybına neden olacaktır. Bu 2 temel problemin çözümünde e-ticaret sitesi ve satıcılar satışlarını arttırırken müşteriler
# ise satın alma yolculuğunu sorunsuz olarak tamamlayacaktır.

###################################################
# Veri Seti Hikayesi
###################################################

# Amazon ürün verilerini içeren bu veri seti ürün kategorileri ile çeşitli metadataları içermektedir.
# Elektronik kategorisindeki en fazla yorum alan ürünün kullanıcı puanları ve yorumları vardır.

# Değişkenler:
# reviewerID: Kullanıcı ID’si
# asin: Ürün ID’si
# reviewerName: Kullanıcı Adı
# helpful: Faydalı değerlendirme derecesi
# reviewText: Değerlendirme
# overall: Ürün rating’i
# summary: Değerlendirme özeti
# unixReviewTime: Değerlendirme zamanı
# reviewTime: Değerlendirme zamanı Raw
# day_diff: Değerlendirmeden itibaren geçen gün sayısı
# helpful_yes: Değerlendirmenin faydalı bulunma sayısı
# total_vote: Değerlendirmeye verilen oy sayısı



###################################################
# GÖREV 1: Average Rating'i Güncel Yorumlara Göre Hesaplayınız ve Var Olan Average Rating ile Kıyaslayınız.
###################################################

# Paylaşılan veri setinde kullanıcılar bir ürüne puanlar vermiş ve yorumlar yapmıştır.
# Bu görevde amacımız verilen puanları tarihe göre ağırlıklandırarak değerlendirmek.
# İlk ortalama puan ile elde edilecek tarihe göre ağırlıklı puanın karşılaştırılması gerekmektedir.


###################################################
# Adım 1: Veri Setini Okutunuz ve Ürünün Ortalama Puanını Hesaplayınız.
###################################################
df1 = pd.read_csv("Measurement_Problems/Case1 - Amazon/amazon_review.csv")
df = df1.copy()
df.head()
df.shape

df["overall"].mean()
# Out : 4.587589013224822


###################################################
# Adım 2: Tarihe Göre Ağırlıklı Puan Ortalamasını Hesaplayınız.
###################################################
df.head()
df.info()

df["reviewTime"] = pd.to_datetime(df["reviewTime"])

df["reviewTime"].max()    # 2014-12-07

current_date = df["reviewTime"].max()
df["day_diff"] = (current_date - df["reviewTime"]).dt.days

df["day_diff"].quantile([0.25, 0.5, 0.75])
# 0.25000   280.00000
# 0.50000   430.00000
# 0.75000   600.00000



def time_based_weighted_average(dataframe, w1=28, w2=26, w3=24, w4=22):
    return dataframe.loc[df["day_diff"] <= 280, "overall"].mean() * w1 / 100 + \
           dataframe.loc[(dataframe["day_diff"] > 280) & (dataframe["days"] <= 430), "overall"].mean() * w2 / 100 + \
           dataframe.loc[(dataframe["day_diff"] > 430) & (dataframe["days"] <= 600), "overall"].mean() * w3 / 100 + \
           dataframe.loc[(dataframe["day_diff"] > 600), "overall"].mean() * w4 / 100

time_based_weighted_average(df)

# Out: 4.577117672912792


###################################################
# Adım 3: Ağırlıklandırılmış puanlamada her bir zaman diliminin ortalamasını karşılaştırıp yorumlayınız.
###################################################

df.loc[df["day_diff"] <= 280, "overall"].mean()
# 4.6957928802588995
df.loc[(df["day_diff"] > 280) & (df["day_diff"] <= 430), "overall"].mean()
# 4.636140637775961
df.loc[(df["day_diff"] > 430) & (df["day_diff"] <= 600), "overall"].mean()
# 4.571661237785016
df.loc[(df["day_diff"] > 600), "overall"].mean()
# 4.4462540716612375

# Sonuçları değerlendirecek olursak müşteri memnuniyeti son zamanlarda giderek artmış.



###################################################
# Görev 2: Ürün için Ürün Detay Sayfasında Görüntülenecek 20 Review'i Belirleyiniz.
###################################################


###################################################
# Adım 1. helpful_no Değişkenini Üretiniz
###################################################

# Not:
# total_vote bir yoruma verilen toplam up-down sayısıdır.
# up, helpful demektir.
# veri setinde helpful_no değişkeni yoktur, var olan değişkenler üzerinden üretilmesi gerekmektedir.


df["helpful_no"] = df["total_vote"] - df["helpful_yes"]
df.head()

###################################################
# Adım 2. score_pos_neg_diff, score_average_rating ve wilson_lower_bound Skorlarını Hesaplayıp Veriye Ekleyiniz
###################################################

def score_up_down_diff(up, down):
    return up - down

def score_average_rating(up, down):
    if up + down == 0:
        return 0
    return up / (up + down)

def wilson_lower_bound(up, down, confidence=0.95):
    """
    Wilson Lower Bound Score hesapla

    - Bernoulli parametresi p için hesaplanacak güven aralığının alt sınırı WLB skoru olarak kabul edilir.
    - Hesaplanacak skor ürün sıralaması için kullanılır.
    - Not:
    Eğer skorlar 1-5 arasıdaysa 1-3 negatif, 4-5 pozitif olarak işaretlenir ve bernoulli'ye uygun hale getirilebilir.
    Bu beraberinde bazı problemleri de getirir. Bu sebeple bayesian average rating yapmak gerekir.

    Parameters
    ----------
    up: int
        up count
    down: int
        down count
    confidence: float
        confidence

    Returns
    -------
    wilson score: float

    """
    n = up + down
    if n == 0:
        return 0
    z = st.norm.ppf(1 - (1 - confidence) / 2)
    phat = 1.0 * up / n
    return (phat + z * z / (2 * n) - z * math.sqrt((phat * (1 - phat) + z * z / (4 * n)) / n)) / (1 + z * z / n)



# score_pos_neg_diff
df["score_pos_neg_diff"] = df.apply(lambda x: score_up_down_diff(x["helpful_yes"],
                                                                             x["helpful_no"]), axis=1)

# score_average_rating
df["score_average_rating"] = df.apply(lambda x: score_average_rating(x["helpful_yes"], x["helpful_no"]), axis=1)

# wilson_lower_bound
df["wilson_lower_bound"] = df.apply(lambda x: wilson_lower_bound(x["helpful_yes"], x["helpful_no"]), axis=1)

##################################################
# Adım 3. 20 Yorumu Belirleyiniz ve Sonuçları Yorumlayınız.
###################################################

df.sort_values("wilson_lower_bound", ascending=False).head(20)

# WLB, hem yararlı oyları hem frekansı göz önünde bulundurur.
# Sonuçlara baktığımızda olumlu oy sayısı ve değerlenmesi yüksek olanları gözlemledik.
# Olumsuz oy almayan az sayıdaki yorumları da görebildik.


