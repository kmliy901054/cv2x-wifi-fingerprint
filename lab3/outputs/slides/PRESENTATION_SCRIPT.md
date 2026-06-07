# WiFi Indoor Localization: from a 1.57 m baseline to a 0.75 m honest cascade

> A step-by-step engineering arc on 1,812 real lab scans — and an honesty audit that caught its own overfitting  
> NYCU CV2X Lab 3

英文投影片標題/條列 + 中文口稿。共 33 張投影片。

可開 `lab3_presentation.pptx` 編輯;這份是給你對稿/排練用的文字版。

---

## 1. From a 1.57 m KNN baseline to a 0.75 m honest cascade
*Standard protocol 0.75 m, strict nested-CV 0.94 m — and we caught our own overfitting*

圖: `outputs/figures/ladder_bar.png`

口稿: 先把這份報告的定位講清楚:它講的是一個工程過程,不是單一模型的炫技。題目來自 Lab 2,我們用 ESP32-S3 裝在 TurtleBot3 上,在約一百八十九平方公尺的實驗室收 WiFi 訊號強度,目標是讓機器人不靠 GPS 在室內定出自己的座標。我會從完全不用訓練的 KNN 基準線 1.57 公尺講起,一步步說明每次卡關、為什麼非換方法不可,最後怎麼把中位數誤差砍掉一半到 0.75。但我真正想讓老師記住的不是數字變小,而是這份工作的誠實:我們一度跑出非常漂亮的 0.65,卻自己揪出那是在測試集上挑出來的假象,於是拒絕當成績,改用最嚴格、完全無洩漏的巢狀五折交叉驗證把真實泛化釘回到約 0.94。所以請先把三個數字記在心裡:標準協定的 0.75 是誠實標題,巢狀交叉驗證的 0.94 是最保守上限,0.65 是我們親手抓出來的海市蜃樓。畫面這張就是貫穿全場的爬坡圖,從 1.57 一路往下,等一下每一格我都會回來細講。(這張是全場的情緒高潮、留到結果段才正式走一遍,開場只需點一下三個數字就推進,別在這裡耗時間。)

---

## 2. Four required sections, told as one honest engineering story
*Problem, Method, Dataset & splits, Results — framed by an honesty audit*

- 題目定義 / Problem: one variable-length scan to (x, y) in metres, no GPS
- 做法 / Method: Set Transformer to grid classification to coarse-gated cascade
- 資料集切分 / Dataset: 1,812 scans, 4 splits (A main, C cross-time hardest)
- 實驗結果 / Results: 1.57 to honest 0.75 m, with a self-caught 0.65 m mirage

口稿: 這張是路線圖,讓老師一眼確認作業規定的四個段落我全都會講到,但我刻意串成一條工程故事線而不是乾巴巴的四點交代。第一段題目定義:輸入是一筆長度會變的 WiFi 掃描、本質是 (BSSID, RSSI) 的集合,輸出是公尺座標,室內沒有 GPS。第二段做法是主軸:把掃描當集合餵進 Set Transformer,把回歸改成在網格上分類,最後做成粗到細的 cascade。第三段資料集與切分:1812 筆掃描,設計四種切法,A 隨機切是主線數字,C 是早訓晚測也是最難的一關。第四段實驗結果:從 1.57 爬到誠實的 0.75,中間穿插那個被我們自己抓出來的 0.65 假象。我用文字卡而不放圖,是要把誠實這個框架先用口頭立起來,真正有衝擊力的誠實驗證圖留到結果段第一次揭曉、效果最強。如果老師問四段比重,做法和結果是重頭戲,題目和資料各用兩三張快速但紮實鋪好背景。

---

## 3. 題目定義 / Problem Definition
*What exactly are we predicting, and why is it hard?*

口稿: 進入第一個規定段落:題目定義。這一段只做一件事——把輸入、輸出、為什麼難講到底,因為後面每一個方法的轉折,本質都是在回應這裡列出的某一個難點。(分段標題,口頭一句話帶過就推進,不要逐句念這段筆記。)

---

## 4. The task: turn one variable-length WiFi scan into a point on the map
*Input = a set of (BSSID, RSSI); output = (x, y) in metres*

- Input: one scan = a variable-length set of (BSSID, RSSI) pairs
- Output: robot position (x, y) on the 189.5 m² floor plan
- No GPS indoors — WiFi fingerprinting is the classic fallback
- Four difficulties: variable set, ambiguous RSSI, sparse coverage, 0.3 m label floor

口稿: 把題目定清楚。輸入是一筆 WiFi 掃描,本質是一個集合:一堆 BSSID 加上各自的 RSSI 訊號強度配成對,而且關鍵是每一筆掃到的 AP 數量都不一樣,有時三四十個、有時只有十幾個。輸出是機器人在這個 189.5 平方公尺實驗室地圖上的座標,單位是公尺。先說評估口徑,後面所有數字都用同一把尺:誤差指的是預測座標到 AMCL 地面真值的歐氏距離、單位公尺,我報的都是中位數。為什麼用 WiFi?因為室內收不到 GPS,而 WiFi 訊號到處都有、現成免裝,指紋定位就是業界最經典的室內備案。

重點是這題為什麼難,而且是四個難點同時並存、互相疊加,後面每一步我都會回來指其中一個。第一,輸入是變動長度的無序集合,不是固定向量,這是之後改用 Set Transformer 的理由。第二,RSSI 多峰又會混淆——房間裡兩個不同地點的訊號組合可能幾乎一模一樣,也就是 ambiguous RSSI,一個輸入對到兩個合理答案,這是後面把回歸改成網格分類的理由。第三,覆蓋稀疏,資料沿著機器人走過的路徑收、不是均勻鋪滿,上半部幾乎沒走到,這是後面用高斯過程補洞的理由。第四,地面真值是 ROS AMCL 給的,本身就帶約 0.3 公尺定位噪聲,這是準度的物理天花板,意思是再強的模型也不可能系統性地比標籤本身還準。預期會被問:既然 GPS 不行,為什麼不用藍牙信標或 UWB?那些都要額外佈建硬體跟成本,而 WiFi 是環境裡免費既有的訊號,這份工作的價值正是只靠現成 WiFi 加對的模型,就把誤差壓到接近標籤精度。

---

## 5. Anatomy of one scan: a ragged bag of (BSSID, RSSI)
*Same place, two scans, different lengths — order means nothing, presence means everything*

- One scan = an unordered bag like {(ap_3f, -52), (ap_1a, -67), (ap_b2, -78), ...}
- Length varies scan-to-scan (~12 to ~40 APs); same spot need not list the same APs
- Force it into an 80-D vector and a typical scan is ~2/3 -100 padding (27.7 of 80)
- Order carries no meaning; only which APs are present and how strong they are

口稿: (這張是深入剖析、屬於可快可慢的彈性投影片:若時間緊,口頭講兩句『一筆掃描就是一袋無序又變長的配對,硬塞成固定向量會塞進一堆假欄位』就推進;若有時間或被問到再展開。)後面整條技術路線的第一個轉折完全建立在一筆掃描長什麼樣這件事上,值得攤開。一筆原始掃描就是一袋無序的配對,像 ap_3f 強度 -52、ap_1a 強度 -67、ap_b2 強度 -78 這樣列下去,單位 dBm,越接近零代表訊號越強、離 AP 越近,大概落在 -40 到 -90 之間。注意兩個字:無序、變長。無序是這一袋裡誰先誰後完全沒意義;變長是不同掃描裝的 AP 數量不一樣,少則十幾個、多則三四十個。

更麻煩的是同一個位置連續掃兩次,列出的 AP 清單都不見得一樣——有的 AP 這次剛好掃到、下次因為通道忙或訊號瞬間掉到底噪就沒列進來。所以同一地點的兩筆掃描可能長度不同、成員也略有出入,卻該對到同一個座標。核心矛盾就在這:資料天生是無序、變長、成員會抖動的集合,可是傳統 MLP 只吃固定維度、有固定欄位順序的向量。硬塞會怎樣?KNN 跟第一版 MLP 就是先固定一張 80 個 AP 的詞彙表,把掃描攤成 80 維,某 AP 沒掃到就在那一格填 -100。問題是一筆典型掃描平均只點亮 27.7 個 AP,等於約三分之二的維度都是 -100 填充,模型有三分之二的輸入是假的、在學著忽略不存在的東西,白白浪費容量。結論一句、也承先啟後:資料本來就是集合,就該用對集合的模型去處理它——這直接預告下一段把每個 AP 當 token、用 Set Transformer 的轉折。預期追問:那把 RSSI 正規化成 0 到 1、缺席填 0 不就好?那只是換個數值,集合無序跟變長兩個本質問題還在,治標不治本。

---

## 6. 資料集切分 / Dataset & Splits
*1,812 real scans, a coverage hole, and four generalisation stress-tests*

口稿: 進入第二個規定段落:資料集與切分。請先記住一個數字 1812、一句話『上半部幾乎沒走到』,這兩個會貫穿全場——資料的形狀決定了模型的天花板。(分段標題,口頭兩句帶過就推進。)

---

## 7. Every scan is a real robot run: ESP32 on a TurtleBot3 doing SLAM
*Hardware-collected data, not simulation — and that dictates every dataset quirk*

- ESP32-S3 sniffer rides a TurtleBot3 running Cartographer SLAM + AMCL
- Each scan ~3 s; ground truth (x, y) is AMCL's pose (~0.3 m noise)
- Coverage = wherever we drove — path-shaped, not a uniform grid
- Two sessions (morning + evening) to expose later cross-time drift

口稿: 在看數字之前先說清楚資料怎麼長出來的,因為後面所有的優點跟毛病都從這裡來。硬體是一顆 ESP32-S3 當 WiFi 嗅探器,綁在一台 TurtleBot3 上;機器人同時跑 Cartographer 做 SLAM 建圖、再用 AMCL 做即時定位。也就是說每一筆掃描的地面真值座標不是人工標的,而是 AMCL 在那一刻吐出的位姿。這帶來三個直接後果。第一,標籤本身有噪音:AMCL 定位誤差約 0.3 公尺,這就是我們反覆提到的物理底線,後面誤差分析那 27% 貼底線的點根源就在這裡。第二,取樣節奏:每筆掃描約 3 秒才完成,一場下來才幾百到一千筆,這也是為什麼訓練資料只有約一千四百筆、規模偏小,後面失敗實驗段才會得出『資料少時容量沒用』的結論。第三、也最關鍵:覆蓋等於我們開過的路徑,機器人走過哪裡那裡才有資料,沒走到的地方就是空白,這直接造成下一張會看到的上半部覆蓋缺口,而那正是後面 GP 合成資料要補的洞。最後,我們刻意分早上、晚上兩場收,就是為了之後驗證跨時段漂移、也就是最難的 C。預期會被問:為什麼不乾脆鋪滿整個房間均勻取樣?因為機器人是自走的、要靠 SLAM 安全導航,上半部在那次任務的路徑規劃裡沒被排到,加上時間有限,所以才留下這個缺口——這也正好讓我們有機會展示用高斯過程補洞的整套方法。(我刻意不在這張放覆蓋圖,把那個空洞的第一次揭曉留給下一張,衝擊力最強。)

---

## 8. 1,812 fingerprints in a 189.5 m² lab — but the upper region is a coverage hole
*Two sessions, variable-length scans, AMCL ground truth (train/test = 1449/363)*

- 1,812 scans = morning 912 + evening 900; train/test split 1449/363
- 115 BSSIDs seen, 80 kept (>=10 hits); the rest are transient noise
- Ground truth from ROS AMCL, ~0.3 m noise — our hard error floor
- Coverage follows the driven path; the upper region is barely visited

圖: `outputs/figures/data_coverage.png`

口稿: 這是整份資料集的全貌,也是覆蓋圖的第一次揭曉,請先看這張圖最重要、也最會被忽略的一件事:地圖上半部幾乎是空的。我們在約 189.5 平方公尺的實驗室收了 1812 筆掃描,早上 912、晚上 900,刻意分兩時段(理由上一張講過,為了驗證跨時段泛化)。最後切成訓練 1449、測試 363,下一段 splits 看到的 test=363 就是從這裡來。每一筆掃描都是大小會變的 BSSID 與 RSSI 集合。關於 AP 詞彙表:現場一共看到 115 個不同 BSSID,但只留出現十次以上的 80 個,其餘多半是鄰場手機熱點、訪客裝置這類進進出出的瞬時雜訊,留著只會讓模型學到不穩定特徵。地面真值由 ROS AMCL 給,本身帶約 0.3 公尺噪音,這是誤差的物理下限。最後請把這個上半部的空洞記在心裡,它在這份報告裡會出現至少三次:它對應到可靠度圖上的高誤差熱區、對應到最差十案例的聚集處、也正是我加合成資料的根本原因。預期會被問:363 筆測試夠不夠代表性?在 A_random 它是全分布隨機抽的,涵蓋走過的主要區域,對同分布準度有代表性;它代表不了的恰恰是沒走到的上半部,這個侷限我會誠實保留。

---

## 9. Four splits, each stress-testing a different generalisation axis
*A_random is the main number; C (cross-time) is the hardest*

- A_random (MAIN): 80/20 over all data, test=363 — accuracy ceiling
- B_morning: same-session — isolates model quality from time drift
- C_morning to evening: train AM, test PM — the hardest, cross-time test
- D_stratified: early/late mix — guards against trajectory-order leakage

口稿: 所有數字背後都要先問一句:這是哪一種切法算出來的?同一個模型在不同切法下可以差到兩三倍,所以切法不是細節,是結果可信度的核心。我們設了四種,每一種刻意壓一個不同的泛化軸。第一,A_random 是主力,把全部 1812 筆隨機八二切,測試集 363 筆;我整場報的中位數預設都是這個切法,它回答『跟訓練同分布時,準度天花板在哪』。第二,B_morning 同時段切,只拿早上那場自己切自己,刻意把模型本身好不好跟時段漂移隔開——如果 B 好但 C 差,就能斷定問題出在時間而不是模型。第三,C 是早上訓練、晚上測,這是最難的一關,中間隔了好幾小時,環境裡的人、門開關、其他裝置都變了,RSSI 會整體漂移,是唯一沒解掉的軸。第四,D 是早晚混合的分層切,防的是一個很隱蔽的陷阱:軌道順序洩漏——機器人連續走,相鄰時間點位置幾乎一樣,若隨機切讓相鄰兩點一個進訓練一個進測試,模型等於偷看到測試點旁邊的答案、會高估準度;D 用分層避免這種相鄰洩漏。一句話幫大家記:A 看上限,B 看純模型力,C 看跨時段,D 防洩漏。各切法的數字我放到結果段、講完 cascade 之後再攤開,免得在還沒解釋模型前就丟數字。預期會被問:既然 C 最接近真實長期部署,為什麼主數字用 A?因為 A 是學界回報指紋定位準度的標準口徑,用它才能跟別人比較;C 我會誠實另外報,當成這份工作的開放問題,而不是藏起難關只報好看的。(這張我用文字卡,不放散點圖——那張散點來自最終 cascade,在還沒介紹 cascade 前放會造成前向引用。)

---

## 10. Coverage, not noise, is the real bottleneck
*RSSI is noisy but repeatable; error is driven by how many known APs match*

- Coverage, not noise, is the bottleneck — the thesis of this whole deck
- Within-cell RSSI std ~3.5 dBm; morning-to-evening drift only ~0.8 dBm
- Error spikes when few known APs match, then flattens
- Diagnosis: fix coverage, not the network — motivates GP-synth later

圖: `outputs/figures/error_vs_aps_drift.png`

口稿: 這張下一個關鍵診斷,而且我把結論直接放第一條,因為它是整份報告的主軸:真正限制準度的是覆蓋,不是噪音。我用兩張子圖把它釘死。先看噪音這條線——直覺上大家以為 WiFi 訊號很吵所以定位不準,所以我們先量到底多吵:同一格、同一位置重複量到的 RSSI 標準差大約只有 3.5 dBm;更關鍵的是早上到晚上的中位數漂移只有約 0.8 dBm,非常小。右邊這張早上對晚上的 RSSI 散點就是證據,點大致緊貼對角線,意思是訊號雖有抖動但高度可重複。也就是說『吵』被高估了,它不是主因。那主因是什麼?看左邊這張才是重點:橫軸是一筆掃描能對到多少個訓練過的已知 AP,縱軸是誤差,圖上那條紅線是每個 AP 數對應的中位誤差,整體中位數約 0.79。當能對到的已知 AP 很少時誤差陡然飆高,一旦對到夠多就很快趨平。兩張拼起來講同一件事:你在某個地方不準,不是因為訊號太吵,而是那裡資料太少、能比對的已知 AP 太少——這正是覆蓋問題,直接對應那塊沒走到的上半部。這個診斷一旦成立,方向就清楚了:與其把網路加深加大去硬扛噪音,不如去把覆蓋補起來,這就是後面用每個 AP 各自擬合一個高斯過程、生合成掃描來填洞的根本動機;後面那張可靠度圖會用『誤差地圖幾乎是密度地圖的鏡像』正式為這個判斷蓋章。預期會被問:0.8 dBm 漂移這麼小,為什麼跨時段的 C 卻差那麼多?那是中位數漂移,看似小卻是系統性的整體平移,加上少數 AP 在晚上整個消失或新出現,這種分布級偏移足以讓沒見過晚場的模型大幅退步——空間覆蓋跟時間漂移是兩個獨立的軸,這張解的是空間軸,時間軸留待結論。

---

## 11. 做法 / Method
*The climb: each method change fixes one named problem*

口稿: 進入第三個、也是最核心的規定段落:做法。先把貫穿全段的關鍵詞定義一次:我整場會一直講『歸納偏置』(inductive bias),白話就是把問題本身的結構——掃描是集合、答案落在地圖網格上——直接寫進模型架構裡,而不是靠堆參數硬學。接下來每換一次方法,都不是趕流行追更大的模型,而是在解前面點過名的某一個難點:變動長度的集合、回歸把多峰答案平均掉、覆蓋不足、對稱位置的訊號混淆。我會用很笨但很誠實的順序爬:先給不訓練的 KNN 當地板,再給第一版神經網路釣出兩個教訓,然後依序解掉表示、覆蓋、平均問題,最後用 cascade 收尾。(分段標題,把『歸納偏置』這個白話定義講出來就推進。)

---

## 12. Classic KNN fingerprinting sets the bar to beat at 1.57 m
*The no-training baseline every neural net must outscore*

- Scan to 80-D RSSI vector; unheard AP padded to -100 dBm (not heard)
- k=5 (a-priori, not tuned on test) nearest neighbours, distance-weighted pose
- Zero training, pure lookup: the honest reference point
- Median error 1.57 m on Split A test

圖: `outputs/figures/architectures/arch_knn.png`

口稿: 從基準線開始,我刻意選最樸素、最沒花招的方法,因為好的爬坡曲線需要誠實的起點。最經典的指紋定位就是 KNN:把一筆掃描攤成 80 維 RSSI 向量,沒掃到的 AP 一律填 -100(就是剛剛那袋硬塞成固定向量的做法),這個 -100 dBm 代表低於接收機底噪、根本收不到,而不是真有 -100 的訊號。然後在這 80 維空間裡找最近的五個訓練樣本,用距離加權平均它們的座標當預測,完全不訓練、純查表。先把一個常見質疑堵起來:k=5 是事先依經驗定的,不是拿測試集調出來的最佳值,否則連基準線都偷看答案、整條爬坡就不誠實了。中位數誤差 1.57 公尺,就是整場要被打倒的對象。我想強調的不只是它是對手,而是它定義了門檻:後面任何一個又大又貴的神經網路,如果連這個零訓練查表都贏不了,就沒有存在的理由,這是我評估每一步的鐵則。請記住 1.57,接下來你會看到它怎麼一步步被砍到剩一半。

---

## 13. Plain neural nets barely beat KNN — and reveal two lessons
*MLP 1.30 m, MaskedMDN 1.37 m: the wins come from fixing these, not from depth*

- Lesson 1: a dense -100-padded vector wastes most of the capacity
- Lesson 2: regression averages multimodal answers into the void
- MLP regression: 1.30 m — only a sliver below KNN
- MaskedMDN: 1.37 m — a mixture head alone does not fix multimodality

圖: `outputs/figures/architectures/arch_masked_mdn.png`

口稿: 我故意把兩個教訓放最上面、數字壓下面,因為這張的價值不在那兩個成績,而在它釣出來的兩個問題,這兩個問題決定了後面整條路。先對齊畫面:這張結構圖是 MaskedMDN,對應下面那個 1.37 的版本,別跟純 MLP 的 1.30 搞混。第一版神經網路只小贏 KNN 一點點,從 1.57 到 1.30,差不多就是查表跟一個普通回歸器的差距,但它讓兩個獨立問題浮出來。教訓一是資料表示的浪費:把每筆掃描硬塞成固定 80 維、缺的補 -100,結果一筆典型掃描裡大半維度都是填充值,模型把容量花在學著忽略假輸入(就是上一張那個三分之二填充的問題)。教訓二更致命,是回歸的死穴:當一筆訊號其實對應到房間裡兩個都合理的位置時,L2 回歸的最佳解就是輸出兩者平均,而那平均往往落在中間空地、兩個正確答案一個都沒命中。我們本想用 MDN 混合高斯輸出讓模型同時表達好幾個可能位置,結果 1.37 還比單純 MLP 的 1.30 差——這個反直覺結果證明光換輸出頭不能真正解掉多峰,還得靠後面在地圖網格上分類。所以接下來順序是:下一張先正面討論為什麼要預測分布而不是單點,再解表示問題,最後回來用網格分類把平均問題算總帳。常見追問:那 MDN 不就白做了?沒有,它讓我們確認問題出在輸出空間的幾何、而不是模型不夠大,這個診斷直接省下後面亂加參數的彎路。

---

## 14. Why predict a distribution, not a single point
*One scan can map to two plausible places — a point estimate splits the difference*

- Symmetric layouts make one RSSI fingerprint match two real positions
- L2 regression's optimum is the mean — it lands in the empty middle
- MDN expresses multiple peaks but its readout still averages them
- Cleanest fix: classify over map cells, where peaks stay separate

口稿: (這是純概念的鋪陳投影片,我刻意把它定位成『一次說清楚、之後兌現兩次』:它是 turn-heatmap 跟 why-cascade 兩張的共同地基。若時間緊,口頭一句『同一筆訊號可能對到兩個地點,回歸取平均會掉到中間空地,所以我們改在網格上分類』就能推進,後兩張再現場兌現。)把問題畫在腦海:我們實驗室有對稱的走道跟相似隔間,結果地圖上兩個物理相距好幾公尺的位置,量到的 AP 集合跟 RSSI 幾乎一樣,這叫對稱混淆。對只輸出單一座標的回歸模型,這是無解的兩難——同一筆輸入卻有兩個都對的標籤,L2 損失逼它輸出兩者平均以最小化期望誤差,而那平均通常落在兩地中間空地,於是它選了一個兩邊都不對的點,這就是上一張回歸贏不了多少的根因。一個自然念頭是用 MDN 輸出好幾個高斯峰,理論上能同時說『我覺得是這裡或那裡』;問題出在最後一步:終究要給一個座標交差,而從混合分布讀出一點時常用的期望值或加權平均又把兩峰拉回中間,繞一圈又掉回平均陷阱,這正是 1.37 沒贏的根因。真正乾淨的解法是改變輸出空間幾何:不在連續座標回歸,而把地圖切成離散格子做分類,讓兩個峰各佔不同格子、不被內插混在一起,需要決策時取最大機率峰、或先用粗尺度證據壓掉錯的那個峰。預期會被問:那分類最後不也要取中心、不也會平均?會,但只在單一連通的峰內取中心,那是合理的次格內插;真正的雙峰是被粗網格的 gating 在相乘那步直接否決掉一個,不是被平均——這個差別到 why-cascade 我會用真實案例指給大家看。

---

## 15. Treating the scan as a SET, not a vector, unlocks 1.09 m
*First real break from the KNN pack (single model)*

- Each AP = one token: BSSID embedding + scaled RSSI
- Set Transformer: 3 self-attention blocks, pooling-by-attention to 192-d
- Permutation-invariant (order doesn't matter); only real APs enter
- Median 1.09 m, -16% vs the MLP

圖: `outputs/figures/architectures/arch_set_transformer_mdn.png`

口稿: 這是第一個真正的轉折,直接回應前面的表示問題。一筆 WiFi 掃描本質就是一個大小會變的集合,不是固定向量;有時三四十個 AP、有時十幾個,硬塞成 80 維本身就錯。所以把每個掃到的 AP 當成一個 token,內容是該 BSSID 的 embedding 加上正規化的 RSSI,丟進 Set Transformer,做三層自注意力,再用 pooling-by-attention 壓成 192 維向量。先把兩個術語用白話講掉,免得 bullet 上看不懂:排列不變(permutation-invariant)就是『AP 的順序變了,輸出答案不變』;pooling-by-attention(PMA)就是『用注意力做加權匯總,讓模型自己決定哪些 AP 比較重要』。這架構天生對順序不敏感,而且只有真正掃到的 AP 會進去,徹底擺脫 -100 填充。中位數一口氣降到 1.09,第一次跟 KNN 拉開明顯距離。提醒一句,1.09 是單模型,後面的里程碑才開始用集成。核心訊息:換對資料的型態,比換更大的模型有用。下一張我會把這個集合到底怎麼被讀的機制拆開,那是老師最容易追問細節的地方。

---

## 16. How a Set Transformer reads a variable-length set
*Tokenise APs, let them attend to each other, then pool by attention*

- Each (BSSID, RSSI) becomes a token: learned ID embedding + scaled RSSI
- Self-attention = APs compare to each other, building joint context
- Permutation-invariant: no positional encoding, order carries no meaning
- PMA: a learned query attends over all tokens to a single 192-d vector

圖: `outputs/figures/architectures/arch_set_transformer_mdn.png`

口稿: (BACKUP / Q&A 用:這是四個機制塞一張的深入投影片,上一張已經摘要過。口頭那一遍只需講『PMA 為什麼勝過平均池化』這一點,其餘除非被問到注意力細節再展開,否則快速推進。)第一步是 token 化:每個掃到的 AP 變成一個 token,組成兩塊——這個 BSSID 透過一張可學習的查找表得到一段 embedding,代表它的身分與隱含空間特性,再把該次量到的 RSSI 正規化後接上去。注意這個查找表是綁定在固定的 80 個 AP 詞彙表上的,所以詞彙表外、從沒見過的新 BSSID 目前不被編碼、會被丟掉,這也是 live demo 要靠八到九成詞彙重疊率的原因。一筆掃描就是一串長度不固定的 token,有幾個 AP 就有幾個 token,完全不補零。第二步自注意力是最關鍵的:三層自注意力讓每個 AP token 去看其他所有 token、互相比較加權,等於模型在問『同時看到這幾個 AP、各自這個強度,這個組合共同指向哪裡』——單看一個 AP 很模糊,但一群 AP 的相對強度關係能把位置鎖得很準。第三步排列不變性:我們刻意不加任何位置編碼,因為掃描裡 AP 的先後純粹是硬體的偶然、沒有語意,不加位置編碼正好讓模型對順序完全免疫,同一組 AP 換任何排列輸出都一致。第四步 PMA:用一個可學習的查詢向量對全部 token 做一次注意力匯總,讓模型自己決定哪些 AP 對定位更重要,收斂成單一個 192 維場景向量交給輸出頭。為什麼用 PMA 不用平均池化?因為平均會把強訊號 AP 跟邊緣弱訊號 AP 一視同仁,而 PMA 能學會偏重資訊量大的 AP,對稀疏掃描特別有幫助——這是這張唯一一定要講出口的點。預期追問:token 數量不一樣不會出問題嗎?不會,自注意力跟 PMA 對任意長度集合都自然成立,這正是我們選它的全部理由。

---

## 17. GP-kriging synthetic data: the single biggest jump, -17% to 0.91 m
*Filling the coverage hole beats any architecture change*

- Per-BSSID Gaussian Process learns a smooth position-to-RSSI field
- Sample 5000 synthetic scans from the empty upper region
- Split A: 1.09 to 0.91 m (-17%) — the project's largest single gain
- In-session win; cross-time drift is revisited at the end

圖: `outputs/figures/synth_ablation.png`

口稿: 這是整份報告單一影響力最大的一招,請在這裡停久一點,因為它證明了整份工作的主軸——限制準度的是覆蓋,不是模型。先把標題那個 kriging 用白話講掉:kriging 就是用高斯過程在沒資料的位置做有原則的空間內插。回到最開頭那張覆蓋圖的痛點:機器人沿路徑收資料,房間上半部幾乎沒走到,模型在那塊根本沒見過樣本、只能瞎猜。解法不是再換更聰明的模型,而是去把缺的資料補回來,但補得有物理根據。具體做法是對每一個 BSSID 各自擬合一個二維高斯過程,輸入位置、輸出該 AP 的 RSSI,等於替每個 AP 學出一張平滑的訊號強度地形圖;高斯過程的好處是它是有原則的平滑內插,在沒資料的地方給平滑而合理的外推與不確定度,生出來的訊號可信而不是隨機噪音。接著在可行空間特別是那塊空白上半部抽樣生成五千筆合成掃描塞回訓練集。看左邊 CDF,整條往左移,中位數從 1.09 掉到 0.91,進步約 17%,比任何一次換架構都大。口徑要講清楚以免被質疑:這個基準是爬坡圖上純真實單模型的 1.093,合成資料把它降到 0.906,這兩個數字在 ladder_bar 上都查得到、同口徑可比;再上五種子集成可到 0.889。右邊長條:A 大進步,但 C 早訓晚測反而從約 1.73 微升到約 1.81——因為合成資料補的是空間覆蓋,解不了時間漂移,跨時段是另一個完全獨立的軸,最後再回來。一句話:資料量只有一千多筆時,正確補上覆蓋勝過任何模型花招。下一張我把這個高斯過程怎麼生出可信掃描、又怎麼驗證它沒亂生,完整拆給大家看。

---

## 18. How GP-kriging fabricates believable coverage
*Per-AP 2-D Gaussian Process + a KNN detection model, sanity-checked against reality*

- One 2-D GP per AP maps position to a smooth RSSI field with uncertainty
- A KNN detection model decides whether each AP is even visible here
- Both models are fit on TRAINING positions only — never on test poses
- Sanity check: synth mean 27.1 APs/scan vs real 27.7 — well-matched

圖: `outputs/figures/synth_ablation.png`

口稿: (BACKUP-eligible:這張的口頭版只要 lead 那個 27.1 對 27.7 的體檢數字就很有說服力,流程細節有時間或被問到再展開。)我要回答一個很合理也很尖銳的質疑:你生的是假資料,憑什麼相信它沒把模型帶歪、甚至沒洩漏測試集?所以這張把生成流程拆成兩個模型加一道體檢,並先把無洩漏這點講死。第一個模型是強度:對每個 AP 各擬一個二維高斯過程,從位置映到 RSSI,GP 不只給內插值還給每個位置的不確定度,本質是平滑場,在沒走過的上半部生出來的訊號會平滑延續鄰近真實量測、不會憑空冒尖峰,這就是 kriging 的精神。但光有強度不夠真實,因為不是每個 AP 在每個位置都收得到,離得遠或被牆擋就掃不到;所以第二個模型是偵測:用一個 KNN 偵測模型,根據鄰近真實掃描判斷在這個位置某個 AP 看不看得見,只有判定可見的 AP 才放進這筆合成掃描、再由 GP 填強度。最關鍵的無洩漏原則:這兩個模型——GP 跟 KNN 偵測——都只用訓練集的位置去擬合,從不碰任何測試位姿,合成掃描也只生在訓練分布裡,所以標準 A 切法的 0.91 是乾淨的;而且在後面巢狀交叉驗證裡,合成資料是每一折用該折的訓練資料重新生成的,不會把外層測試折的資訊漏進去,否則那個 0.94 就不成立。我們在可行空間抽樣五千個位置生成,偏重那塊空白上半部。最後那道體檢是這張的重點數字:合成掃描平均每筆 27.1 個 AP,真實掃描平均 27.7,兩者非常接近——代表偵測模型沒生出每筆塞滿一百個 AP 的假掃描、也沒生出稀疏到不真實的掃描,合成資料的稀疏結構跟真實世界對得上。我把這個 27.1 對 27.7 當成這招可信度的單一最強證據。預期追問:GP 在完全沒資料的地方外推會不會不準?會,離真實量測越遠不確定度越大,但即使是平滑帶不確定度的合理猜測,也遠勝模型在那塊完全沒訓練訊號、只能隨機亂猜,實測 -17% 就是這個取捨划算的證明。

---

## 19. The training recipe: augment, 5 seeds, geometric-median ensemble
*How we turn one architecture into a stable, deployable predictor*

- Train on real + 5000 synthetic scans together, not in stages
- Augment: RSSI jitter +-4 dBm and 10% random AP dropout per scan
- Train 5 seeds; combine with a geometric-median ensemble (outlier-robust)
- Same recipe across every neural milestone — a fair, fixed comparison

圖: `outputs/figures/train_recipe.png`

口稿: (BACKUP-eligible:這張是工程細節,口頭版一定要講出口的是最後一點——固定配方讓爬坡的進步可歸因於架構;其餘擴增與集成的細節有時間或被問到再展開。)從 set-transformer 之後的每一個神經里程碑都用同一套訓練配方,先講清楚後面報數字才公平。畫面這張流程圖就是配方三環節。第一,資料:真實資料跟前一張生的五千筆合成資料是混在一起一次訓練的,不是先用真實再微調,因為我們要模型把合成的上半部覆蓋當成訓練分布的一等公民。第二,擴增,目的是逼模型對 WiFi 訊號天生的不穩定產生抵抗力:RSSI jitter 加減 4 dBm,模擬同一位置不同時刻訊號的小幅起伏,讓模型別把某個精確 dBm 值背死;AP dropout 隨機丟掉 10% 的 AP,模擬某些 AP 偶爾掃不到,逼模型不要把全部賭注押在某一兩個 AP 上。第三,集成:每個架構用五個不同隨機種子各訓一個,再用幾何中位數把五個預測合起來。先把幾何中位數用白話講掉:它就是一種對離群點穩健的『平均』,跟算術平均不同,少數跑很遠的壞預測不會把它往外拉。為什麼用它?因為五個種子裡若有一個在某筆掃描上被對稱混淆騙到跑很遠,幾何中位數會自動把它壓下去、只取多數共識的位置,這正好對應我們的誤差結構——大尾巴來自少數嚴重離群。最重要、也是這張存在的理由:這套配方是固定的,從 set-transformer、heatmap 到 cascade 全部沿用,所以爬坡圖上那些進步可以歸因於架構本身,而不是某一步偷換了更好的訓練技巧。順帶補一個成本數字,因為後面巢狀交叉驗證等於把這套配方重跑五遍:單一種子訓練約十幾分鐘量級,五種子加合成資料一輪在單張 GPU 上是數小時可完成的工程量,這也是為什麼我們把嚴格巢狀 CV 留給主線、沒對每個 split 都重跑。預期追問:五個種子差異大嗎?有自然波動,這也是後面誠實審計用 bootstrap 標準誤判斷一個改動是不是真有意義、還是只是種子噪音的原因。

---

## 20. Classifying on a masked grid beats regressing a coordinate: 0.88 m
*Predict a probability map, not a point — kills the averaging problem*

- Predict P(cell) over a 40x33 grid (~0.4 m cells), not a coordinate
- Free-cell mask zeroes walls; centroid reads off the position
- Classification can express multiple peaks; regression cannot
- Median drops 1.09 to 0.88 m (5-seed ensemble)

圖: `outputs/figures/architectures/arch_heatmap.png`

口稿: 現在回來解 first-nn 埋下的第二顆地雷,也就是 why-distribution 鋪陳過的回歸取平均問題(一句話回顧:同一筆訊號對到兩個地點,回歸會掉到中間空地)。前兩步已把編碼器換成正確的集合表示、又用合成資料補了覆蓋,環境乾淨了,可以專心對付多峰。做法就是把 why-distribution 的結論落地:不在連續座標回歸,改成在地圖上分類。我們把整層切成 40 乘 33 的網格,每格約 0.4 公尺見方,模型對每格輸出機率,等於畫一張位置的機率熱圖,而不是吐一個座標。這裡有一個工程上很關鍵的細節——free-cell 遮罩:把落在牆裡、家具裡、物理上機器人不可能站的格子機率直接歸零,等於把地圖結構當先驗硬編進輸出,模型不必浪費容量學那些格子不可能,也杜絕了預測點掉進牆裡的荒謬。最後從遮罩過的熱圖取機率加權中心讀出座標。結果中位數從 set-transformer 的 1.09 直接壓到 0.88,一樣五種子幾何中位數集成、配方完全相同。順帶說一下為什麼是 0.4 公尺一格:這個解析度刻意比 AMCL 0.3 公尺的標籤底線略粗一點,因為再細也分不出比標籤更精的位置、只會讓每格樣本變太少而難訓練,0.4 公尺正好在『細到能定位、又粗到每格有足夠樣本』之間取得平衡。核心訊息:分類這個輸出形式天生能在不同格子同時表達多個峰,回歸在數學上做不到,這就是它贏的根因。但它還留一個破口——兩個峰勢均力敵時,取加權中心仍會被拉到中間,單層分類自己解不掉,這個破口交給下一張的 cascade 用粗網格守門員來補。

---

## 21. Our champion: a coarse 10x9 gate over the fine 40x33 grid lands 0.79 m
*One encoder, two heads — loss-weight tuning then takes it to 0.75 m*

- Shared encoder feeds a coarse 10x9 + a fine 40x33 head
- Gate: P_fine(cell) x P_coarse(parent region) before the centroid
- 0.79 m median (5-seed) — best of every config we tried
- Loss-weight tuning (mse_w 0.55) to 0.752 m — the reported headline

圖: `outputs/figures/architectures/arch_cascade.png`

口稿: 這是整份報告的冠軍架構,粗到細的串接,它正面解掉上一張 heatmap 留下的最後破口——勢均力敵的雙峰。結構:一個共用的 Set Transformer 編碼器,後面接兩個分類頭,粗網格 10 乘 9、細網格 40 乘 33。直覺是分工合作:粗頭先回答大概在哪個區域,細頭負責區域內的精確位置。真正的關鍵在 gating——在取座標之前,把細網格每一格的機率乘上它所屬粗格的機率,讓粗頭當守門員;細頭就算在某個錯誤區域冒出一個很高的假峰,只要粗頭不相信那個區域、給低機率,相乘後那假峰就被壓到趨近零、等於被否決,加權中心不再被它往外拉。這就是為什麼 cascade 能解掉前面所有方法都解不掉的對稱混淆。結果中位數從 heatmap 的 0.88 再降到 0.79,一樣五種子集成。我要強調這不是運氣:它是把更大、更複雜的各種架構全比過一輪後實測勝出留下來的那個,後面爬坡圖跟失敗實驗會佐證。順帶回答一個關於規模的常見疑問:這個冠軍 cascade 其實是個輕量模型,參數量遠小於後面失敗的 cascade-big 那 170 萬——贏的是結構不是體積,這正呼應整份報告的主軸。最後一條很重要,先把橋接講清楚免得爬坡圖最後一格讓人意外:把兩個頭的損失權重調整、特別是把 mse 權重設到 0.55 之後,中位數再降到 0.752,這個 aggressive 版的 0.75 就是我整場一直用的標題數字。下一張我用一個真實案例,把這個守門員怎麼把雙峰變單峰一格一格指給大家看。

---

## 22. Why it works: the coarse gate kills the fine grid's false second peak
*A real example — one multiplication turns two peaks into one*

- Fine head alone is bimodal: correct peak + a plausible decoy
- Probability-weighted centroid lands between them, off target
- Coarse head is confident about the region, suppresses the decoy
- After gating: one clean peak, prediction snaps back on target

圖: `outputs/figures/gating_before_after.png`

口稿: 用一個真實案例證明 gate 不是玄學,我會邊講邊指畫面上三個面板,這也是 why-distribution 那條伏筆的正式兌現。左邊是細網格單獨的輸出,你會清楚看到它是雙峰:正確位置附近有一個峰,但地圖另一頭也冒出第二個幾乎一樣高的峰——這就是前面一再提到的對稱混淆,兩地的 AP 集合跟 RSSI 指紋太像、細頭自己分不出來。因為我們取機率加權的期望位置,兩個勢均力敵的峰一拉扯,預測點被拖到中間空地、整個偏掉,圖上這一筆單看細頭是 2.59 公尺。中間是粗網格的輸出,它很篤定地只點亮正確那一個區域、完全沒被假峰騙到,因為在粗尺度上整個區域的證據被一起匯總,那個錯誤區域的整體證據不足以支撐。右邊是兩者相乘後的結果:落在錯誤區域的假峰被粗頭的低機率乘成幾乎為零,整張圖只剩正確位置一個乾淨單峰,加權中心立刻彈回——同一筆從 2.59 降到 0.90。這就是粗網格當守門員的全部價值:一次簡單相乘,解掉了回歸跟單層分類都解不掉的對稱混淆,而且它解的方式是否決錯峰、不是平均兩峰,正好回應 why-distribution 留的伏筆。預期追問:萬一粗頭自己也判錯區域怎麼辦?那確實會連帶把細頭帶偏,這也是為什麼這類案例還是會出現在最後誤差分析的離群裡;但絕大多數情況下粗尺度證據比細尺度更穩、更不容易被局部混淆騙到,統計上這個守門員是淨賺的,爬坡圖跟 CDF 都證明它整體最好。看完原理,我們正式進入實驗結果那一段,上完整爬坡圖宣布 cascade 勝出。

---

## 23. 實驗結果 / Experimental Results
*The full climb, the honesty audit, the failures, and the live demo*

口稿: 進入第四個、也是最後一個規定段落:實驗結果。如果只記一件事,就記標準協定 0.75、最嚴格巢狀交叉驗證 0.94,這兩個口徑我從頭用到尾。(分段標題,口頭兩句帶過就推進。)

---

## 24. The full climb: every method change bought real metres, 1.57 to 0.75 m
*Same Split A test set (363 scans), median error per experiment*

- Two biggest jumps: the set representation and synthetic coverage
- Right inductive bias, not raw capacity, drove every gain
- Green = kept milestone, red = abandoned variant
- Caption: set-transformer onward = 5-seed; KNN/MLP/MDN = single model

圖: `outputs/figures/ladder_bar.png`

口稿: 這張是整份報告的主結果,也是開場那張爬坡圖的正式揭曉,請大家花點時間看,我會在這裡放慢、把該強調的講透。所有 bar 都跑在同一個 Split A 測試集、同樣 363 筆掃描上,每一條是一次實驗的中位數誤差,單位公尺,越低越好。我特意把原本一長串箭頭文字拿掉了,因為這張長條圖本身已經把故事講清楚,文字再抄一次只會讓畫面變髒。口頭走一遍這條路:最樸素的 KNN 是 1.57;把掃描從固定向量改成集合、用 Set Transformer 編碼,掉到 1.09;加 GP-kriging 合成資料補那塊空白覆蓋,降到 0.91;把輸出從回歸座標改成在網格上分類,到 0.88;再用粗網格守門細網格的 cascade,到 0.79;最後損失權重調好的 aggressive 版落在 0.75。整段有兩個最大跳躍、各砍了將近兩成:一個是把掃描當集合而不是向量,一個是用合成資料補覆蓋——注意這兩個贏的都不是把模型做大,而是換對的歸納偏置,這正是整份報告希望老師記住的核心訊息。顏色上綠色是留下的里程碑、紅色是試過放棄的變體。誠實的小字 caption 我會口頭講出來免得被質疑:從 set-transformer 之後是五個種子集成,前面 KNN、MLP、MDN 是單模型,所以這條線是工程歷程、不是嚴格同口徑的對照。常被問:那為什麼不全部都用集成重跑?因為集成只在已選定的好架構上才有意義,KNN 這種查表法沒有種子可言,我們要呈現的是真實的開發順序。最乾淨、最誠實的標題就是 0.75 公尺。

---

## 25. Across all four splits: in-session is strong, cross-time is the honest weak spot
*Cascade median per split — A measured, B/C/D honest estimates*

- A_random: 0.75 m (MEASURED, main headline)
- B_morning: ~0.85 m (same-session, estimate)
- D_stratified: ~0.90 m (all-session deployment, estimate)
- C_morning to evening: ~2.3 m (cross-time, ESTIMATE) — the open problem

圖: `outputs/figures/synth_ablation.png`

口稿: 兌現前面 splits 那張的承諾——把每種切法的數字真正給出來,我特意把它放在講完 cascade、看過完整爬坡之後,這樣大家已經知道 cascade 是什麼,這些數字才有意義。先把誠實聲明放最前面,這是這張的關鍵:只有 A_random 的 0.75 是 cascade 完整重跑、確切測得的(bullet 上我用 MEASURED 標出);B、C、D 因為時間限制沒有對每個 phase 都把整套 cascade 加合成資料重訓重測,所以是根據早期 baseline 與消融趨勢推出來的估計值,bullet 上都標 estimate,請大家不要把約 2.3 當成精確結果引用——這正是這份報告強調誠實的一致態度。趨勢本身很清楚也很合理:A 約 0.75;同時段的 B 約 0.85,只比 A 略差;涵蓋早晚的部署情境 D 約 0.90,難一點;而跨時段的 C 一口氣跳到約 2.3,差一個量級。這個落差不是模型爛,而是訊號本身在幾小時內漂掉了,模型訓練時根本沒看過晚上的分布。畫面這張合成資料的消融長條圖正好佐證:合成資料在 A 帶來大進步,但在 C 反而從約 1.73 微升到約 1.81——因為合成資料補的是空間覆蓋、解不了時間漂移。結論一句:同時段我們做得很好且有實測背書,跨時段是這份工作唯一沒解掉、也誠實承認的弱點,解法(IMU、時序融合、晚場資料)留到結論。預期會被問:為什麼不現在把 C 也重跑出精確值?坦白說是算力與時間取捨,我們把有限資源投在把主線 A 做到最嚴格、做到誠實審計;與其給四個都不完整的精確數字,不如給一個紮實的主數字加三個誠實標註的估計。

---

## 26. The whole distribution improved, not just the median
*Cascade is best at every percentile (median 0.79 / mean 1.12 / p90 2.56 m)*

- Cascade CDF sits left of every other model at every percentile
- 27% of predictions within the 0.3 m AMCL floor
- Median 0.79 m, mean 1.12 m, p90 2.56 m
- >2 m tail from sparse regions + symmetric-layout ambiguity

圖: `outputs/figures/error_cdf.png`

口稿: 這張誤差累積分布圖,專門回答一個我預期老師一定會問的質疑:你是不是只挑了中位數好看、其他地方其實很糟?答案是沒有,我用整條曲線證明。CDF 的讀法是:橫軸誤差公尺、縱軸是有多少比例的預測落在這個誤差以內,曲線越往左上越好。Cascade 這條線在每一個百分位都壓在其他所有模型左邊,意思是不論你看中位數、看 25 分位、還是看尾巴,它都比較好,這個贏是全面的不是局部的。三個關鍵數字要記:中位數 0.79、平均 1.12、p90 2.56。這裡要特別講一件事:為什麼平均 1.12 比中位數 0.79 高那麼多?因為有一條超過兩公尺的長尾把平均往上拉,平均對少數極端離群很敏感,而中位數不會——這個落差本身就是『分布有重尾』的證據,也預告了後面要拆的離群問題。圖上我畫了一條 0.3 公尺的參考線特別重要:有整整 27% 的預測落在這條線以內,而 0.3 正是 AMCL 標籤本身的定位噪音,意思是這些點已經準到跟地面真值一樣準、不可能再進步。右邊那條超過兩公尺的尾巴是這份工作真正的難點,來自兩個來源:稀疏區域跟對稱位置混淆——這兩個判斷下一張可靠度圖會佐證稀疏、再後面最差十案例圖會佐證對稱混淆。預期會被問:那為什麼不直接優化平均、把尾巴壓下去?因為這任務有一塊物理上不可解的離群(後面誤差分析會證明約 16% 是同指紋對到兩地),優化平均會被那一小撮不可解的點綁架、反而犧牲典型表現,中位數才是穩健、能真實反映一般使用者體驗的指標。

---

## 27. Where to trust it: reliability tracks data density, not network size
*Per-cell error mirrors the sample-density map — the coverage thesis, proven*

- Per-cell mean error mirrors the sample-density map almost exactly
- Dense, well-travelled regions: well under 0.5 m
- Sparse upper region: error climbs above 2 m
- The fix is more data, not a bigger model

圖: `outputs/figures/region_reliability.png`

口稿: 這張圖正式為貫穿整份報告的主軸蓋上句點:到底是什麼在限制我們的準度。畫面是並排兩張地圖,左邊是把測試誤差畫回每一個網格格子的平均誤差熱圖,右邊是每一格的訓練樣本密度。請老師直接比對兩張的形狀——它們幾乎是鏡像,也就是『誤差地圖約等於密度地圖』,這就是整份報告最直接的證明。走得多、資料密的主要動線區域誤差遠低於 0.5 公尺,模型在那裡非常可靠;而房間上半部那塊從第一張覆蓋圖就一直空著的區域,誤差爬到兩公尺以上,因為那裡根本沒有足夠訓練資料讓模型學。這個對比給出一個很乾淨的結論:限制準度的天花板是覆蓋,不是訊號噪音、也不是網路容量。這件事有三重意義。第一,從實驗上驗證了前面加 GP 合成資料補覆蓋是對的方向。第二,直接預告結論的未來方向——要再進步該去補資料、而不是疊更深的網路。第三、也最重要:它解釋了為什麼下一段誠實審計裡,我們把模型加大到 170 萬參數反而沒用,因為瓶頸從來不在容量,這正是下一張失敗實驗的伏筆。所以這不只是一張診斷圖,它把整條故事線收束起來。預期會被問:那為什麼不把可靠度當成信心指標輸出給使用者?這是很好的延伸,我們可以用每格密度當作預測信心的代理,讓系統在稀疏區主動標示低信心,這是部署層面的好點子。

---

## 28. How we measured honestly: leak-free nested 5-fold CV is the strictest mirror
*Per-fold full retrain, only 4/5 of the data each fold — no test peeking anywhere*

- Outer 5 folds: each fold is a held-out test, never touched during training
- Per fold we retrain everything from scratch on the other 4/5 — no shared weights
- Model/hparam selection happens INSIDE each fold, never on the outer test
- Result: 0.94 ± 0.04 m — the most pessimistic, most trustworthy number

圖: `outputs/figures/nested_cv.png`

口稿: 在揭曉我們怎麼抓出自己的過擬合之前,我得先花一整張投影片把我們用什麼尺去量講清楚,否則下一張的數字會沒有重量。問題的根源是:一次性的 train 到 test 切分,只要你在開發過程中反覆看那個測試集去挑架構、挑超參,測試集就慢慢被你偷看光了,報出來的數字會偏樂觀。要徹底擋掉這種洩漏,最嚴格的做法是巢狀交叉驗證。先把 nested 這個詞用白話講掉:巢狀的意思就是有兩層、而且選模型用的內層跟最後評估用的外層完全分開、互不污染。照圖講:外層把全部資料切成五折,每一折輪流當完全沒被碰過的測試集;對每一折,我們不是共用同一份權重去測,而是從零開始、只用另外五分之四的資料把整個模型重新訓練一遍——注意這個細節,每折只有 80% 的資料可用,本來就會比用全部資料訓練略差,這是嚴格性的代價。最關鍵的一點在內層:所有模型選擇、超參數調整,全部關在那一折內部、只用該折的訓練資料做,絕不准拿外層那個測試折來挑任何東西;連前面的合成資料也是每折用該折訓練資料重新生成、不讓外層測試折漏進來。五折跑完,把五個誤差平均,得到 0.94 加減 0.04。我要強調這個數字的性格:它最悲觀、最保守,因為同時承受了每折只有 80% 資料、又完全禁止偷看的雙重懲罰,但正因如此它也最可信、最接近真實部署的泛化。預期會被問:那為什麼標題還是報 0.75 而不是 0.94?因為 0.75 是學界通用的標準 full-train 到 test 協定、跟別人論文可比;0.94 是我們自己加碼的最嚴格內部稽核,兩個都報、口徑都標清楚才是誠實。下一張就用這把尺照出那個 0.65 的海市蜃樓。

---

## 29. We caught our own overfitting: 0.650 m was a mirage, 0.752 m is the honest headline
*Three numbers, one figure: 0.752 standard / 0.94 strict nested-CV / 0.650 mirage*

- Greedy ensemble of 33 candidates hit 0.650 m — but picked ON the test set
- Held-out val exposed it at 0.710 m — a ~0.07 m selection bias we refused to report
- Strictest leak-free nested 5-fold CV (4/5 data per fold) = 0.94 ± 0.04 m
- SWA (top literature method) moved 0.025 < bootstrap SE 0.037 -> not adopted

圖: `outputs/figures/honest_validation.png`

口稿: 這是我整場最想讓大家看的一張,也是這份作業真正的價值所在,請給我多一點時間。我用上一張那把巢狀交叉驗證的尺,加上這張圖,把四個數字的關係一次釘死,免得被誤會成在挑好看的講。我會直接走這張圖,因為一次出現四個數字、聽眾容易跟丟:圖上從左到右是我們親手抓出的 0.650、獨立驗證集現形的 0.710、誠實標題 0.752、以及最嚴格的巢狀 CV 0.94。故事是這樣:我們曾跑出 0.650 這個非常漂亮的數字,做法是在 33 個候選模型裡用貪婪搜尋一個一個加進集成、看哪個組合在測試集上最低就留。但我們自己警覺到——這個搜尋是直接拿測試集當評分標準的,等於一邊考試一邊偷看答案,挑出來的當然好看。為驗證這懷疑,我們拉了一個從頭到尾沒碰過的獨立驗證集去重測,結果立刻現形、從 0.650 變成 0.710,中間約 0.07 公尺的落差就是赤裸裸的選擇偏差。於是我們做了一個決定:拒絕把 0.650 當成績,把誠實的 0.752 當標題,那是規規矩矩 full-train 一次、test 一次的結果。接著往最嚴格走,就是上一張那套無洩漏巢狀五折,得到 0.94 加減 0.04。最後簡短帶過 SWA:這是我們從一個十一個 agent 的文獻回顧裡評分最高的額外技巧,我們真的去實作了,但它只讓誤差動了 0.025,比 bootstrap 估出來的標準誤 0.037 還小。先把 bootstrap SE 用白話講掉:就是用重抽樣去估這個數字本身有多少不確定度;0.025 的改善小於 0.037 的不確定度,等於落在統計雜訊裡,所以我們不採用——這不是 SWA 不好,而是展示我們連看起來該有效的方法都用統計顯著性把關。我最期待被問:那你們不會覺得自爆很可惜嗎?完全不會,抓出自己的過擬合、並用更嚴格的尺把真實表現釘回來,這本身就是這份報告最硬的成果,比多砍 0.1 公尺有價值得多。

---

## 30. Bigger and fancier consistently lost: at 1.4k samples, bias beats capacity
*Honest failures — the three most instructive ones (full list in notes)*

- Diffusion head 1.82 m: sampling noise on a deterministic task
- Cascade-big 1.7M params 0.93 m: capacity just overfits at this scale
- CNN floor-plan cross-attn 0.91 m: extra machinery, no extra signal
- (also: C-Mixup 0.94, A+B/CNN+cascade combo 0.89, 3-level+12k synth 0.80)

圖: `outputs/figures/failures.png`

口稿: 這張誠實攤開沒成功的實驗,我認為失敗的清單往往比成功的數字更能說明一個團隊真正學到了什麼。我把投影片收斂到三個最有教育意義的失敗、其餘留在 bullet 末尾跟口頭,免得六條密密麻麻讓人既讀不完又聽不進。畫面這張長條圖把每個失敗方法跟它的 Split A 中位數並排,綠色那根是冠軍 cascade 的 0.752 基準線,高過它的全是退步。三個主打:第一,最慘的擴散頭 1.82 公尺,失敗原因很本質——定位是確定性的回歸問題,而擴散模型是生成式、靠採樣引入隨機性,等於在一個有唯一答案的題目上硬塞雜訊,方向就錯了。第二,容量陷阱的代表 cascade-big,把參數衝到 170 萬卻只有 0.93、徹底過擬合,這正是上一張可靠度圖結論的反面佐證:瓶頸在覆蓋不在容量,所以加容量當然沒用。第三,用樓層平面圖做交叉注意力的 CNN,0.91,失敗原因是它加了一堆額外機制,但平面圖的幾何先驗在這個訊號主導的任務上沒有額外資訊可榨。其餘三個我口頭快速帶過:C-Mixup 這種樣本間插值增強 0.94;A+B 把兩個時段資料合起來訓練只有 0.89(圖上標 CNN+cascade combo)——更多 session 解的是跨時段的軸,而 A 切法壓的是同分布準度,軸對不上自然沒幫助;三層 cascade 再加到一萬二千筆合成資料是 0.80,多一層 gating 跟更多合成幾乎沒換到東西、卻把系統弄複雜,所以我們不採用。提醒一個對照細節:圖上是絕對公尺。一句話總結:在我們只有約 1449 筆單次掃描的規模下,正確的歸納偏置遠遠勝過模型容量,這跟整條爬坡贏的都是換表示、換輸出形式而非堆參數,完全一致。

---

## 31. The error splits cleanly: 27% solved at the label floor, 16% unsolvable
*27% at the 0.3 m AMCL floor; ~16% are symmetric-ambiguity outliers*

- 27% of predictions sit at the 0.3 m AMCL floor (label-limited, not improvable)
- ~16% are >2 m outliers from WiFi symmetric ambiguity (two places, same RSSI)
- Worst-10 cluster in the sparse upper region and mirror-symmetric spots
- No model can separate identical fingerprints — only new signals can

圖: `outputs/figures/cascade_worst10.png`

口稿: 把剩下的誤差解剖開,你會發現它分成非常乾淨、性質完全相反的兩塊,理解這兩塊就理解了這個系統的天花板在哪。第一塊是好消息:有 27% 的預測已經貼在 0.3 公尺的地面真值底線上。要說清楚這 0.3 是什麼——它是 AMCL 這套定位本身的噪音,也就是我們的標籤本身就只準到 0.3 公尺,所以這 27% 的點其實已經準到跟標籤一樣、沒有任何再進步空間。我用詞很小心:我說的是受標籤限制,而不是宣稱我們完美。第二塊是壞消息、也是這份工作真正無解的部分:大約 16% 是超過兩公尺的離群,根源是 WiFi 對稱混淆——地圖上兩個物理不同的地方訊號指紋幾乎一模一樣,就像前面 gating 那張看到的雙峰。畫面這張畫的就是最差的十個預測案例,請看它們落在哪裡:幾乎全部集中在稀疏的上半部、以及鏡像對稱的位置,正好同時呼應前面可靠度圖的稀疏結論跟 CDF 的長尾。最關鍵的判斷:這 16% 不是換一個更厲害的模型就能解的,因為輸入訊號本身在數學上就不可區分,任何模型看到一模一樣的指紋都只能猜,唯一出路是加入新的、能打破對稱的訊號源,這就直接帶到最後結論。預期會被問:那 27% 加 16% 中間那塊呢?中間那塊才是正常、可被一般改進影響的工作區間,也是我們這一路爬坡真正在優化的部分;兩端一端被標籤鎖死、一端被物理鎖死,所以真正的戰場其實是中間。

---

## 32. Live in the lab: ESP32 to 9 ms CPU inference to live heatmap
*[ Recorded demo video plays here ]*

- Real ESP32 streaming live scans; CPU-only inference ~9 ms/scan (no GPU)
- Live position + probability heatmap rendered on the floor plan
- 80-95% of live APs match the trained vocabulary (no drift)
- Precision standing still <0.1 m spread; accuracy vs ground truth 0.4-0.8 m

圖: `outputs/figures/demo_snapshot.png`  ▶ **影片占位 — 換成 live demo 錄影**

口稿: 這張是現場示範,我會直接播放錄好的影片,投影片中央這塊就是預留給影片的位置;那張 demo_snapshot 只是萬一現場影片播不出來時的靜態備援。影片內容是真的 ESP32 開發板在實驗室裡即時串流 WiFi 掃描進到我們的模型做推論,我逐一強調幾個關鍵數字,因為它們把離線成績搬到真實世界做了跨域驗證。第一,整個推論是純 CPU 跑的、每筆掃描約 9 毫秒,完全不需要 GPU——代表這套系統真的可以部署在便宜的邊緣裝置上即時運作,這是我最愛引用的一句,因為很多漂亮的研究數字一旦要上低成本硬體就垮了,我們沒有。第二,輸出是即時的位置點加機率熱圖、直接疊在樓層平面圖上,人走到哪熱區就跟到哪,讓抽象的機率分布一眼可懂。第三跟第四是兩個重要的真實環境驗證,而且我要把『精度』跟『準度』分開講清楚:即時掃到的 AP 有八成到九成五仍落在我們訓練過的 80 個 BSSID 詞彙表裡,代表幾個月下來環境沒大幅漂移、模型的輸入假設站得住;精度方面,當人站著不動時連續預測的抖動小於 0.1 公尺,這是 precision、講的是穩定性、不代表它離真值只有 0.1。準度方面才是離真值多近——我會在影片播放時口頭補一個現場比對:隨機點一個真實已知位置去看單點誤差,結果落在 0.4 到 0.8 公尺,這個 accuracy 跟我們離線報的 0.75 中位數是同一個量級,這一句話就把整份報告的可信度從紙上拉到了現場。預期會被問:漂移怎麼辦?定期用新掃描微調詞彙表跟編碼器即可,八到九成的重疊率代表這種維護成本很低。

---

## 33. Honest verdict: ~0.75 m is real, sub-0.65 m needs new data, not a deeper net
*Standard-protocol median 0.752 m; strict nested-CV 0.94 m*

- Halved KNN error to an honest 0.752 m median, validated on real hardware
- Won by inductive bias (set input, grid classification, coarse gate), not capacity
- Honesty is the result: exposed a 0.650 m mirage, reported 0.94 m nested-CV
- To break 0.6x: upper-region data, IMU/heading, temporal fusion — not depth

圖: `outputs/figures/ladder_bar.png`

口稿: 總結這整份報告。最直接的成果:我們把最樸素 KNN 基準的 1.57 公尺砍了一半,得到一個誠實的 0.752 公尺中位數(標準協定);在我們自己加碼的最嚴格巢狀交叉驗證下是 0.94——這兩個口徑我從第一張投影片用到最後一張、完全一致沒有矛盾,這點希望老師特別認可。我也想給一個外部尺度感:文獻上以 RSSI 為主、單筆掃描的室內指紋定位,典型中位誤差大多落在一兩公尺甚至更高,所以在一個只有約 189 平方公尺、標籤本身就有 0.3 公尺噪音的場域裡做到 0.75,是把誤差壓到接近標籤精度、屬於很有競爭力的結果。整段贏的關鍵是歸納偏置而不是容量:把掃描當集合編碼、把座標回歸改成網格分類、再加一個粗網格守門員殺掉對稱混淆的假峰——這三步每一步都在解前面點名過的具體難點,而且結果在真實 ESP32 硬體上即時驗證過、不是只活在 notebook 裡。但我最想被記住的是把誠實本身當成核心成果:我們主動抓出自己 0.650 的海市蜃樓,也誠實報出 0.94 的巢狀交叉驗證,寧可報一個比較高但站得住的數字,也不要一個漂亮但偷看過答案的數字。最後未來方向講實在的:真正的 0.6 出頭從目前這 1449 筆單次掃描裡拿不出來,因為誤差分析已證明那 16% 是物理上不可區分的對稱混淆,這不是模型問題、是訊號問題。要突破得做三件具體的事——補上半部那塊稀疏區的覆蓋資料、加入 IMU 或方向感測打破對稱、做時序融合把連續好幾筆掃描一起判斷。一句話收尾,也是整份報告的精神:下一步的增益在新的訊號,不在更深的網路。謝謝大家,歡迎提問。

---
