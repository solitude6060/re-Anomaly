SMT產線瑕疵檢測難題研究與新模型規劃
背景與問題陳述
SMT（表面黏著技術）製造產線上的零件瑕疵檢測是一項具有挑戰性的任務。傳統上很難取得大量的不良樣本 (NG) 圖片作為監督訓練資料，因此我們現行採用**異常檢測 (Anomaly Detection)**的方式，只以正常樣本來學習模型以偵測瑕疵。當前方案利用 FastFlow 結合 DINOv3 視覺骨幹模型，每種零件的每個腳位都獨立訓練一個模型進行檢測。然而，我們面臨以下困境：
資料極度匱乏：有時候實際不良 (瑕疵) 圖片幾乎沒有，甚至連正常 (OK) 圖片都僅有10～20張，導致模型難以學到可靠的正常分佈。
微小瑕疵難以檢出：目前系統對於非常小的瑕疵（例如細微刮痕、細小的焊點缺陷等）靈敏度不足，常常漏檢這類細微異常。
結構性/邏輯性異常漏失：某些瑕疵必須出現在特定位置或以特定結構才算異常（例如零件某處缺損或錯位），但現有模型對此類結構邏輯瑕疵辨識不佳，容易將其視為正常變異而漏檢。
下游模型訓練不穩定：在資料很少的情況下進行FastFlow下游任務訓練，模型收斂和表現不穩定，不同次訓練結果波動較大。
現有預訓練提升有限：我們擁有三套工業影像資料集（數百萬張），其中兩套為SMT場景（分佈相對集中），另一套較多樣。利用這些無標籤資料進行DINOv3骨幹預訓練後，雖然有些許提升，但仍然無法充分解決上述問題，對模型性能的改善可說是
。
嚴苛的產線要求：製造產線要求0漏失（不允許漏檢任何瑕疵），同時誤報率 (overkill)需控制在10%以內，目標直通率希望達到80%。也就是說，在確保不放過任何不良的前提下，對正常產品的誤判不能過高。這對模型的準確性和可靠度提出了極高要求。
系統資源限制：產線部署環境對推理速度和模型大小也有要求（需越快越好，模型體積有限）。同時，我們的影像解析度不高（約200×200），有時還會遇到因壓縮造成解析度異常偏低的圖片，需要模型對此具有一定魯棒性。
上述問題使得目前的FastFlow+DINO方案難以滿足生產需求。我們需要對最新技術進行調研，並據此設計一套新的瑕疵檢測模型方案，以解決小樣本異常檢測、微小瑕疵辨識和結構性缺陷識別的難題，滿足產線的高標準要求。
技術調研：工業異常檢測最新進展
為了解決上述挑戰，我們調研了近年來在工業瑕疵/異常檢測領域的相關研究，包括無監督異常檢測、一級（one-class）學習以及少樣本異常偵測等方法。以下是調研重點發現：
預訓練視覺模型的應用： 大型基礎模型 (Foundation Models)如DINO系列自監督ViT，已被證明能提供強大的圖像特徵，有助於異常檢測。DINO預訓練模型提取的特徵具有局部與全局資訊，對影像不同視角和裁剪具有魯棒性，並且受益於大規模資料學習
。研究顯示這類預訓練ViT特徵非常適合用於異常偵測，因為它們能捕捉產品外觀的細節和整體結構，同時對非關鍵變化不敏感
。我們目前採用的DINOv3骨幹即屬此類，未來可考慮使用最新的DINOv2/v3或類似的大模型特徵以提升效果。
多尺度特徵與瑕疵大小： 工業瑕疵可能大小不一，多尺度特徵對於同時偵測微小缺陷與大型結構異常非常重要
。深度模型不同層的特徵具有金字塔式的尺度性質：低層（高解析度）特徵對細微紋理變化敏感，有助檢出細小刮痕等微瑕疵；高層（抽象語意）特徵擅長表示整體結構，可用於識別缺件、錯位等大範圍異常
。例如，PatchCore等方法就利用了深度模型的多層特徵來同時兼顧精細缺陷與全局異常
。我們目前在DINO中已採多尺度特徵融合，但未來可進一步強化這方面的設計。
瑕疵分佈建模方法： 近年的異常檢測方法大多在預訓練特徵空間建模正常樣本分佈，以判定離群的異常。主要方法包括：
最近鄰特徵匹配 (kNN) 與記憶庫： 代表如PatchCore
。PatchCore使用ImageNet預訓練模型的patch特徵，建立正常樣本的記憶庫，推理時計算每個圖像區塊與記憶庫中最近鄰的距離作為異常分數。此方法無需額外訓練且對少量樣本有效，並在MVTec工業瑕疵基準上取得了99.6%的影像級檢測AUC，將前人錯誤率減半
。它在少樣本情境下表現也相當競爭
。這類深度最近鄰方法的成功證明，只需少量正常樣本也能透過強大的預訓練特徵取得高精度異常檢測
。
機率分佈建模： 利用統計模型擬合正常特徵分佈，例如PaDiM（以多元高斯分佈近似每個特徵位置的正常變異）或Mahalanobis距離等
。這類方法計算正常樣本特徵的均值和協方差，推理時以馬氏距離衡量新樣本偏離程度。它們計算簡單、資料需求低，在少量樣本時比訓練深網路更穩定。不過線性高斯假設可能限制了複雜度。
正規化流模型 (Normalizing Flows)： 例如我們現用的FastFlow模型
。Flow通過可逆變換將正常特徵映射到高斯等可控基底分佈，透過對數似然衡量樣本屬於正常分佈的機率
。FastFlow提出在二维空間對特徵圖做flow建模，可同時考慮局部與全局特徵關係
。研究顯示FastFlow搭配ResNet或ViT骨幹在MVTec中達到99.4%的AUC，且推理速度快
。Flow方法能提供端到端的機率評分，但在極少樣本時參數估計可能不穩定，需要注意避免過度擬合正常樣本。
教師-學生知識蒸餾： 此類無監督方法利用一個預訓練模型作為教師，訓練一個結構相同的學生模型去模仿教師對正常樣本的特徵表示。代表如STPM (Student-Teacher Feature Pyramid Matching)
。透過讓學生網路在正常樣本上逼近教師的多層特徵，學生將學到正常模式的表示。一旦輸入有異常，學生的特徵輸出將無法精確匹配教師，兩者差異即可定位異常區域
。特別地，STPM融合了多尺度特徵金字塔進行監督，使其能偵測各種尺寸的異常，從小瑕疵到大缺陷皆有良好表現
。此方法在MVTec上也取得優異結果，且能產生像素級的異常熱點圖，方便定位。
合成瑕疵生成與對比學習： 一些方法透過生成擬真異常來輔助訓練，從而在僅有正常資料時提升模型辨識能力。其中DRÆM就是一個成功案例
。它在訓練時對正常圖像隨機疊加合成的異常圖案（例如Perlin noise形狀的斑點、刮痕等）來模擬瑕疵，同時訓練一個重建網路去還原正常圖像以及一個判別網路去辨識異常區域
。透過這種重建+分割聯合學習，模型學得正常與異常的決界
。DRÆM在MVTec資料集上將無監督SOTA性能大幅推進，甚至在DAGM瑕疵資料上接近全監督方法的檢測水準，同時在定位精度上顯著優於全監督方法
。重要的是，DRÆM僅用模擬異常訓練，就能適應多樣的真實瑕疵型態
。這證明了透過合理的合成數據增強，可以在零不良樣本的情況下培養模型對細微和結構異常的敏感度。
少樣本/零樣本異常檢測： 近來有研究專注於few-shot場景下的異常檢測新方案，假設只有極少正常樣本甚至不需訓練即可運作。例如：
PatchCore改進與擴充： 最新研究如 AnomalyDINO
將PatchCore框架與最新的DINOv2特徵結合，並融入多尺度增強與穩健統計方法，進一步提升少樣本下的準確性和穩定性
。DINOv2提供的高品質patch特徵讓方法在維持訓練自由（無需額外微調）的同時，增強了對異常的敏感度。
基礎模型+非線性嵌入投影： 一項2025年的研究 (FoundAD) 發現大型預訓練基礎視覺編碼器本身蘊含著少樣本異常偵測能力
。他們提出對預訓練模型的嵌入空間學習一個非線性投影，將影像投射回自然影像流形上，再以嵌入差異衡量異常程度
。這種方法能有效偵測結構性異常，且支援多類別場景，同時模型非常輕量快速
。實驗顯示，不論使用DINOv3等最新ViT或其他基礎模型，此方法在few-shot下都取得優異表現
。
多解析度檢測： 工業異常在尺度上差異巨大，最新有學者提出多解析度融合策略來提升零樣本/少樣本檢測能力。例如一項ICLR2026的工作指出，低解析度輸入可提供全局語境來判斷異常是否存在，高解析度輸入則細化異常區域的邊界
。他們的MRF-AD (Multi-Resolution Fusion)方法在推理時對圖像做多種縮放，經預訓練模型抽取多尺度特徵後融合。結果顯示此無需訓練的簡單策略在MVTec AD等基準上達到SOTA級別，證明多解析度輸入對同時捕捉宏觀結構異常與微觀細節瑕疵的有效性
。
統一多類別模型： 傳統上每類產品需單獨一個異常檢測模型，但也有研究致力於單一模型處理多種零件正常樣本的Unified AD。例如Hierarchical Gaussian Mixture Model結合Flow的方法，試圖用分層高斯混合學習多類正常分佈以避免不同類別間特徵「同質映射」問題
。這種方法對我們未來若要擴展到統一模型檢測多種零件瑕疵有一定參考價值。不過，目前統一模型在各類別精度上可能略遜於單類模型，需要足夠資料和巧妙建模來保持各類別的區分度
。
綜上所述，最新研究趨勢對我們問題的啟發包括：利用強大的預訓練特徵、多尺度（特徵與解析度）處理來兼顧微缺陷與結構異常、記憶庫或流形投影等輕量模型適合少樣本情境，以及合成數據增強可在缺乏不良樣本時提升模型對異常模式的學習。這些技術要點將指引我們新模型的設計。
產品需求與目標 (PRD)
根據上述背景與調研洞見，我們確立本瑕疵檢測系統的新方案需滿足以下產品需求與目標：
檢測性能：在不洩漏任何瑕疵 (0漏失)的前提下，將誤報率控制在10%以內，達成約80%的良品直通率。換言之，模型對微小瑕疵和特殊位置/結構異常必須100%檢出，且對正常樣本的誤判率不超過一成。
少樣本學習：能在每種零件僅有10～200張正常樣本、無不良樣本的條件下有效訓練或調適。模型應充分利用預訓練知識和無監督方法克服資料匱乏，同時避免過度依賴大數據。
定位解釋能力：不僅判斷零件是否異常，還應給出異常區域的定位或熱點圖，方便人工確認與後續分析。尤其對於細微異常或結構錯誤，需要提供直觀的可視化結果（如在圖像中標示出瑕疵位置）。
多場景魯棒性：能處理影像在解析度、壓縮質量上的差異。例如對於解析度低於預期或有壓縮失真的圖片，模型依然能產生可靠的特徵和檢測結果。亦即模型對輸入尺寸和畫質具有一定適應性。
模型效率與部署：模型大小和推理速度需符合產線實時檢測要求。初步目標是模型參數量不超過數千萬級（方便部署在GPU或邊緣裝置上），單張ROI推理時間在幾十毫秒級以內（具體依硬體而定），以保證不成為產線瓶頸。必要時可通過模型剪枝、量化或輕量骨幹網路來優化部署效率。
可擴展性：模型應具有可延展的設計，以便未來：
能快速適配新的零件類型（透過少量新樣本調整模型記憶庫或經過簡易微調，而不必推倒重訓整個模型）。
可以融合多類別的學習（未來若希望一個模型支持多種產品檢測，可通過調整架構或增加分類標識等方式擴展）。
可靈活納入新的模組（如加入合成瑕疵增強或第二階段分類器來進一步降低誤報）。
新模型方案設計規格
針對上述需求，我們規劃開發一套改良的異常檢測模型，結合多尺度特徵提取與輕量異常分佈建模技術。以下是新模型的架構與功能規格：
1. 模型架構概述
新模型採用**「預訓練骨幹 + 多尺度特徵 + 正常分佈表示 + 異常判斷」**的模組化架構：
骨幹網路：使用預訓練的視覺Transformers（如我們現有的DINOv3骨幹）作為特徵提取器。考慮到推理效率與模型大小，可選擇適當規模的ViT（如ViT-S/16或ViT-B/16）並固定其權重（或僅輕微微調最後幾層）。該骨幹已在大規模資料上學到強健的特徵表示
。我們也可以將之前工業資料集的預訓練權重繼續沿用，以保留對SMT領域特性的學習。
多尺度特徵提取：骨幹網路將對輸入圖像提取出多層次的特徵圖。我們將攫取不同層級（例如Transformer第4層、第8層、第12層）的輸出作為多尺度特徵金字塔。低層特徵圖具有高解析度（接近輸入圖像大小），對微小紋理差異敏感，可檢測細微瑕疵；高層特徵維度抽象，感受野大，適合辨識結構性異常
。我們會對各尺度特徵適當進行上/下採樣對齊尺寸，然後融合這些特徵以供後續異常判斷使用。融合方式可採簡單的層疊(channel concatenate)或在每層分別評估異常再彙總。該多尺度設計預期提升模型對不同大小異常的靈敏度
。
多解析度輸入 (可選強化)：為進一步提高對尺度極端差異異常的檢出率，可引入多解析度輸入策略：對同一ROI圖像，同時輸入原始尺寸以及縮放到更低解析度的一版到骨幹網路，分別提取特徵再融合。不同比例輸入讓骨幹以不同視角看圖
——低解析度圖像讓模型更容易捕捉全局異常（如整個零件位置偏移或缺損），高解析度圖像保留細節紋理便於發現極小缺陷
。融合不同解析度的特徵（金字塔與輸入雙重多尺度）將使異常檢測更加全面。由於DINOv3等ViT對輸入尺寸有彈性（可插值位置嵌入），我們可以在不增加骨幹參數的情況下實現該功能。
正常特徵分佈建模：在得到融合特徵後，核心是建模正常樣本的特徵分佈，以區分偏離此分佈的異常。鑒於我們僅有正常資料且樣本極少，新模型採用輕量無監督的方法：
記憶庫最近鄰 (Memory Bank + kNN)：我們將從每張正常樣本的多尺度特徵中抽取大量patch級特徵向量，全部存入正常特徵記憶庫。為控制記憶庫規模，可對特徵進行核心集採樣 (coreset)以移除冗餘
。推理時，對待測影像提取同樣的patch特徵，計算每個patch特徵與記憶庫中最近的鄰居距離作為異常度量
。正常圖像的patch應在記憶庫找到相近的對應，因此距離小；而異常區域的patch特徵將無法在正常庫中找到近似匹配，距離顯著較大。此距離圖可直接轉換成異常熱度圖以定位瑕疵區域，最大距離或其統計量則作為該圖的異常分數。
正規化流/高斯模型 (備選)：在內部實驗中，我們也可比較使用正規化流來擬合正常特徵分佈。即針對融合特徵空間訓練FastFlow模型，學習將正常特徵映射為高斯分佈
。推理時計算每個patch特徵的對數似然作為異常度（似然低則異常）。Flow方法理論上能捕捉特徵間複雜相關性，不過在我們極少樣本情境下要注意訓練穩定性。因此，默認方案以記憶庫+kNN方法為主，Flow作為可選替代/輔助，用於對比評估性能。
非線性特徵投影 (進階備選)：未來亦可嘗試實現FoundAD類似的非線性投影模組
。即訓練一個小型MLP，將輸入影像的深度特徵投影回正常特徵流形，再衡量投影前後差異來判斷異常
。這相當於學習一個自適應的特徵自編碼器，只需要正常樣本即可訓練。該方法報導對結構性異常尤為敏感，且模型參數很少
。若我們發現記憶庫法對某些特殊異常判別仍有不足，可將此投影模組串接於骨幹之後作為改進。
異常判決與後處理：模型將輸出每張待測ROI的異常分數以及對應的異常熱力圖。我們將設定一個判決閾值：異常分數高於閾值則判為瑕疵NG，低於則為OK良品。為滿足產線誤報率限制，我們會利用驗證集正常樣本來調整閾值，使得正常樣本的誤報率約在10%以內。同時，我們採取以下後處理策略提高可靠性：
噪聲濾除：對異常熱力圖採用小型形態學過濾或連通域分析，去除零星散點式的高分（可能來自感測雜訊或無害輕微變化），保留成片連續的異常區域，以減少偽警報。
結構規則約束：針對已知特殊邏輯異常（例如必須在特定位置出現才算異常的情況），可在判決時加入區域遮罩或規則校驗。比如：如果某瑕疵只有出現在零件邊緣才危險，則對熱力圖中邊緣區域給予更高權重，非邊緣區降低權重，以符合人員檢驗標準。這確保模型判斷與實際產品判定邏輯一致，降低誤判和漏判。
分數融合 (多尺度/多模型)：若採用多解析度輸入或多種檢測頭（記憶庫+Flow雙路），我們將對多路結果取最大異常響應或使用簡單規則融合，確保任何一種機制檢出的異常都不被忽略。同時若多路都報告異常則提高置信度。這種融合有助於進一步趨近0漏失並抑制偶發性誤報
。
**模型大小與效率：**整體架構設計考慮了效率：骨幹ViT固定參數且可根據需要選小型版本；記憶庫查找屬於輕量的非參數化運算（可用Ball Tree、FAISS等加速kNN）；無需大型反向傳播訓練迴圈，只有輕量模組可能需要少量訓練（例如FoundAD投影MLP，參數量極少）。因此部署時推理計算量主要在ViT特徵提取，其餘部分計算開銷低且易優化。若需進一步加速，我們可以預先提取正常庫特徵離線存儲，加速最近鄰查找；此外也可考慮裁剪ViT層數或使用蒸餾技術得到精簡模型版本，以滿足特定硬體的實時性要求。
2. 模型功能清單
根據架構設計，新模型將具備以下功能特性，以滿足產品需求：
**少樣本學習能力：**只需10～200張正常樣本即可建構模型的正常特徵分佈。無需任何不良瑕疵數據，亦不依賴大量追加資料，即可完成模型部署（支援後續增量地加入新正常樣本更新記憶庫來提升模型穩健性）。
無監督異常檢測：模型在推理時自動判別輸入影像是否偏離正常範圍。如有異常則報警，無異常則通過，整個過程不需要人工標記介入。
多尺度瑕疵檢出：利用深度特徵金字塔與多解析度處理，模型對各種尺寸的瑕疵均具備高靈敏度。無論是細如髮絲的小刮痕還是整個零件缺件/移位這樣的宏觀異常，都能被偵測
。這將顯著改善當前“小瑕疵抓不到、大結構漏判”的問題。
異常區域定位：模型輸出直觀的異常熱力圖，高亮出疑似瑕疵所在的位置及範圍。檢測人員可據此快速確認瑕疵內容，輔助排除誤報。同時該熱力圖也可用於後處理，如聚類連通區域得到精確的瑕疵邊界。
零件結構邏輯融合：針對需考慮零件結構/位置判定的異常，模型允許設定關注區域或關鍵點。可對特定圖像區域的異常分數設定不同閾值，或基於多個區域的異常模式觸發複合判定邏輯，滿足複雜檢驗規範。
**動態閾值調整：**提供靈活的異常分數閾值設定接口。使用者可根據實際品質策略（例如放寬或嚴格）調整靈敏度。閾值調整後，模型的0漏失/誤報率權衡可即時更新，以適應不同生產批次或客戶要求。
**高效推理與部署：**模型在標準GPU上可實現實時推理，每秒可處理多張200×200 ROI影像（實際速度將根據硬體環境調優）。模型大小精簡，所有參數存儲和計算能在邊緣設備（如工業PC或Jetson裝置）上流暢運行。如需在CPU上運行，也可透過縮減ViT尺寸或採用量化優化得到可接受的速度。整體架構模組清晰，可方便地整合進現有產線軟體流程。
**易於維護擴充：**由於採用了記憶庫和預訓練模型，當產品設計稍有變動（導致外觀改變）或新增新零件型號時，只需收集少量新正常樣本更新記憶庫或增加對應類別，即可快速適配，無需從頭訓練龐大模型。模型亦支持後續融入其他資料源（如不同光源下的圖像）進行特徵庫融合，以提高環境魯棒性。
3. 模型開發與測試計畫
資料準備與增強：首先收集並標註現有正常樣本，進行資料清洗。對於只有10張左右的類別，透過圖像增強（旋轉、鏡像、光強變化等）適當增加擬增資料至數十張，以豐富正常變異。若圖片出現非預期壓縮失真，考慮在增強中加入JPEG壓縮模擬，以提升模型對壓縮artifact的適應性。
**模型實現與離線訓練：**根據上述架構搭建模型原型。利用我們的工業預訓練骨幹提取正常樣本特徵，建立每類別的正常特徵記憶庫。若採用Flow或投影模型，使用正常樣本進行訓練/微調。在此過程中調整多尺度融合策略和記憶庫採樣策略，確保模型在訓練集上對正常樣本幾乎零誤報且特徵分佈學習穩定。**注意：**由於無不良樣本，我們會以部分正常樣本作為驗證，用於防止模型過適應訓練集（例如對記憶庫過度壓縮導致正常樣本反而被判異常）。
離線評估 (Internal Test)：在內部測試集上評估新模型性能。測試集包含若干實際標註的瑕疵樣本和更多正常樣本。我們將計算以下指標：
漏檢率（漏失率）：期望實現0漏檢，即測試集中所有瑕疵樣本皆成功檢出。
誤報率：統計正常測試樣本中被誤判為異常的比例，調整模型閾值使其不超過10%。
圖片級檢測AUC：評估模型區分正常/異常的整體曲線下方積分，參考與前述FastFlow基線比較性能提升幅度
。
像素級定位精度：如果有像素級瑕疵標註，計算IoU或PRO指標，確定熱力圖定位合理。
推理時間與資源占用：在實際推理環境測量每張ROI處理時間，確保達到實時要求。同時監控記憶體占用與模型大小，確認符合部署限制。
**閾值標定：**根據內部測試結果調整異常分數閾值。尤其關注在保持0漏失時，誤報率是否滿足要求。如發現誤報略高，分析誤判類型，可能通過:
加強後處理規則（如忽略某些無影響區域的異常信號）。
擴充正常樣本庫（收集更多正常變化類型）。
或適度降低模型靈敏度（在能容忍極少漏檢的情況下權衡，但根據需求0漏失應為硬性目標，不輕易妥協）。
**與現有方案對比：**將新模型與當前FastFlow+DINO方案在相同測試集上對比：
微小瑕疵檢出率提升情況（預期大幅改善此前漏檢的細小缺陷）。
特殊結構異常（如錯位/缺件）檢出情況改善。
模型輸出穩定性（多次重訓結果的變異減小）。
推理速度及資源開銷差異。如果新模型稍有增加但在可接受範圍，則以精度優先。
產線試運行：在實驗室測試達標後，部署模型至產線進行試運行（旁路檢測）。收集一段時間內模型對實際產線圖片的判斷結果，與人工檢驗結果比對，記錄任何漏檢或誤報案例以便分析調整。尤其觀察模型對未知新型瑕疵的反應，驗證其泛化能力。根據現場反饋進一步優化模型閾值或規則。
經過上述開發與測試流程，我們期望新模型能夠全面解決現有瑕疵檢測難題。在內測中，如果新方案實現如預期的0漏失和低誤報，將推進其作為生產線主力檢測系統，逐步替換舊有方案，並持續監控模型表現以進行持續改進。
參考文獻與調研來源
Roth et al. “Towards Total Recall in Industrial Anomaly Detection (PatchCore).” CVPR 2022.
Yu et al. “FastFlow: Unsupervised Anomaly Detection and Localization via 2D Normalizing Flows.” arXiv 2021.
Wang et al. “Student-Teacher Feature Pyramid Matching for Anomaly Detection.” BMVC 2021.
Zavrtanik et al. “DRÆM: A Discriminatively Trained Reconstruction Embedding for Surface Anomaly Detection.” ICCV 2021.
Damm et al. “AnomalyDINO: Boosting Patch-Based Few-Shot Anomaly Detection with DINOv2.” WACV 2025.
Shen et al. “Foundation Visual Encoders Are Secretly Few-Shot Anomaly Detectors (FoundAD).” ICLR 2026 (submitted).
Zou et al. “Multi-Resolution Fusion: An Effective Approach to Anomaly Detection.” ICLR 2026 (withdrawn).
Yao et al. “Hierarchical Gaussian Mixture Normalizing Flow Modeling for Unified Anomaly Detection.” ECCV 2024.
Coenen et al. “PADiM: A Patch Distribution Modeling Framework for Anomaly Detection.” CVPR 2020.
 (提及)
Bergmann et al. “MVTEC AD: A Comprehensive Real-World Dataset for Unsupervised Anomaly Detection.” CVPR 2019. (提供本研究中引用的基準數據集)
引用

DRAEM - A Discriminatively Trained Reconstruction Embedding for Surface Anomaly Detection

https://openaccess.thecvf.com/content/ICCV2021/papers/Zavrtanik_DRAEM_-_A_Discriminatively_Trained_Reconstruction_Embedding_for_Surface_Anomaly_ICCV_2021_paper.pdf

AnomalyDINO: Boosting Patch-Based Few-Shot Anomaly Detection with DINOv2

https://openaccess.thecvf.com/content/WACV2025/papers/Damm_AnomalyDINO_Boosting_Patch-Based_Few-Shot_Anomaly_Detection_with_DINOv2_WACV_2025_paper.pdf

Towards Total Recall in Industrial Anomaly Detection

https://openaccess.thecvf.com/content/CVPR2022/papers/Roth_Towards_Total_Recall_in_Industrial_Anomaly_Detection_CVPR_2022_paper.pdf

Towards Total Recall in Industrial Anomaly Detection

https://openaccess.thecvf.com/content/CVPR2022/papers/Roth_Towards_Total_Recall_in_Industrial_Anomaly_Detection_CVPR_2022_paper.pdf

Towards Total Recall in Industrial Anomaly Detection

https://openaccess.thecvf.com/content/CVPR2022/papers/Roth_Towards_Total_Recall_in_Industrial_Anomaly_Detection_CVPR_2022_paper.pdf

Towards Total Recall in Industrial Anomaly Detection

https://openaccess.thecvf.com/content/CVPR2022/papers/Roth_Towards_Total_Recall_in_Industrial_Anomaly_Detection_CVPR_2022_paper.pdf

AnomalyDINO: Boosting Patch-Based Few-Shot Anomaly Detection with DINOv2

https://openaccess.thecvf.com/content/WACV2025/papers/Damm_AnomalyDINO_Boosting_Patch-Based_Few-Shot_Anomaly_Detection_with_DINOv2_WACV_2025_paper.pdf

[2111.07677] FastFlow: Unsupervised Anomaly Detection and Localization via 2D Normalizing Flows

https://arxiv.org/abs/2111.07677

[2111.07677] FastFlow: Unsupervised Anomaly Detection and Localization via 2D Normalizing Flows

https://arxiv.org/abs/2111.07677

[2111.07677] FastFlow: Unsupervised Anomaly Detection and Localization via 2D Normalizing Flows

https://arxiv.org/abs/2111.07677

[2103.04257] Student-Teacher Feature Pyramid Matching for Anomaly Detection

https://arxiv.org/abs/2103.04257

DRAEM - A Discriminatively Trained Reconstruction Embedding for Surface Anomaly Detection

https://openaccess.thecvf.com/content/ICCV2021/papers/Zavrtanik_DRAEM_-_A_Discriminatively_Trained_Reconstruction_Embedding_for_Surface_Anomaly_ICCV_2021_paper.pdf

DRAEM - A Discriminatively Trained Reconstruction Embedding for Surface Anomaly Detection

https://openaccess.thecvf.com/content/ICCV2021/papers/Zavrtanik_DRAEM_-_A_Discriminatively_Trained_Reconstruction_Embedding_for_Surface_Anomaly_ICCV_2021_paper.pdf

DRAEM - A Discriminatively Trained Reconstruction Embedding for Surface Anomaly Detection

https://openaccess.thecvf.com/content/ICCV2021/papers/Zavrtanik_DRAEM_-_A_Discriminatively_Trained_Reconstruction_Embedding_for_Surface_Anomaly_ICCV_2021_paper.pdf

AnomalyDINO: Boosting Patch-Based Few-Shot Anomaly Detection with DINOv2

https://openaccess.thecvf.com/content/WACV2025/papers/Damm_AnomalyDINO_Boosting_Patch-Based_Few-Shot_Anomaly_Detection_with_DINOv2_WACV_2025_paper.pdf

AnomalyDINO: Boosting Patch-Based Few-Shot Anomaly Detection with DINOv2

https://openaccess.thecvf.com/content/WACV2025/papers/Damm_AnomalyDINO_Boosting_Patch-Based_Few-Shot_Anomaly_Detection_with_DINOv2_WACV_2025_paper.pdf

AnomalyDINO: Boosting Patch-Based Few-Shot Anomaly Detection with DINOv2

https://openaccess.thecvf.com/content/WACV2025/papers/Damm_AnomalyDINO_Boosting_Patch-Based_Few-Shot_Anomaly_Detection_with_DINOv2_WACV_2025_paper.pdf
Foundation Visual Encoders Are Secretly Few-Shot Anomaly Detectors | OpenReview

https://openreview.net/forum?id=YRrlJ8oVEH
Foundation Visual Encoders Are Secretly Few-Shot Anomaly Detectors | OpenReview

https://openreview.net/forum?id=YRrlJ8oVEH
Foundation Visual Encoders Are Secretly Few-Shot Anomaly Detectors | OpenReview

https://openreview.net/forum?id=YRrlJ8oVEH
Foundation Visual Encoders Are Secretly Few-Shot Anomaly Detectors | OpenReview

https://openreview.net/forum?id=YRrlJ8oVEH
Multi-Resolution Fusion: An Effective Approach to Anomaly Detection | OpenReview

https://openreview.net/forum?id=Mg5S5Twrfn
Multi-Resolution Fusion: An Effective Approach to Anomaly Detection | OpenReview

https://openreview.net/forum?id=Mg5S5Twrfn
Multi-Resolution Fusion: An Effective Approach to Anomaly Detection | OpenReview

https://openreview.net/forum?id=Mg5S5Twrfn
https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/04634.pdf
https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/04634.pdf
https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/04634.pdf

Towards Total Recall in Industrial Anomaly Detection

https://openaccess.thecvf.com/content/CVPR2022/papers/Roth_Towards_Total_Recall_in_Industrial_Anomaly_Detection_CVPR_2022_paper.pdf
所有資料來源

openaccess.thecvf

arxiv
openreview
ecva