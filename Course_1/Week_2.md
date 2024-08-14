# 2. Week 2: Sentiment Analysis with Naive Bayes

## 2.1. **Probability and Bayes’ Rule**

Tưởng tượng rằng bạn có một corpus of tweets có thể phân loại giữa positive và negative. Để tích xác suất của Positive là đếm số lần xuất hiện như ở dưới đây

<p align="center">
  <img src="https://github.com/nvsthinh/NLP_Coursera/blob/main/Course_1/data/W210.png" width="500"/>
</p>

Tới với ví dụ tiếp theo, với dòng tweets có chứa “happy” trong câu

<p align="center">
  <img src="https://github.com/nvsthinh/NLP_Coursera/blob/main/Course_1/data/W211.png" width="500"/>
</p>

## 2.2. Baye’s Rule

Conditional probabilities giúp chúng ta giảm không giam tìm kiếm mẫu. Với ví dụ giống như trên:

<p align="center">
  <img src="https://github.com/nvsthinh/NLP_Coursera/blob/main/Course_1/data/W220.png" width="500"/>
</p>

Với công thức như này, thì việc tìm kiếm $P("happy")$ và $P(Positive\cap"happy")$ dễ dàng hơn là tính $P(Positive|"happy")$

Với công thức của Conditional Probability. Với hai trường hợp $P(Positive|"happy")$ và $P("happy"|Positive)$. Sau đó, chúng ta chỉ tìm kiếm trong vòng tròn màu xanh ở trên. Tử số sẽ là phần màu đỏ và mẫu số sẽ là phần màu xanh. Điều này dẫn chúng ta đến kết luận sau:

<p align="center">
  <img src="https://github.com/nvsthinh/NLP_Coursera/blob/main/Course_1/data/W221.png" width="500"/>
</p>

Thay tử số vào vế phải của phương trình đầu tiên, chúng tasẽ nhận được kết quả sau:

<p align="center">
  <img src="https://github.com/nvsthinh/NLP_Coursera/blob/main/Course_1/data/W222.png" width="500"/>
</p>

**→ Bayes Rule**

## 2.3. Naive Bayes Introduction

Để xây dựng bộ phân loại, trước tiên chúng ta sẽ bắt đầu bằng cách tạo các xác suất có điều kiện theo bảng sau:

<p align="center">
  <img src="https://github.com/nvsthinh/NLP_Coursera/blob/main/Course_1/data/W230.png" width="500"/>
</p>

Điều này cho phép chúng ta tính toán bảng xác suất sau:

<p align="center">
  <img src="https://github.com/nvsthinh/NLP_Coursera/blob/main/Course_1/data/W231.png" width="500"/>
</p>

Khi bạn đã có xác suất, bạn có thể tính likelihood score như sau:

<p align="center">
  <img src="https://github.com/nvsthinh/NLP_Coursera/blob/main/Course_1/data/W232.png" width="500"/>
</p>

Điểm lớn hơn 1 biểu thị lớp đó là tích cực, ngược lại là tiêu cực

## 2.4. Laplacian Smoothing

Chúng tôi thường tính xác suất của một từ cho một lớp như sau:

$$
P(w_i \mid \text{class}) = \frac{\text{freq}(w_i, \text{class})}{N_{\text{class}}} \quad \text{class} \in \{ \text{Positive}, \text{Negative} \}
$$

Tuy nhiên, nếu một từ không xuất hiện trong quá trình đào tạo, thì nó sẽ tự động nhận được xác suất là 0, để khắc phục điều này, chúng tôi thêm smoothing như sau

$$
P(w_i \mid \text{class}) = \frac{\text{freq}(w_i, \text{class}) + 1}{N_{\text{class}} + V}
$$

Lưu ý rằng chúng ta đã thêm $1$ vào tử số và vì có $V$ từ cần chuẩn hóa, nên chúng ta thêm $V$ vò mẫu số.

$𝑁_{𝑐𝑙𝑎𝑠𝑠}$: tần suất của tất cả các từ trong lớp

$V$: số lượng từ duy nhất trong vocabulary

## 2.5. **Log Likelihood**

Để tính log likelihood, chúng ta cần lấy các tỷ lệ và sử dụng chúng để tính score cho phép chúng ta quyết định xem một tweet là tích cực hay tiêu cực. Tỷ lệ càng cao, thì từ đó càng tích cực:

<p align="center">
  <img src="https://github.com/nvsthinh/NLP_Coursera/blob/main/Course_1/data/W250.png" width="500"/>
</p>

Theo công thức toán thì biểu diễn như sau:

<p align="center">
  <img src="https://github.com/nvsthinh/NLP_Coursera/blob/main/Course_1/data/W251.png" width="300"/>
</p>

Khi $*m$* lớn hơn, chúng ta có thể gặp phải vấn đề về việc số sẽ quá nhỏ, do đó chúng ta sẽ sử dụng $\log$, ta có phương trình như sau: 

<p align="center">
  <img src="https://github.com/nvsthinh/NLP_Coursera/blob/main/Course_1/data/W252.png" width="500"/>
</p>

Thành phần đầu tiên được gọi là log prior và thành phần thứ hai là log likelihood. Chúng tôi giới thiệu thêm $𝜆$ như sau:

<p align="center">
  <img src="https://github.com/nvsthinh/NLP_Coursera/blob/main/Course_1/data/W253.png" width="500"/>
</p>

Sau khi chúng ta tính toán được $𝜆$ dictionary, việc suy luận trở nên đơn giản:

<p align="center">
  <img src="https://github.com/nvsthinh/NLP_Coursera/blob/main/Course_1/data/W254.png" width="500"/>
</p>

Như bạn có thể thấy ở trên, vì $3.3>0$, chúng ta sẽ phân loại tài liệu là dương. Nếu chúng ta nhận được số âm, chúng ta sẽ phân loại nó vào lớp âm.

## 2.6. Training naïve Bayes

Để train mô hình naïve Bayes classifier, chúng ta phải thực hiện các bước sau:

**1) Chuẩn bị một bộ dataset tweets có 2 lớp positive và negative.**

**2) Preprocess the tweets: process_tweet(tweet) ➞ [w1, w2, w3, ...]:**

- Lowercase - Chuyển sang chữ thường
- Remove punctuation, urls, names - Xóa dấu câu, URLs và tên riêng
- Remove stop words - Xóa stop words
- Stemming - Chuẩn hóa từ
- Tokenize sentences - Tách từ

**3) Tính freq(w, class):**

<p align="center">
  <img src="https://github.com/nvsthinh/NLP_Coursera/blob/main/Course_1/data/W260.png" width="500"/>
</p>

**4) Get $𝑃(𝑤∣𝑝𝑜𝑠),𝑃(𝑤∣𝑛𝑒𝑔)$**

Chúng ta có thể sử dụng bảng trên để tính xác suất.

**5) Tính $𝜆(𝑤)$**

<p align="center">
  <img src="https://github.com/nvsthinh/NLP_Coursera/blob/main/Course_1/data/W261.png" width="300"/>
</p>

**6) Tính $logprior=\log(P(pos)/P(neg))$**

$logprior=\log\displaystyle\frac{D_{pos}}{D_{neg}}$ Trong đó, $D_{pos}$ và $D_{neg}$ tương ứng với số lượng sample positive và negative.

## 2.7. Testing naïve Bayes

<p align="center">
  <img src="https://github.com/nvsthinh/NLP_Coursera/blob/main/Course_1/data/W270.png" width="500"/>
</p>

Ví dụ trên cho thấy cách bạn có thể đưa ra dự đoán dựa trên $*λ$* dictionary. Trong ví dụ này, $𝑙𝑜𝑔𝑝𝑟𝑖𝑜𝑟$ là 0 chúng ta có cùng số lượng tweets của positive và negative ($\log1=0$).

## 2.8. Applications of Naïve Bayes

Có nhiều ứng dụng của naïve Bayes bao gồm:

- Author Identification - Xác định tác giả
- Spam Filtering - Lọc thư rác
- Information Retrieval - Truy xuất thông tin
- Word Disambiguation - Phân biệt từ ngữ

Phương pháp này thường được sử dụng như một đường cơ sở đơn giản. Nó cũng thực sự nhanh.

## 2.9. Naïve Bayes Assumptions

<p align="center">
  <img src="https://github.com/nvsthinh/NLP_Coursera/blob/main/Course_1/data/W290.png" width="600"/>
</p>

Giả định chính của Naïve Bayes là các từ trong một câu độc lập với nhau. Tuy nhiên, giả định này có thể là một vấn đề vì các từ trong một câu thường có liên quan với nhau.

- Ví dụ như nếu chúng ta có câu "Trời nắng và nóng ở sa mạc Sahara", các từ "nắng" và "nóng" thường xuất hiện cùng nhau và có liên quan với nhau. Nhưng Naïve Bayes cho rằng các từ trong một câu là độc lập, điều này có thể dẫn đến việc đánh giá thấp hoặc đánh giá cao xác suất của từng từ.

<p align="center">
  <img src="https://github.com/nvsthinh/NLP_Coursera/blob/main/Course_1/data/W291.png" width="600"/>
</p>

Vấn đề khác của Naïve Bayes là nó phụ thuộc vào phân phối của tập dữ liệu huấn luyện. Nếu tập dữ liệu huấn luyện không đại diện cho dữ liệu thực tế, hiệu suất của mô hình có thể bị ảnh hưởng.

- Ví dụ, nếu các tweet tích cực xuất hiện nhiều hơn so với các tweet tiêu cực trong dữ liệu thực tế, nhưng tập dữ liệu huấn luyện lại cân bằng, mô hình có thể không dự đoán đúng cảm xúc một cách chính xác.

Mặc dù có những hạn chế này, Naïve Bayes vẫn có thể hoạt động tốt trong một số tình huống

## 2.10. Error Analysis

Khi chúng ta sử dụng các phương pháp NLP để phân tích văn bản, có thể xảy ra các lỗi. Những lỗi này có thể xảy ra vì nhiều lý do khác nhau.

- Lost Semantic Meaning: Đôi khi, trong quá trình tiền xử lý, ý nghĩa của một câu có thể bị mất. Ví dụ, nếu chúng ta loại bỏ dấu chấm câu từ một tweet biểu thị một khuôn mặt buồn, tweet đã qua xử lý có thể chỉ chứa những từ tích cực, tạo ra một cảm xúc khác.

<p align="center">
  <img src="https://github.com/nvsthinh/NLP_Coursera/blob/main/Course_1/data/W2100.png" width="500"/>
</p>

- Word Order: Thứ tự của các từ trong một câu có thể ảnh hưởng đến ý nghĩa của nó. Nếu chúng ta loại bỏ một số từ như "no" hoặc "this," văn bản đã qua xử lý có thể truyền đạt một cảm xúc khác so với câu gốc.

<p align="center">
  <img src="https://github.com/nvsthinh/NLP_Coursera/blob/main/Course_1/data/W2101.png" width="500"/>
</p>

- Quirks of Language: Ví dụ, một bài đánh giá phim tích cực có thể chứa chủ yếu các từ tiêu cực, dẫn đến một dự đoán cảm xúc sai.