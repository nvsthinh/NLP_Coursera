# 1. Week 1: Setiment Analysis with Logistic Regression

## 1.1. Supervised ML & Sentiment Analysis

### 1.1.1. Supervised ML Workflow

Về Suprervised ML, thì có một Workflow chung như sau:

- **Bước 1**: Từ Input là đặc trưng $X$ đưa vào Model theo $\theta$ - Prediction Function sẽ cho ra Output dự đoán $\hat{Y}$
- **Bước 2**: Từ Output dự đoán $\hat{Y}$ và Output thực tế $Y$, sẽ tính Cost Function và tối ưu hàm Cost đó để cập nhật tham số $\theta$

**Bước 1 → 2** sẽ lặp lại $N$ lần và $N$ là số lượng Epochs

<p align="center">
  <img src="https://github.com/nvsthinh/NLP_Coursera/blob/main/Course_1/data/W111.png" width="400"/>
</p>

### 1.1.2. Sentiment Analysis Workflow

Bài toán về Sentiment Analysis này, sẽ nhận đầu vào là Text, đi qua **Supervised ML Workflow** và trả về Label

<p align="center">
  <img src="https://github.com/nvsthinh/NLP_Coursera/blob/main/Course_1/data/W112.png" width="400"/>
</p>

## 1.2. Vocabulary & Feature Extraction

Bước đầu tiên của Workflow là làm sao để biến đổi Text thành X để Model có thể học được.

Cách cơ bản nhất, khởi tạo một Vector với độ dài là bằng với Vocabulary $V$. Sau đó, những từ nào xuất hiện trong đó sẽ bằng 1, còn lại sẽ bằng 0

Nhưng điểm yếu của phương pháp này, là nếu $V$ quá lớn thì sẽ mất nhiều thời gian để huấn luyện cũng như đưa ra dự đoán

<p align="center">
  <img src="https://github.com/nvsthinh/NLP_Coursera/blob/main/Course_1/data/W120.png" width="400"/>
</p>

## 1.3. Feature Extraction with Frequencies

Có một cách để tránh điểm yếu của phương pháp trên là đếm số lần xuất hiện của từ đó trên Positive Tweets và cả Negative Tweets. Chúng ta có ví dụ như sau:

Cho một corpus với positive và negative tweets như sau:

<p align="center">
  <img src="https://github.com/nvsthinh/NLP_Coursera/blob/main/Course_1/data/W130.png" width="400"/>
</p>

Chúng ta cần phải tạo một dictionary để map từ trong câu và class của hiện ứng với từ đó (positive hay negative) là số lượng xuất hiện của từ đó theo class

<p align="center">
  <img src="https://github.com/nvsthinh/NLP_Coursera/blob/main/Course_1/data/W131.png" width="400"/>
</p>

Chúng ta có thể tạo vector đặc trưng bằng cách đếm số từ xuất hiện như sau đối với class positive:

<p align="center">
  <img src="https://github.com/nvsthinh/NLP_Coursera/blob/main/Course_1/data/W132.png" width="400"/>
</p>

Tương ứng với class negative 

<p align="center">
  <img src="https://github.com/nvsthinh/NLP_Coursera/blob/main/Course_1/data/W133.png" width="400"/>
</p>

## 1.4. Preprocessing

Trước khi biến từ Text sang $X$, thì chúng ta cần phải clear Text đó về cùng một format. Ví dụ như sau:

<p align="center">
  <img src="https://github.com/nvsthinh/NLP_Coursera/blob/main/Course_1/data/W140.png" width="400"/>
</p>

## 1.5. Putting it all together

Tiếp theo thì chúng ta sẽ gom lại tất cả các bước đã học để ứng dụng bài toán như sau

Về tổng quan, Chúng ta bắt đầu với một đoạn Text, xử lý Preprocessing, và trích xuất đặc trưng để chuyển từ Text sang Number như sau:

<p align="center">
  <img src="https://github.com/nvsthinh/NLP_Coursera/blob/main/Course_1/data/W150.png" width="400"/>
</p>

Làm tương tự với $m$ samples, thì chúng ta có $𝑋$ có chiều tương ứng $(𝑚,3)$ là

<p align="center">
  <img src="https://github.com/nvsthinh/NLP_Coursera/blob/main/Course_1/data/W151.png" width="400"/>
</p>

Với cách diễn giải bằng code ta có như sau

<p align="center">
  <img src="https://github.com/nvsthinh/NLP_Coursera/blob/main/Course_1/data/W152.png" width="400"/>
</p>

## 1.6. Logistic Regression Overview

Sau khi trích xuất đặc trưng rồi, thì chúng ta sẽ sử dụng mô hình để huấn luyện và đưa ra dự đoán phù hợp đối với Output của bài toán. Về cơ bản thì chúng ta sẽ bắt đầu với Logistic Regression

Logistic regression sử dụng hàm Sigmoid để cho Output là xác suất nằm giữa 0 và 1. Hàm Sigmoid viết dưới dạng có biến là tham số $\theta$ và Inpu $𝑥(𝑖)$ như sau:

<p align="center">
  <img src="https://github.com/nvsthinh/NLP_Coursera/blob/main/Course_1/data/W160.png" width="400"/>
</p>

Ví dụ cụ thể như sau:

<p align="center">
  <img src="https://github.com/nvsthinh/NLP_Coursera/blob/main/Course_1/data/W161.png" width="400"/>
</p>

## 1.7. **Logistic Regression: Training**

Sau khi hiểu model Logistic Regression, thì tới bước huấn luyện theo flow như sau:

<p align="center">
  <img src="https://github.com/nvsthinh/NLP_Coursera/blob/main/Course_1/data/W170.png" width="400"/>
</p>

Thì khi huấn luyện xong thì hàm Cost có giá trị theo Epoch như sau:

<p align="center">
  <img src="https://github.com/nvsthinh/NLP_Coursera/blob/main/Course_1/data/W180.png" width="400"/>
</p>

## 1.8. L**ogistic Regression: Testing**

Sau khi huấn luyện, thì chúng ta sẽ kiểm thử với $X_{val}, Y_{val}, \theta$

Với Output đầu ra là $h(X_{val},\theta)$. Chúng ta sẽ lấy những sample có xác suất lớn hơn 0.5 là 1 và nhỏ hơn là 0. Như sau:

<p align="center">
  <img src="https://github.com/nvsthinh/NLP_Coursera/blob/main/Course_1/data/W180.png" width="400"/>
</p>

Sau khi có Output có format như class thì chúng ta sẽ tính Accuracy giữa Dự đoán và Thực tế

<p align="center">
  <img src="https://github.com/nvsthinh/NLP_Coursera/blob/main/Course_1/data/W181.png" width="400"/>
</p>

## 1.9. **Logistic Regression: Cost Function**

Hàm Cost Function của Logistic Regression là hàm Binary Cross Entropy

$$
J(\theta) = -\frac{1}{m} \sum_{i=1}^{m} \left[ y^{(i)} \log h\left(x^{(i)}, \theta\right) + \left(1 - y^{(i)}\right) \log \left(1 - h\left(x^{(i)}, \theta\right)\right) \right]
$$

<p align="center">
  <img src="https://github.com/nvsthinh/NLP_Coursera/blob/main/Course_1/data/W190.png" width="400"/>
</p>