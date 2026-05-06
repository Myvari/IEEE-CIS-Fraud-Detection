# IEEE-CIS-Fraud-Detection

### Kaggle-ის კონკურსის მოკლე მიმოხილვა
IEEE-CIS Fraud Detection არის Kaggle-ის კლასიფიკაციის კონკურსი, რომლის მიზანია ონლაინ ტრანზაქციებში თაღლითური შემთხვევების აღმოჩენა. მონაცემები მოწოდებულია Vesta Corporation-ის რეალური ტრანზაქციების საფუძველზე და თითოეული ჩანაწერისთვის საჭიროა განისაზღვროს არის თუ არა ეს ტრანზაქცია თაღლითური (`isFraud`).

ამ ამოცანაში სამიზნე ცვლადი ბინარულია, კონკურსის შეფასების მეტრიკაა ROC-AUC, სასწავლო მონაცემებში თაღლითური ტრანზაქციების წილი დაახლოებით 3.5%-ია, ამიტომ accuracy არ არის საკმარისად ინფორმატიული მეტრიკა. მაგალითად  ყველა ტრანზაქციის არათაღლითურად მონიშვნა მაღალ accuracy-ს მოგვცემდა, მაგრამ მოდელი რეალურად ვერ აღმოაჩენდა fraud შემთხვევებს.

მონაცემები გაყოფილია ორ ძირითად ფაილად:
- `train_transaction.csv` / `test_transaction.csv` - ტრანზაქციის ძირითადი ინფორმაცია, მათ შორის თანხა, დრო, პროდუქტის ტიპი, ბარათთან და მისამართთან დაკავშირებული ანონიმიზებული ცვლადები, ასევე `C`, `D`, `M` და `V` ტიპის მახასიათებლები.
- `train_identity.csv` / `test_identity.csv` - იდენტობასთან და მოწყობილობასთან დაკავშირებული დამატებითი ინფორმაცია, მაგალითად `DeviceType`, `DeviceInfo` და `id_*` ცვლადები.

ამ ფაილების მარცნივ დაერთება ხდება `TransactionID` სვეტით, რადგან identity ფაილში ინფორმაცია ყველა ტრანზაქციისთვის არ არსებობს, რის გამოც გაერთიანების შემდეგ დიდი რაოდენობით missing value ჩნდება. ბევრი სვეტის აღწერა არ არის პირდაპირ მოცემული, ამიტომ პროექტის მნიშვნელოვანი ნაწილი დაეთმო EDA-ს, missing pattern-ების ანალიზს, time-based split-ს, feature engineering-ს და სხვადასხვა მოდელის შედარებას.

---

## რეპოზიტორიის სტრუქტურა

```text
IEEE-CIS-Fraud-Detection/
│
├── data/
│   └── raw/
│       ├── train_transaction.csv
│       ├── train_identity.csv
│       ├── test_transaction.csv
│       └── test_identity.csv
│
├── model_experiment_LogisticRegression.ipynb
├── model_experiment_RandomForest.ipynb
├── model_experiment_AdaBoost.ipynb
├── model_experiment_XGBoost.ipynb
│
├── model_inference.ipynb
│
└── README.md
```

## ფაილების აღწერა

| ფაილი | აღწერა |
|------|--------|
| `train_transaction.csv`, `train_identity.csv` | საწყისი train მონაცემები. ორივე ფაილი ერთიანდება TransactionID სვეტით |
| `test_transaction.csv`, `test_identity.csv` | საწყისი test მონაცემები, რომლებზეც საბოლოო submission-ისთვის კეთდება პროგნოზი |
| `model_experiment_LogisticRegression.ipynb` | ძირითადი notebook EDA-სთვის, Logistic Regression-ის ექსპერიმენტები, preprocessing pipeline და MLflow logging |
| `model_experiment_RandomForest.ipynb` | Random Forest-ის ექსპერიმენტები tree-based preprocessing-ით და სხვადასხვა feature selection მიდგომით |
| `model_experiment_AdaBoost.ipynb` | ძAdaBoost-ის ექსპერიმენტები IV preprocessing ვერსიებით |
| `model_experiment_XGBoost.ipynb` | XGBoost-ის ექსპერიმენტები, დამატებითი feature engineering და საბოლოო საუკეთესო მოდელი |
| `model_inference.ipynb` | საუკეთესო pipeline-ის ჩატვირთვა MLflow artifact-ებიდან და Kaggle submission-ის დაგენერირება |

---

## მონაცემთა გაწმენდა და დამუშავება (Cleaning & Preprocessing)


**1. Train მონაცემების გაერთიანება**

პირველ ეტაპზე `train_transaction.csv` და `train_identity.csv` გაერთიანდა `TransactionID` სვეტით. გამოყენებული იყო `left join` რადგან ძირითადი მონაცემი transaction ფაილშია, identity ფაილში კი ინფორმაცია მხოლოდ ტრანზაქციების ნაწილზე არსებობს.
```python
df = pd.merge(df_transaction, df_identity, on="TransactionID", how="left")
```
ყველა ტრანზაქციისთვის არ არსებობს, რაც შემდეგ ეტაპზე missing value-ების ანალიზში განსაკუთრებით მნიშვნელოვანი აღმოჩნდა.

**2. Train/Validation გაყოფა**

ამ competition-ში ჩვეულებრივი random split არ არის საუკეთესო არჩევანი, რადგან `TransactionDT` დროის მიმდევრობას ასახავს. Kaggle-ის test set-იც train set-ის შემდეგ პერიოდს ეკუთვნის:
<img width="1189" height="490" alt="image" src="https://github.com/user-attachments/assets/529c77c7-bf1a-47f1-ad0d-41fff02aada2" />
ამიტომ validation უფრო რეალისტური რომ ყოფილიყო, მონაცემები ჯერ დალაგდა `TransactionDT`-ის მიხედვით და შემდეგ გაიყო დროით.

ანუ მოდელი სწავლობდა უფრო ადრეულ ტრანზაქციებზე და მოწმდებოდა უფრო გვიანდელ ტრანზაქციებზე. ეს მიდგომა უფრო მკაცრია, მაგრამ უკეთ ამოწმებს, შეძლებს თუ არა მოდელი მომავალ პერიოდზე განზოგადებას.

**3. Missing value-ების ანალიზი**

EDA-მ აჩვენა, რომ მონაცემებში missing value-ები ძალიან დიდი რაოდენობითაა:
<img width="690" height="490" alt="image" src="https://github.com/user-attachments/assets/a8e92044-2844-4441-82a3-a65ad7cc95e3" />

ბევრი identity/device feature missing არის იმიტომ, რომ კონკრეტულ ტრანზაქციაზე ასეთი ინფორმაცია საერთოდ არ დაფიქსირდა. ამიტომ missingness ზოგ შემთხვევაში ცალკე სიგნალადაც შეიძლება ჩაითვალოს.
<img width="842" height="270" alt="image" src="https://github.com/user-attachments/assets/59d0547b-39b1-4f09-afd3-a3c7bb0cb264" />
ამის გამო tree-based მოდელებში numeric imputer-ს დაემატა ```add_indicator=True```, რაც ქმნის დამატებით binary feature-ებს იმის აღსანიშნავად, კონკრეტული მნიშვნელობა თავდაპირველად missing იყო თუ არა.

**4. სვეტების სახელების გასწორება**

EDA-ს დროს გამოჩნდა, რომ train identity ფაილში სვეტები იყო ფორმატით:
```
id_01, id_02, ...
```
ხოლო test identity ფაილში:
```
id-01, id-02, ...
```
ამიტომ cleaning ეტაპზე დაემატა სვეტების სახელების სტანდარტიზაცია:
```python 
X.columns = X.columns.str.replace("id-", "id_", regex=False)
```
**5. უსარგებლო და ტექნიკური სვეტების წაშლა**

ყველა მოდელში იშლებოდა `TransactionID`, რადგან ის მხოლოდ უნიკალური იდენტიფიკატორია და პირდაპირ predictive მნიშვნელობა არ აქვს.
Tree-based მოდელებში ასევე წაიშალა DeviceInfo, რადგან ეს იყო ძალიან high-cardinality categorical feature და preprocessing/training-ს ზედმეტად ამძიმებდა.
<img width="187" height="190" alt="image" src="https://github.com/user-attachments/assets/1f0154e4-062d-40c2-b02b-310da92738bc" />

დამატებით, ზოგიერთ tree-based pipeline-ში გამოყენებული იყო missing_threshold=0.95, იშლებოდა ისეთი სვეტები, სადაც მნიშვნელობების 95%-ზე მეტი missing იყო.

**6. Numeric და categorical preprocessing**

Preprocessing განსხვავდებოდა მოდელის ტიპის მიხედვით.
Logistic Regression-ისთვის გამოყენებული იყო:
```text
numeric missing values       -> median
categorical missing values   -> "missing"
categorical encoding         -> OneHotEncoder
scaling                      -> StandardScaler
```
Logistic Regression coefficient-based მოდელია, ამიტომ numeric feature-ების მასშტაბი მნიშვნელოვანია. ასევე categorical feature-ებისთვის 
OneHotEncoder უფრო სწორი არჩევანია, რადგან ordinal encoding-მა შეიძლება ხელოვნური რიგითობა შექმნას.

Tree-based მოდელებისთვის გამოყენებული იყო:
```text
numeric missing values       -> median + missing indicators
categorical missing values   -> "missing"
categorical encoding         -> OrdinalEncoder
scaling                      -> არ გამოიყენება
```
Random Forest, AdaBoost და XGBoost split-based მოდელებია, ამიტომ scaling საჭირო არ არის. OneHotEncoder-ის ნაცვლად OrdinalEncoder გამოვიყენე, რადგან OHE ძალიან ზრდიდა feature count-ს და training-ს ამძიმებდა, განსაკუთრებით დიდი IEEE-CIS მონაცემებისთვის.

**7. Constant და near-zero variance feature-ების მოცილება**

Cleaning ეტაპზე იშლებოდა constant სვეტები, ანუ ისეთი სვეტები, სადაც მხოლოდ ერთი უნიკალური მნიშვნელობა იყო.

Tree-based pipeline-ებში დამატებით გამოყენებული იყო `NearZeroVarianceReducer`, რომელიც შლიდა ისეთ feature-ებს, სადაც ერთი მნიშვნელობა თითქმის მთლიანად დომინირებდა.

```
Near-zero variance dominant ratio threshold: 0.995
```

ამან შეამცირა feature space და მოაშორა ისეთი სვეტები, რომლებსაც მოდელისთვის პრაქტიკულად მცირე ინფორმაცია მოჰქონდა.


**8. მოდელებს შორის განსხვავებები**

Cleaning ყველა მოდელში ერთნაირი არ ყოფილა:
- Logistic Regression-ში გამოყენებული იყო OneHotEncoder, StandardScaler.
- Random Forest-ში გამოყენებული იყო OrdinalEncoder, missing indicators.
- AdaBoost-ში გამოყენებული იყო Random Forest-ის მსგავსი tree-based preprocessing, მაგრამ ნაკლები ექსპერიმენტით, რადგან training შედარებით ნელი აღმოჩნდა.
- XGBoost-ში გამოყენებული იყო tree-based preprocessing (შემდეგ დამატებითი FE), რომელმაც საუკეთესო შედეგი აჩვენა.

ამ განსხვავებების მიზანი იყო, preprocessing მორგებოდა კონკრეტული მოდელის ბუნებას. Linear model-ს სჭირდება scaled და one-hot encoded feature-ები, ხოლო tree-based მოდელებისთვის უფრო ეფექტური აღმოჩნდა კომპაქტური ordinal representation, missing indicator-ები და ნაკლები dimensionality.

---
### Feature Engineering

Feature Engineering ეტაპზე მიზანი იყო raw სვეტებიდან ისეთი ახალი ცვლადების შექმნა, რომლებიც მოდელებს ტრანზაქციის კონტექსტის უკეთ დაჭერაში დაეხმარებოდა. რადგან IEEE-CIS მონაცემებში ბევრი feature ანონიმიზებულია და პირდაპირი აღწერა არ აქვს, ახალი ცვლადები ძირითადად შეიქმნა დროის, თანხის, identity coverage-ის, email domain-ების და card/address კომბინაციების საფუძველზე.

#### საბაზისო Feature Engineering

პირველ ვერსიაში დაემატა რამდენიმე ზოგადი feature, რომლებიც ყველა მოდელში ან მოდელების ნაწილში გამოიყენებოდა:

**1. Identity coverage feature**

Identity ფაილი ყველა ტრანზაქციისთვის არ არსებობდა, ამიტომ დაემატა binary feature:

```python
has_identity
```

<img width="545" height="393" alt="image" src="https://github.com/user-attachments/assets/6be39adb-7fc7-46d1-99c4-386baa87f964" />


**2. TransactionAmt feature-ები**

`TransactionAmt` იყო ერთ-ერთი მთავარი numeric feature, ამიტომ მისგან შეიქმნა დამატებითი ცვლადები:

```python
TransactionAmt_log
TransactionAmt_decimal
```
`TransactionAmt_log` ამცირებს თანხის ძლიერი right-skew განაწილების ეფექტს,
<img width="578" height="413" alt="image" src="https://github.com/user-attachments/assets/f0b0f654-524d-485b-8ab8-8098dfde358f" />

ხოლო `TransactionAmt_decimal` ინახავს თანხის ათობით ნაწილს, რაც ზოგიერთ fraud pattern-ში შეიძლება მნიშვნელოვანი იყოს.
<img width="989" height="590" alt="image" src="https://github.com/user-attachments/assets/da825d3a-6b59-4ed7-8fa7-d7ee12b3acdf" />

**3. Time-based feature-ები**

`TransactionDT` გადაყვანილი იყო უფრო ინტერპრეტირებად დროით ცვლადებში:

```python
transaction_day
transaction_week
transaction_hour_sin
transaction_hour_cos
transaction_weekday_sin
transaction_weekday_cos
transaction_day_of_month_sin
transaction_day_of_month_cos
```
სინუსისა და კოსინუსის გარდაქმნები გამოყენებულია ციკლური ცვლადებისთვის რადგან საათი, კვირის დღე და თვის დღე წრიული ბუნებისაა. მაგალითად 23:00 და 00:00 ერთმანეთთან ახლოს უნდა აღიქმებოდეს და არა როგორც შორეული მნიშვნელობები.

(სახელები ფარდობითია)
<img width="1990" height="1490" alt="image" src="https://github.com/user-attachments/assets/bcdb5397-aa6b-4fc0-ab37-46bff8506cd4" />
<img width="1990" height="1189" alt="image" src="https://github.com/user-attachments/assets/3828341b-f35e-4e61-9e05-cc9646fdea59" />


**4. Email domain feature-ები**

`P_emaildomain` და `R_emaildomain` დაიშალა provider და suffix(XGBoost-ისთვის) ნაწილებად:

```python
P_emaildomain_provider
P_emaildomain_suffix
R_emaildomain_provider
R_emaildomain_suffix
```
email domain-ის სრული მნიშვნელობა შეიძლება ზედმეტად სპეციფიკური იყოს, ხოლო provider/suffix ნაწილები უფრო ზოგად pattern-ებს აჩვენებს.

**5. Skewed numeric feature-ების log transform ტესტი**

EDA-ში numeric feature-ებისთვის შემოწმდა skewness, განსაკუთრებით `C`, `D`, `id` და `V` ჯგუფების სვეტებზე. მიზანი იყო ისეთი სვეტების პოვნა, რომლებსაც ძლიერი right-skew ჰქონდათ და log transform-ით შეიძლებოდა განაწილების შედარებით დასტაბილურება.

<img width="452" height="346" alt="image" src="https://github.com/user-attachments/assets/9780686c-9ffa-4e9f-8807-a42ebcd2695c" />


იდეა იყო, რომ მაღალი skewness-ის მქონე numeric სვეტებისთვის შექმნილიყო დამატებითი log feature-ები. მაგალითად:
```python
log1p(C1), log1p(C13), log1p(D1)
```
თუმცა ტესტებმა აჩვენა რომ ეს მიდგომა არც ისე გამოსადეგი აღმოჩნდა, ამიტომ აქტიურად არ გამოვიყენეთ.

#### XGBoost Feature Engineering V2

საუკეთესო შედეგის მიღების შემდეგ XGBoost-ისთვის შეიქმნა დამატებითი Feature Engineering V2. ამ ვერსიის მიზანი იყო არა მხოლოდ ცალკეული raw feature-ების გამოყენება, არამედ ტრანზაქციის ქცევითი კონტექსტის შექმნა: რამდენად ხშირია კონკრეტული card/address/email კომბინაცია და რამდენად უჩვეულოა თანხა ამ ჯგუფისთვის.

**1. UID feature-ები**

შეიქმნა რამდენიმე კომბინირებული id-სებრი feature:

```python
uid_card_addr -> card1 + addr1
uid_card_addr_product -> card1 + addr1 + ProductCD
uid_card_email -> card1 + P_emaildomain
```

ეს სვეტები აერთიანებს card, address, product და email ინფორმაციას, მიზანი იყო ისეთი კომბინაციების დაჭერა, რომლებიც ცალკე აღებული feature-ებით არ ჩანდა და დამახასიათებელი იყო UID-თ იდენტიფიცირებული ტრანზაქციებისთვის.

**2. Frequency Encoding**

შემდეგ ეტაპზე დაემატა frequency encoding. თითოეული არჩეული სვეტისთვის ითვლებოდა, რამდენჯერ გვხვდებოდა კონკრეტული მნიშვნელობა train მონაცემებში.

```python
card1, card2, card3, card5,
addr1, addr2,
P_emaildomain, R_emaildomain,
ProductCD,
uid_card_addr,
uid_card_addr_product,
uid_card_email
```
ასეთი სახის ინფორმაცია მნიშვნელოვანია fraud detection-ში, რადგან იშვიათი card/address/email კომბინაციები ხშირად განსხვავებულ რისკს ატარებს.

**3. TransactionAmt group statistics**

შემდეგ შეიქმნა feature-ები, რომლებიც ამოწმებს, რამდენად უჩვეულოა ტრანზაქციის თანხა კონკრეტული ჯგუფისთვის.

ჯგუფებისთვის გამოყენებული იყო:

```python
card1
addr1
ProductCD
uid_card_addr
uid_card_addr_product
uid_card_email
```
თითოეული ჯგუფისთვის train set-ზე დაითვალა:
- mean TransactionAmt
- median TransactionAmt
- std TransactionAmt

შემდეგ თითოეული ტრანზაქციისთვის შეიქმნა ratio და z-score feature-ები:
+ TransactionAmt_to_{group}_mean
+ TransactionAmt_to_{group}_median
+ TransactionAmt_zscore_{group}

მაგალითად თუ კონკრეტული card/address კომბინაციისთვის საშუალო თანხა არის 100 და მიმდინარე ტრანზაქცია არის 300, მაშინ:

```
TransactionAmt_to_uid_card_addr_mean = 3.0
```

ეს მოდელს აძლევს ინფორმაციას, რომ თანხა ამ კონკრეტული card/address ჯგუფისთვის ჩვეულებრივზე სამჯერ მაღალია.

**4. D-time feature-ები**

D ტიპის სვეტები დროით delta feature-ებს წარმოადგენენ. ამიტომ დაემატა ახალი ცვლადები ნაკლები missingness-ის მქონე D feature-ებს:

```python
transaction_day_minus_D* = transaction_day - D*
```

ამ მიდგომის მიზანი იყო მიახლოებითი reference day-ის მიღება. ასეთი feature-ები შეიძლება დაეხმაროს მოდელს განმეორებითი ქცევების ან account/card ისტორიის pattern-ების დაჭერაში.

---

### Feature Selection

Feature Selection ეტაპზე მიზანი იყო feature-ების რაოდენობის შემცირება, noise-ის მოცილება და ისეთი ცვლადების დატოვება, რომლებსაც fraud prediction-ისთვის რეალური ინფორმაცია მოჰქონდათ. სხვადასხვა მოდელისთვის feature selection ერთნაირი არ ყოფილა, რადგან Logistic Regression და tree-based მოდელები feature-ებს განსხვავებულად იყენებენ.

- **1. Information Value Selector**

პირველი გამოყენებული მიდგომა იყო **Information Value (IV)**. ეს მეთოდი თითოეულ feature-ს ცალ-ცალკე აფასებს (univariate) და ამოწმებს, რამდენად კარგად განასხვავებს fraud და non-fraud კლასებს.

Numeric feature-ებისთვის მნიშვნელობები იყოფოდა bin-ებად, ხოლო categorical feature-ები მუშავდებოდა category-level განაწილებით. თითოეული feature-ისთვის ითვლებოდა IV score და შემდეგ რჩებოდა მხოლოდ ის feature-ები, რომელთა IV მნიშვნელობა threshold-ზე მაღალი იყო.

გამოყენებული threshold:
```text
IV threshold = 0.01 / 0.02
```
IV selector განსაკუთრებით სასარგებლო იყო Logistic Regression-ში, რადგან OneHotEncoder-ის შემდეგ feature space ძალიან იზრდებოდა. IV-მ საშუალება მოგვცა, weak/uninformative feature-ების ნაწილი preprocessing-მდე მოგვემორებინა.

Tree-based მოდელებშიც გაიტესტა IV preprocessing ვერსია. Random Forest-სა და AdaBoost-ში IV ვერსიამ ზოგ baseline-ზე უკეთესი შედეგი აჩვენა, თუმცა XGBoost-ის საუკეთესო შედეგი საბოლოოდ Feature Engineering V2 + baseline tree preprocessing-ით მივიღე.

- **2. Correlation Reducer**

მეორე მიდგომა იყო highly correlated numeric feature-ების შემცირება. IEEE-CIS მონაცემებში განსაკუთრებით V ჯგუფის feature-ებს შორის ბევრი მაღალი კორელაცია იყო.
<img width="883" height="790" alt="image" src="https://github.com/user-attachments/assets/387b8ac6-0419-4e46-98ba-f37a9ac9b6b2" />

გამოყენებული threshold:
```text
Correlation threshold = 0.95
```

ეს მიდგომა უფრო მეტად Random Forest-ის ექსპერიმენტებში გაიტესტა. მიზანი იყო redundant feature-ების შემცირება და training-ის გამარტივება 
თუმცა საბოლოო შედეგებში correlation-based preprocessing არ აღმოჩნდა საუკეთესო ვერსია. 

Tree-based მოდელებს ხშირად შეუძლიათ correlated feature-ებთან მუშაობა, ამიტომ აგრესიული correlation filtering ზოგჯერ სასარგებლო ინფორმაციის დაკარგვასაც იწვევს.

- **3. RFE - Recursive Feature Elimination**

Logistic Regression notebook-ში გაიტესტა RFE-ც. RFE estimator-ად გამოყენებული იყო Logistic Regression, რადგან წრფივი მოდელის coefficient-ები feature importance-ის მსგავს სიგნალს იძლევა.

თუმცა IEEE-CIS მონაცემებზე RFE საკმაოდ ნელი აღმოჩნდა, რადგან feature count დიდი იყო, განსაკუთრებით OneHotEncoder-ის შემდეგ. ამიტომ საბოლოო workflow-ში RFE გადავწყვიტე არ ჩამერთო რადგან დროსთან მიმართებაში მიღებული სარგებელი საკმარისი არ იყო. ის უფრო მეტად გამოყენებული იყო როგორც დამატებითი ექსპერიმენტი.

- **4. Near-Zero Variance Reducer**

Tree-based მოდელებისთვის გამოყენებული იყო Near-Zero Variance Reducer. ეს მეთოდი შლის ისეთ feature-ებს, სადაც ერთი მნიშვნელობა თითქმის ყველა row-ში დომინირებს.
```text
dominant ratio threshold = 0.995
```
ეს განსაკუთრებით სასარგებლო იყო Random Forest, AdaBoost და XGBoost notebook-ებში, რადგან tree-based მოდელებში ბევრი sparse ან თითქმის constant feature ზედმეტ split candidate-ებს ქმნიდა.

---

### Pipeline-ის მიმდევრობა

საბაზისო pipeline-ის ზოგადი მიმდევრობა იყო:

- 1. **Data Cleaning**

  სწორდება სვეტების სახელები, იშლებოდა ტექნიკური სვეტები, და საჭიროების შემთხვევაში მაღალი missingness-ის მქონე სვეტების მოცილება.
- 3. **Feature Engineering**
  
  ემატებოდა ახალი feature-ები, XGBoost-ის საბოლოო ვერსიაში დამატებით გამოყენებული იყო `XGBoostFeatureEngineerV2`.
- 4. **Feature Selection**
  
  მოდელის მიხედვით გამოიყენებოდა სხვადასხვა feature selection მიდგომა.
- 5. **Preprocessing / ColumnTransformer**
  
  ამ ეტაპზე numeric და categorical სვეტები ცალ-ცალკე მუშავდებოდა.


საბოლოოდ ყველაზე სტაბილური მიდგომა აღმოჩნდა tree-based მოდელებისთვის ზომიერი feature filtering: high-missing/constant feature-ების მოცილება, near-zero variance filtering და შემდეგ მოდელისთვის ინფორმაციული engineered feature-ების მიწოდება. ზედმეტად აგრესიული selection, მაგალითად correlation filtering ან ძალიან მკაცრი IV threshold, ყოველთვის არ აუმჯობესებდა შედეგს, რადგან fraud detection-ში ზოგიერთი იშვიათი ან correlated feature მაინც შეიძლება სასარგებლო სიგნალს შეიცავდეს.

---

## ტრენინგი და ექსპერიმენტები

პროექტში ტესტირებულია რამდენიმე კლასიფიკაციის მოდელი:

- Logistic Regression
- Random Forest Classifier
- AdaBoost Classifier
- XGBoost Classifier

თითოეული მოდელისთვის შეიქმნა ცალკე notebook და ცალკე MLflow experiment. 
ექსპერიმენტებში იცვლებოდა როგორც preprocessing/feature selection მიდგომა, ისე მოდელის ჰიპერპარამეტრები.

გამოყენებული ძირითადი მეტრიკები:

- `ROC-AUC` - მთავარი მეტრიკა, რადგან Kaggle competition სწორედ ამით ფასდება
- `Average Precision` - მნიშვნელოვანი მეტრიკა არაბალანსირებული კლასიფიკაციისთვის
- `Log Loss` - probability prediction-ის ხარისხის შესაფასებლად
- `Precision`, `Recall`, `F1` - fraud კლასის ქცევის შესაფასებლად
- `train_holdout_gap` / `roc_auc_gap` - overfitting-ის შესაფასებლად

Validation-ისთვის გამოყენებული იყო time-based split, რადგან `TransactionDT` დროით მიმდევრობას წარმოადგენს და competition-ის test set train set-ის შემდეგ პერიოდს ეკუთვნის. ეს random split-ზე უფრო რეალისტური შეფასებაა, რადგან მოდელი მოწმდება უფრო გვიანდელ ტრანზაქციებზე.

---

### Logistic Regression

Logistic Regression გამოვიყენე როგორც baseline მოდელი. ამ მოდელისთვის preprocessing განსხვავდებოდა tree-based მოდელებისგან: categorical feature-ებზე გამოყენებული იყო OneHotEncoder, numeric feature-ებზე StandardScaler, ხოლო feature selection-ისთვის Information Value Selector.

ჰიპერპარამეტრებში გაიტესტა:

- `penalty`: L1 და L2
- `C`: რეგულარიზაციის სიძლიერე
- `solver`: `lbfgs`, `liblinear`
- `class_weight`: `None`, `balanced`

პირველ ეტაპზე ჩავატარე სწრაფი holdout screening, რადგან სრული TimeSeriesCV ყველა პარამეტრისთვის ძალიან დიდ დროს იღებდა. საუკეთესო holdout შედეგები აჩვენა balanced მოდელებმა:


| run_name                     |   holdout_roc_auc |  holdout_average_precision |   holdout_log_loss |   metrics.holdout_recall |
|:----------------------------------------|--------------------------:|------------------------------------:|---------------------------:|-------------------------:|
| LogReg_FAST_l1_C0.01_liblinear_balanced |                  0.836343 |                            0.19029  |                   0.85661  |                 0.769439 |
| LogReg_FAST_l2_C0.01_lbfgs_balanced     |                  0.834842 |                            0.20011  |                   0.832681 |                 0.759596 |
| LogReg_FAST_l2_C0.1_lbfgs_balanced      |                  0.833974 |                            0.20046  |                   0.824159 |                 0.750246 |
| LogReg_FAST_l2_C0.1_lbfgs_balanced      |                  0.830953 |                            0.196315 |                   0.831791 |                 0.753937 |
| LogReg_FAST_l2_C1.0_lbfgs_unbalanced    |                  0.823723 |                            0.198194 |                   0.421261 |                 0.261073 |


შემდეგ საუკეთესო candidate-ზე (lbfgs რადგან 12-ჯერ უფრო სწრაფი იყო) გავუშვი TimeSeriesCV. CV შედეგი იყო:

| Run Name | cv_train_roc_auc  | cv_val_roc_auc | roc_auc_gap |
|---|---:|---:|---:|
| LogReg_Finalist_TimeSeriesCV | 0.8880 | 0.8419 | 0.0461 |

საბოლოოდ full pipeline თავიდან დავ-fit-ეთ train split-ზე და შეფასდა holdout validation set-ზე.

| Metric | Value |
|---|---:|
| Holdout ROC-AUC | 0.8348 |
| Average Precision | 0.2001 |
| Log Loss | 0.8327 |
| F1 | 0.1773 |
| Precision | 0.1004 |
| Recall | 0.7596 |

Logistic Regression-მა აჩვენა, რომ balanced class weight fraud კლასის recall-ს ზრდიდა, მაგრამ precision ძალიან დაბალი რჩებოდა. მოდელი ბევრ fraud შემთხვევას პოულობდა, თუმცა ამასთან ერთად ბევრ non-fraud ტრანზაქციასაც fraud-ად ნიშნავდა. ეს მოსალოდნელი იყო, რადგან მონაცემებში ბევრი არაწრფივი დამოკიდებულებაა, Logistic Regression კი მხოლოდ წრფივ decision boundary-ს სწავლობს.

ამიტომ Logistic Regression კარგი baseline აღმოჩნდა, მაგრამ საბოლოო მოდელად არ შეირჩა.

---

### Random Forest

Random Forest გამოვიყენე როგორც bagging-based tree ensemble. Logistic Regression-თან შედარებით, Random Forest-ს უკეთ შეუძლია არაწრფივი დამოკიდებულებების დაჭერა და feature interaction-ებთან მუშაობა.

Random Forest-ზე გავტესტე სამი ძირითადი preprocessing მიდგომა:

1. **Baseline preprocessing** - tree-based preprocessing feature selection-ის გარეშე;
2. **IV preprocessing** - Information Value selector-ით შემცირებული feature set;
3. **Correlation preprocessing** - მაღალი correlation-ის მქონე numeric feature-ების შემცირება.

ჰიპერპარამეტრებში გაიტესტა:

- `n_estimators`
- `max_depth`
- `min_samples_leaf`
- `min_samples_split`
- `max_features`
- `class_weight`

#### Baseline Random Forest

Baseline preprocessing-ით საუკეთესო შედეგები აჩვენა უფრო ღრმა balanced Random Forest-მა:

| run_name                                                   |   cv_train_auc |   cv_val_auc |   auc_gap |   holdout_auc |   holdout_ap |   holdout_log_loss |
|:-----------------------------------------------------------|---------------:|-------------:|----------:|--------------:|-------------:|-------------------:|
| RF_depthNone_split5_leaf10_trees300_featuressqrt_balanced  |       0.997242 |     0.884028 | 0.113214  |      0.893314 |     0.487518 |           0.192474 |
| RF_depth18_split5_leaf10_trees200_featuressqrt_balanced    |       0.976085 |     0.866295 | 0.10979   |      0.876401 |     0.454638 |           0.282879 |
| RF_depth14_split10_leaf10_trees200_featuressqrt_balanced   |       0.95259  |     0.866243 | 0.086347  |      0.872688 |     0.458246 |           0.335594 |
| RF_depth10_split20_leaf20_trees150_featuressqrt_balanced   |       0.910215 |     0.86302  | 0.0471945 |      0.86627  |     0.440514 |           0.402689 |

აქ კარგად ჩანს underfitting/overfitting trade-off. მცირე სიღრმის მოდელები (`max_depth=6`) ნაკლებად overfit-დებოდა, მაგრამ მათი validation და holdout score დაბალი იყო:
| run_name                                                   |   cv_train_auc |   cv_val_auc |   auc_gap |   holdout_auc |   holdout_ap |   holdout_log_loss |
|:-----------------------------------------------------------|---------------:|-------------:|----------:|--------------:|-------------:|-------------------:|
| RF_depth6_split20_leaf20_trees100_featuressqrt_balanced    |       0.868853 |     0.84675  | 0.0221027 |      0.845572 |     0.3932   |           0.465229 |

უფრო ღრმა მოდელებმა ბევრად უკეთესი AUC აჩვენეს, თუმცა train/validation gap საგრძნობლად გაიზარდა.

საუკეთესო baseline მოდელს ჰქონდა ყველაზე მაღალი holdout ROC-AUC - **0.8933**, მაგრამ gap **0.1132** იყო, რაც overfitting-ის ნიშანია.

#### IV preprocessing

IV preprocessing-მა feature-ების რაოდენობა შეამცირა და fast screening ეტაპზე კარგი შედეგი აჩვენა, თუმცა baseline-ის საუკეთესო შედეგს ვერ აჯობა.

| run_name                                   |   holdout_auc |   holdout_ap |   holdout_log_loss | preprocessing    |
|:-------------------------------------------|--------------:|-------------:|-------------------:|:-----------------|
| RF_FAST_depth18_leaf10_trees200_balanced   |      0.884452 |     0.475683 |           0.275212 | IV_Preprocessing |
| RF_FAST_depth14_leaf10_trees200_unbalanced |      0.867088 |     0.436857 |           0.1039   | IV_Preprocessing |
| RF_FAST_depth10_leaf30_trees100_balanced   |      0.866834 |     0.444509 |           0.405226 | IV_Preprocessing |
| RF_FAST_depth10_leaf20_trees150_balanced   |      0.8654   |     0.442568 |           0.405449 | IV_Preprocessing |
| RF_FAST_depth6_leaf50_trees50_balanced     |      0.846165 |     0.394835 |           0.464917 | IV_Preprocessing |

IV finalist TimeSeriesCV შედეგი იყო:

| run_name | cv_train_auc  | cv_val_auc | auc_gap  |
|---|---:|---:|---:|
| RandomForest_Finalist_TimeSeriesCV | 0.9811 | 0.8762 | 0.1049 |

საბოლოო full pipeline:

| Metric | Value |
|---|---:|
| Holdout ROC-AUC | 0.8845 |
| Average Precision | 0.4757 |
| Log Loss | 0.2752 |
| F1 | 0.4012 |
| Precision | 0.3138 |
| Recall | 0.5561 |

IV preprocessing-ის საუკეთესო holdout AUC იყო **0.8845**. ეს baseline-ის საუკეთესო შედეგზე დაბალია, მაგრამ Average Precision საკმაოდ ძლიერი იყო - **0.4757**. მიუხედავად ამისა, CV gap კვლავ მაღალი დარჩა, ამიტომ IV-მ overfitting პრობლემა ბოლომდე ვერ მოაგვარა.

#### Correlation preprocessing

Correlation preprocessing-ის მიზანი იყო highly correlated numeric feature-ების შემცირება. შედეგებმა აჩვენა, რომ correlation filtering გარკვეულწილად ამცირებდა redundancy-ს, მაგრამ საუკეთესო baseline-ს ვერ აჯობა.

| run_name                                   |   cv_train_auc |   cv_val_auc |   auc_gap |   holdout_auc |   holdout_ap |   holdout_log_loss |
|:-------------------------------------------|---------------:|-------------:|----------:|--------------:|-------------:|-------------------:|
| RF_CORR_depth18_leaf10_trees200_balanced   |       0.976543 |     0.866129 | 0.110414  |      0.876401 |     0.454638 |           0.282879 |
| RF_CORR_depth16_leaf10_trees200_unbalanced |       0.914017 |     0.871959 | 0.0420578 |      0.86567  |     0.426943 |           0.104399 |
| RF_CORR_depth16_leaf10_trees300_unbalanced |       0.914528 |     0.872439 | 0.0420895 |      0.865588 |     0.426989 |           0.104451 |
| RF_CORR_depth14_leaf5_trees200_unbalanced  |       0.902158 |     0.866746 | 0.0354124 |      0.861265 |     0.420362 |           0.10549  |

Correlation version-ში საინტერესო იყო unbalanced მოდელების ქცევა. მათ balanced მოდელებზე დაბალი holdout AUC ჰქონდათ, მაგრამ gap გაცილებით მცირე იყო და log loss ბევრად უკეთესი გამოვიდა. მაგალითად, `depth16` unbalanced მოდელს ჰქონდა gap მხოლოდ **0.0421**, მაშინ როცა balanced deep მოდელებში gap დაახლოებით **0.11** იყო.

ეს ნიშნავს, რომ balanced Random Forest fraud კლასზე უფრო აგრესიულად სწავლობდა, რაც AUC-ს ზრდიდა, მაგრამ probability calibration-ს და generalization stability-ს აუარესებდა.

#### Random Forest-ის შეჯამება

| preprocessing             | run_name                                                  |   cv_train_auc |   cv_val_auc |   auc_gap |   holdout_auc |   holdout_ap |   holdout_log_loss |
|:--------------------------|:----------------------------------------------------------|---------------:|-------------:|----------:|--------------:|-------------:|-------------------:|
| Baseline_Preprocessing    | RF_depthNone_split5_leaf10_trees300_featuressqrt_balanced |       0.997242 |     0.884028 |  0.113214 |      0.893314 |     0.487518 |           0.192474 |
| IV_Preprocessing          | RandomForest_Final_FullPipeline                           |       0.98108  |     0.876215 |  0.104865 |      0.884452 |     0.475683 |           0.275212 |
| Correlation_Preprocessing | RF_CORR_depth18_leaf10_trees200_balanced                  |       0.976543 |     0.866129 |  0.110414 |      0.876401 |     0.454638 |           0.282879 |

Random Forest-მა Logistic Regression-ზე ბევრად უკეთესი შედეგი აჩვენა. საუკეთესო holdout ROC-AUC იყო:
```text
0.8933
```
საბოლოოდ Random Forest ძლიერი intermediate მოდელი აღმოჩნდა, მაგრამ საუკეთესო საბოლოო მოდელად არ შეირჩა.


---

### AdaBoost

AdaBoost გავტესტე როგორც boosting-ის უფრო მარტივი მოდელი XGBoost-მდე. Base estimator-ად გამოყენებული იყო მცირე სიღრმის `DecisionTreeClassifier`, რადგან AdaBoost ეტაპობრივად ამატებს სუსტ მოდელებს და ყოველი შემდეგი estimator ცდილობს წინა შეცდომების გამოსწორებას.

AdaBoost-ზე ექსპერიმენტები ჩავატარე მხოლოდ **IV preprocessing** ვერსიით, რადგან training საკმაოდ ნელი აღმოჩნდა. ამიტომ Random Forest-ის მსგავსად baseline/correlation ვერსიების ფართო შედარება აღარ გამიკეთებია.

ჰიპერპარამეტრებში გაიტესტა:
- `n_estimators`
- `learning_rate`
- base estimator-ის `max_depth`
- base estimator-ის `class_weight`

| run_name | holdout_roc_auc | preprocessing_type |
|---|---:|---|
| AdaBoost_FAST_IV_depth2_trees100_lr0.2_unbalanced | 0.8496 | IV_Preprocessing |
| AdaBoost_FAST_IV_depth1_trees100_lr0.5_unbalanced | 0.8441 | IV_Preprocessing |
| AdaBoost_FAST_IV_depth1_trees50_lr0.5_unbalanced | 0.8401 | IV_Preprocessing |
| AdaBoost_FAST_IV_depth2_trees100_lr0.2_balanced | 0.7004 | IV_Preprocessing |

საუკეთესო AdaBoost configuration იყო:

```text
n_estimators = 100
learning_rate = 0.2
base estimator max_depth = 2
class_weight = None
```

ამ მოდელზე დამატებით გავუშვი TimeSeriesCV:
| run_name                              |   cv_train_roc_auc |   cv_val_roc_auc  |   roc_auc_gap  |
|:--------------------------------------------------|---------------------------:|-------------------------:|----------------------:|
| AdaBoost_Finalist_TimeSeriesCV                    |                   0.879107 |                  0.85732 |             0.0217867 |

CV gap საკმაოდ მცირე იყო, რაც ნიშნავს, რომ AdaBoost Random Forest-ის ღრმა მოდელებთან შედარებით ნაკლებად overfit-დებოდა. თუმცა მისი holdout ROC-AUC მაინც დაბალი იყო Random Forest-სა და XGBoost-ზე.

საბოლოო AdaBoost full pipeline-ის შედეგი:
| Metric | Value |
|---|---:|
| Holdout ROC-AUC | 0.8496 |
| Average Precision | 0.3515 |
| Log Loss | 0.2644 |
| F1 | 0.2984 |
| Precision | 0.8163 |
| Recall | 0.1826 |

<img width="428" height="265" alt="image" src="https://github.com/user-attachments/assets/d4195314-b7ee-497a-94c1-2d506a9d29d8" />

AdaBoost-ის შედეგი საინტერესო იყო precision/recall trade-off-ის მხრივ. Fraud კლასზე precision ძალიან მაღალი გამოვიდა, მაგრამ recall დაბალი იყო. ანუ მოდელი fraud-ს იშვიათად ნიშნავდა, თუმცა როცა ნიშნავდა, ხშირად სწორად ნიშნავდა. გარკვეულწილად მოსალოდნელია მაღალი precision inherent class imbalance-ის გამო.

Confusion matrix-ის მიხედვით false positive-ები ძალიან ცოტა იყო, მაგრამ fraud შემთხვევების დიდი ნაწილი გამორჩა. ამიტომ AdaBoost შეიძლება ჩაითვალოს conservative fraud detector-ად, მაგრამ ამ competition-ის მთავარი მეტრიკისთვის ის საკმარისად ძლიერი არ აღმოჩნდა.

Balanced base estimator-მა ყველაზე ცუდი შედეგი აჩვენა. სავარაუდოდ, class balancing-მა AdaBoost ზედმეტად აგრესიული გახადა fraud კლასის მიმართ და decision boundary გააუარესა. ამიტომ საუკეთესო AdaBoost მოდელი unbalanced configuration აღმოჩნდა.

საბოლოოდ AdaBoost არ შეირჩა საუკეთესო მოდელად.

---

### XGBoost

XGBoost იყო ყველაზე მნიშვნელოვანი მოდელი ამ პროექტში, gradient boost-მა უკეთესი შედეგები დადო ვიდრე linear models ან Random Forest.

XGBoost-ში გამოვიყენე `early_stopping` მექანიზმი, რის გამოც დამატებითი eval_set-ი შევქმენი ხელით ვალიდაციის გამოყოფით (time sorted by TransactionDT), რათა training არ გაგრძელებულიყო მაშინ, როცა validation score აღარ უმჯობესდებოდა. ეს განსაკუთრებით მნიშვნელოვანი იყო, რადგან `n_estimators` მაღალი მქონდა დაყენებული და რეალურად მოდელი საუკეთესო იტერაციაზე უნდა გაჩერებულიყო.

ჰიპერპარამეტრებში გაიტესტა:
- `n_estimators`
- `max_depth`
- `learning_rate`
- `min_child_weight`
- `subsample`
- `colsample_bytree`
- `reg_lambda`
- `reg_alpha`
- `scale_pos_weight`

პირველ ეტაპზე XGBoost გავტესტე baseline preprocessing-ით. ამ ვერსიაში გამოყენებული იყო ძირითადი cleaning, feature engineering, near-zero variance filtering და tree-friendly preprocessing. საწყისი XGBoost შედეგები უკვე უკეთესი იყო Logistic Regression-ზე, AdaBoost-ზე და Random Forest-ის უმეტესი კონფიგურაციით.

Baseline XGBoost:
| run_name                                   |   holdout_auc |   holdout_ap |   holdout_log_loss |   holdout_f1 |   holdout_precision |   holdout_recall |
|:-------------------------------------------|--------------:|-------------:|-------------------:|-------------:|--------------------:|-----------------:|
| XGB_BASELINE_depth5_lr0.03_child20_spw5.24 |      0.890108 |     0.481369 |           0.150115 |     0.437305 |            0.389177 |         0.499016 |
| XGB_BASELINE_depth5_lr0.03_child10_spw5.24 |      0.889848 |     0.488086 |           0.138776 |     0.454586 |            0.419335 |         0.496309 |
| XGB_BASELINE_depth4_lr0.05_child5_spw5.24  |      0.889257 |     0.477762 |           0.150656 |     0.437925 |            0.385479 |         0.50689  |
| XGB_BASELINE_depth4_lr0.05_child5_spw1.00  |      0.888374 |     0.474429 |           0.100636 |     0.449353 |            0.626209 |         0.350394 |

ამის შემდეგ დავამატე უფრო განვითარებული feature engineering pipeline `FE_V2_Preprocessing`.

FE_V2 preprocessing-მა baseline-თან შედარებით შედეგი გააუმჯობესა:
| run_name                                 |   holdout_auc |   holdout_ap |   holdout_log_loss |   holdout_f1 |   holdout_precision |   holdout_recall |
|:-----------------------------------------|--------------:|-------------:|-------------------:|-------------:|--------------------:|-----------------:|
| XGB_FE_V2_depth6_lr0.03_child10_spw6.55  |      0.904689 |     0.504395 |          0.139654  |     0.47545  |            0.438331 |         0.519439 |
| XGB_FE_V2_depth6_lr0.03_child10_spw5.24  |      0.903711 |     0.504446 |          0.130098  |     0.47687  |            0.46145  |         0.493356 |
| XGB_FE_V2_depth7_lr0.05_child10_spw5.24  |      0.903098 |     0.505948 |          0.128253  |     0.488727 |            0.481605 |         0.496063 |
| XGB_FE_V2_depth5_lr0.03_child20_spw5.24  |      0.900998 |     0.49281  |          0.147018  |     0.458611 |            0.407216 |         0.524852 |
| XGB_FE_V2_depth5_lr0.03_child10_spw5.24  |      0.900889 |     0.507913 |          0.137966  |     0.471774 |            0.432977 |         0.518209 |
| XGB_FE_V2_depth6_lr0.03_child30_spw1.00  |      0.899525 |     0.497664 |          0.0960295 |     0.477927 |            0.682815 |         0.367618 |
| XGB_FE_V2_depth4_lr0.03_child10_spw3.93  |      0.898613 |     0.487765 |          0.13991   |     0.457578 |            0.419713 |         0.502953 |
| XGB_FE_V2_depth6_lr0.015_child40_spw1.00 |      0.898126 |     0.495182 |          0.0967283 |     0.476647 |            0.680987 |         0.366634 |
| XGB_FE_V2_depth6_lr0.015_child10_spw1.00 |      0.895752 |     0.49081  |          0.0957893 |     0.455516 |            0.731628 |         0.330709 |
| XGB_FE_V2_depth4_lr0.015_child5_spw1.00  |      0.891664 |     0.484618 |          0.0973602 |     0.453629 |            0.715042 |         0.332185 |


ეს აჩვენებს, რომ დამატებითმა feature engineering-მა მოდელს მისცა ახალი სასარგებლო სიგნალები, თუმცა ძირითადი predictive power კვლავ მოდიოდა raw/anonymized `V`, `C`, `card` და address-related feature-ებიდან.

შემდეგ ეტაპზე გამოვიყენე parameter tuning. ხელით შერჩეული რამდენიმე configuration-ის შემდეგ გავტესტე უფრო ფართო parameter region, random sampling-ით

საბოლოო tuned XGBoost full pipeline-ის შედეგებია:

| run_name                                   |   holdout_auc |   holdout_ap |   holdout_log_loss |   holdout_f1 |   holdout_precision |   holdout_recall |     auc_gap |
|:-------------------------------------------|--------------:|-------------:|-------------------:|-------------:|--------------------:|-----------------:|------------:|
| XGBoost_FE_V2_Final_FullPipeline_Tuned     |      0.924402 |     0.573057 |          0.120048  |     0.539702 |            0.528013 |         0.551919 |   0.0563119 |
| XGBoost_FE_V2_Final_FullPipeline           |      0.918555 |     0.549461 |          0.121532  |     0.524518 |            0.520077 |         0.529035 |   0.0485837 |
| XGBoost_Final_FullPipeline                 |      0.910955 |     0.532247 |          0.127728  |     0.504498 |            0.498558 |         0.510581 |   0.0447638 |


---

## საუკეთესო მოდელი

საუკეთესო მოდელია XGBoost_FE_V2_Final_FullPipeline_Tuned - ჰქონდა საუკეთესო ROC-AUC და Average Precision ყველა დატესტილ მოდელს შორის. AUC gap იყო დაახლოებით `0.056`, რაც მიუთითებს მცირედ overfitting-ზე, თუმცა Random Forest-ის ღრმა მოდელებთან შედარებით შედეგი მნიშვნელოვნად დაბალანსებული იყო.

<img width="1110" height="129" alt="image" src="https://github.com/user-attachments/assets/8f1ecefa-1fe3-4e6b-9abe-f303a501b125" />

---

### MLflow ექსპერიმენტები DagsHub-ზე

ყველა run დარეგისტრირებულია: [Dagshub](https://dagshub.com/myvari/IEEE-CIS-Fraud-Detection)

თითოეულ run-ში დაილოგა:

- ყველა ჰიპერპარამეტრი
- Train / Val / Test(holdout) მეტრიკები (roc-auc, ap, precision, log_loss, f1, recall, gap)
- დატრენინგებული მოდელის + preprocessing არტეფაქტი (.joblib)

---
მთავარი ნაწილი ამ პროექტის შეადგენდა მონაცემთა დიდი რაოდენობის სწორ ანალიზს და competition-specific feature engineering-ი, რომელიც დამალულ სტრუქტურებს აღმოაჩენინებს მოდელებს.
კერძოდ დიდი ყურადღება უნდა ეთმობოდეს uid სტილის feature-ებს რომლებიც ზუსტად შეძლებენ კარტის მფლობელობის განსაზღვრაას და ქცევების გამოხატვას, 
შესაძლოა group მონაცემების დახმარებით ძლიერ uid იდენტიფიკატორებზე საგრძნობლად გაზარდოს მაჩვენებელი. uid ამ შემთხვევაში იძლევა კარგ groupings, რომელსაც შემდეგ tree-based მოდელები უფრო უკეთესად დაყოფენ.



