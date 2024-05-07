# Networks_Intrusion

## Dataset Overview and Location

The Network Intrusion Dataset is a unified collection of network traffic data, consolidating various smaller datasets into a comprehensive resource for network intrusion detection research. It includes a wide range of network flows from different setups and attack scenarios. This dataset has 11,994,893 records, with 76.77% being benign flows and 23.23% classified as attacks. It has modified attack categories for uniformity, which are: DoS, DDoS, brute force, and SQL injection.

### Dataset Link: 
Network Intrusion Detection: https://www.kaggle.com/datasets/aryashah2k/nfuqnidsv2-network-intrusion-detection-dataset 

## Features

- **IPV4_SRC_ADDR** (IPv4 source address)
- **L4_SRC_PORT** (IPv4 destination address)
- **IPV4_DST_ADDR** (IPv4 source port number)
- **L4_DST_PORT** (IPv4 destination port number)
- **PROTOCOL** (IP protocol identifier byte)
- **L7_PROTO** (Layer 7 protocol (numeric))
- **IN_BYTES** (Incoming number of bytes)
- **IN_PKTS** (Incoming number of packets)
- **OUT_BYTES** (Outgoing number of bytes)
- **OUT_PKTS** (Outgoing number of packets)
- **TCP_FLAGS** (Cumulative of all TCP flags)
- **CLIENT_TCP_FLAGS** (Cumulative of all client TCP flags)
- **SERVER_TCP_FLAGS** (Cumulative of all server TCP flags)
- **FLOW_DURATION_MILLISECONDS** (Flow duration in milliseconds)
- **DURATION_IN** (Client to Server stream duration (msec))
- **DURATION_OUT** (Client to Server stream duration (msec))
- **MIN_TTL** (Min flow TTL)
- **MAX_TTL** (Max flow TTL)
- **LONGEST_FLOW_PKT** (Longest packet (bytes) of the flow)
- **SHORTEST_FLOW_PKT** (Shortest packet (bytes) of the flow)
- **MIN_IP_PKT_LEN** (Len of the smallest flow IP packet observed)
- **MAX_IP_PKT_LEN** (Len of the largest flow IP packet observed)
- **SRC_TO_DST_SECOND_BYTES** (Src to dst Bytes/sec)
- **DST_TO_SRC_SECOND_BYTES** (Dst to src Bytes/sec)
- **RETRANSMITTED_IN_BYTES** (Number of retransmitted TCP flow bytes (src->dst))
- **RETRANSMITTED_IN_PKTS** (Number of retransmitted TCP flow packets (src->dst))
- **RETRANSMITTED_OUT_BYTES** (Number of retransmitted TCP flow bytes (dst->src))
- **RETRANSMITTED_OUT_PKTS** (Number of retransmitted TCP flow packets (dst->src))
- **SRC_TO_DST_AVG_THROUGHPUT** (Src to dst average thpt (bps))
- **DST_TO_SRC_AVG_THROUGHPUT** (Dst to src average thpt (bps))
- **NUM_PKTS_UP_TO_128_BYTES** (Packets whose IP size <= 128)
- **NUM_PKTS_128_TO_256_BYTES** (Packets whose IP size > 128 and <= 256)
- **NUM_PKTS_256_TO_512_BYTES** (Packets whose IP size > 256 and <= 512)
- **NUM_PKTS_512_TO_1024_BYTES** (Packets whose IP size > 512 and <= 1024)
- **NUM_PKTS_1024_TO_1514_BYTES** (Packets whose IP size > 1024 and <= 1514)
- **TCP_WIN_MAX_IN** (Max TCP Window (src->dst))
- **TCP_WIN_MAX_OUT** (Max TCP Window (dst->src))
- **ICMP_TYPE** (ICMP Type * 256 + ICMP code)
- **ICMP_IPV4_TYPE** (ICMP Type)
- **DNS_QUERY_ID** (DNS query type)
- **DNS_QUERY_TYPE** (DNS_TTL_ANSWER)
- **TTL of the first A record (if any)**
- **FTP_COMMAND_RET_CODE** (FTP client command return code)
- **Label**
- **Attack** (attack type)
- **Dataset** (Datapoint source)

## Machine Learning Model

This project aims to develop a machine-learning model to distinguish between benign and malicious network flows and classify attack types. The label column identifies whether a flow is benign or malicious and specifies the attack category. I will most likely employ a Random Forest classifier model, but test out logistic regression and a K-means unsupervised model.

## Data Acquisition and Preparation

1. **Downloaded the Kaggle API Token File:** Generated a new API token from Kaggle account settings under the API section, which prompted the download of the `kaggle.json` file to my local system.
2. **Located Dataset on Kaggle:** Found the required network intrusion detection dataset for the project. 
3. **Created a Virtual Machine in Google Cloud Platform:** Created a virtual instance following all default requirements (Iowa central server, E2 instance, size e2 medium) except changing boot disk size to 50GB and renaming the instance to “network-intrusion project-v1”.
4. **Connected to the Linux Instance with a Secure Shell:** Accessed the VM by clicking the SSH button next to the instance name in the GCP console, establishing a secure connection.
5. **Uploaded and Copied the Kaggle API Token File:** Transferred the `kaggle.json` file to the VM and placed it in the `~/.kaggle` directory to authenticate with Kaggle's API.
6. **Installed Additional Software and Set Up Python Environment:** On the VM, installed necessary software like Python and kaggle using `pip install`. Set up a virtual environment for Python development.
7. **Downloaded Dataset from Kaggle:** Utilized the Kaggle API to download the network intrusion detection dataset directly to the VM.
8. **Unzipped the Archive File:** Extracted the dataset files from the downloaded zip file using the unzip command.
9. **Copied Files to Google Cloud Storage:** Moved the dataset files from the VM to Google Cloud Storage for later access, and organized them into necessary folders like landing, cleaned, code, models, and trusted.
10. **Stopped and Deleted the Linux Instance:** To manage resources and costs, stopped the VM when tasks were completed and then deleted the instance from the GCP console.

## EDA and Data Cleaning

While cleaning the data, I made several key observations:
- The data was already cleaned as there were no null or duplicate values.
- A few data types that weren't int64, such as object data types and float, were converted into int data types to standardize them.
- Given the project's focus on anomaly analysis, max or min values were crucial and couldn't be disregarded.
- The dataset had a potential issue of overfitting due to its 46 columns; hence, some columns needed removal to simplify the model.
- The label had a class imbalance with more benign values than malicious, which needed addressing.
- Removed columns deemed insignificant to the label such as 'SRC_TO_DST_SECOND_BYTES', 'DST_TO_SRC_SECOND_BYTES', 'FTP_COMMAND_RET_CODE', 'ICMP_IPV4_TYPE' after a heatmap analysis.
- Chunking of data was necessary during processing to prevent system crashes due to resource limitations.

## Feature Engineering

- Dropped the two address columns on the professor's suggestion as they were private IPs and had little relevance.
- Initialized a Spark session and read the cleaned data from the GCS bucket.
- Encoded two object data type features, dataset and attack, in addition to converting float to int.
- Employed vector assemblers to combine features into a single vector and normalized them through MinMaxScaler for modeling.
- Utilized RandomForestClassifier for modeling and integrated vector assembler and scalar into a Pipeline.
- Performed hyperparameter tuning using TrainValidationSplit to optimize the model based on the ROC metric.
- Evaluated the trained model on the test dataset to produce predictions and calculated the ROC AUC metric.

## Done Visualization

- **Random Forest ROC Curve:** Nearly perfect ROC curve indicating the model is very good at predicting positive and negative classes.Very few false positives and false negatives.
- **Logistic Regression ROC Curve:** Good ROC curve indicating the model is good at predicting positive and negative classes but some false positives and some false negatives do appear.
- **Precision-Recall for Random Forest:** Nearly perfect curve indicating high precision and recall.
- **Heatmap:** Used to check for multicollinearity among features which might affect the model's performance.According to the heatmap, there are a few variables that correlate with one another that should be addressed if I were to revisit the model.

## Conclusion

The Network Intrusion Detection System project leverages machine learning to differentiate between benign and malicious network flows while categorizing various types of network attacks. Utilizing the Network Intrusion Dataset from Kaggle, which comprises 11,994,893 records of network traffic data including DoS, DDoS, brute force, and SQL injection attacks, the project pipeline included: data cleaning to standardize data types and address class imbalances, feature engineering to encode categorical data and normalize features, and the development of a RandomForestClassifier model. The Logistic Regression model was tested as well but not as accurate as the random forest model. Hyperparameter tuning was then employed for the random forest model through cross-validation. After this was done, the random forest’s overfitting issue was addressed and the performance with accuracy, precision, recall, and F1 scores were all above 0.98, outperforming a baseline Logistic Regression model. The visualization analysis included ROC and Precision-Recall curves to validate model effectiveness and a heatmap to identify multicollinearity among features. These visual tools confirmed the model's high precision and recall, making it well-suited for environments where false positives are particularly undesirable. However, it was detected that multi-collinearity could interfere with some of the models. So to improve later on, I can remove some more variables to prevent this

