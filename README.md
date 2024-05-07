# Networks_Intrusion

## Dataset Overview and Location

The Network Intrusion Dataset is a unified collection of network traffic data, consolidating various smaller datasets into a comprehensive resource for network intrusion detection research. It includes a wide range of network flows from different setups and attack scenarios. This dataset has 11,994,893 records, with 76.77% being benign flows and 23.23% classified as attacks. It has modified attack categories for uniformity, which are: DoS, DDoS, brute force, and SQL injection.

### Dataset Link: 
Network Intrusion Detection 

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

This project aims to develop a machine-learning model to distinguish between benign and malicious network flows and classify attack types. The label column identifies whether a flow is benign or malicious and specifies the attack category. I will most likely employ a Random Forest classifier model, but test out a decision tree and a K-means unsupervised model.
