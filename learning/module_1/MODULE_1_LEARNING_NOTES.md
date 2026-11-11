# MODULE 1: COMPUTER NETWORKING FOUNDATIONS & THE CICIoT2023 DATASET
**Course**: Capstone Research Mentorship & Technical Training  
**Author / Mentor**: Antigravity Technical Assistant & Research Mentor  
**Date**: November 2026  
**Audience**: Senior B.Tech Researcher (Self-Defense & Publication Focus)  
**Status**: Authoritative Learning Document — Module 1 Complete

---

## Table of Contents
1. [Pedagogical Objective & Research Context](#1-pedagogical-objective--research-context)
2. [Section 1: Computer Networking Foundations](#2-section-1-computer-networking-foundations)
   - 2.1 What is a Computer Network?
   - 2.2 Network Topologies and Device Addressing (MAC vs. IP)
   - 2.3 Ports, Sockets, and End-to-End Multiplexing
   - 2.4 The TCP/IP Protocol Stack vs. OSI Reference Model
   - 2.5 Anatomy of a Network Packet (Ethernet, IP, TCP/UDP Headers)
   - 2.6 The Mechanics of IoT Device Communication
3. [Section 2: Network Traffic, Flows, and Tabular Transformation](#3-section-2-network-traffic-flows-and-tabular-transformation)
   - 3.1 Packets vs. Connections vs. Sessions vs. Flows
   - 3.2 The Fundamental 5-Tuple Definition
   - 3.3 Bidirectional Flows (Biflows) vs. Unidirectional Flows
   - 3.4 Sliding-Window Flow Feature Aggregation
   - 3.5 Packet Capture (PCAP) to Tabular CSV Feature Extraction
4. [Section 3: Intrusion Detection Systems (IDS) Fundamentals](#4-section-3-intrusion-detection-systems-ids-fundamentals)
   - 4.1 IDS Taxonomy: NIDS vs. HIDS
   - 4.2 Detection Paradigms: Signature-Based vs. Anomaly-Based (ML)
   - 4.3 Classification Granularity: Binary vs. 8-Category vs. 34-Class
   - 4.4 Machine Learning Workflow in Network Security
5. [Section 4: The CICIoT2023 Dataset — Deep Scientific Audit](#5-section-4-the-ciciot2023-dataset--deep-scientific-audit)
   - 5.1 Origin, Topology, and Testbed Architecture (Neto et al., 2023)
   - 5.2 The 39 Numerical Features: Complete Physical & Mathematical Dictionary
   - 5.3 The 34 Attack Labels & 8 Functional Attack Profiles
   - 5.4 The Mechanics of Class Imbalance in CICIoT2023
   - 5.5 Scientific Boundaries: What CICIoT2023 Can and Cannot Establish
6. [Section 5: Mapping Concepts to Repository Implementation](#6-section-5-mapping-concepts-to-repository-implementation)
7. [Section 6: Comprehensive Networking & IDS Glossary](#7-section-6-comprehensive-networking--ids-glossary)
8. [Section 7: Self-Defense Checklist for Oral Examination](#8-section-7-self-defense-checklist-for-oral-examination)

---

## 1. Pedagogical Objective & Research Context

The overarching objective of this capstone research is to detect and classify cyberattacks targeting Internet of Things (IoT) ecosystems using the benchmark dataset **CICIoT2023**.

However, machine learning algorithms do not operate on abstract mathematical tensors in a vacuum; they process quantitative measurements derived from physical electrical, optical, and radio signals traveling through copper cables, fiber optics, and wireless airwaves. When a classifier makes a decision—for example, predicting that a flow is `DDOS-SYN_FLOOD` rather than `BENIGN`—it relies entirely on statistical shifts in packet headers and timing dynamics.

To design, evaluate, debug, and defend an IoT Intrusion Detection System (NIDS), you must understand the underlying networking mechanics from first principles:
- Why do packet headers exist, and what information do they carry?
- Why do IoT devices behave differently from general-purpose desktop computers?
- How is raw packet telemetry converted into a 39-feature tabular row?
- Why does class imbalance occur naturally in network traffic?
- Why do certain attacks look numerically identical to normal traffic when certain packet headers (like IP addresses) are removed?

---

## 2. Section 1: Computer Networking Foundations

### 2.1 What is a Computer Network?
* **Simple Definition**: A computer network is an interconnected collection of autonomous computing devices (nodes) that exchange digital information using a shared communication medium and a standardized set of rules (protocols).
* **Why it Exists**: Without networks, computational systems operate in isolation. Networks enable resource sharing, distributed computation, real-time sensing, and remote actuation.
* **IoT Context**: In an IoT smart home, an ambient temperature sensor node does not possess a screen, hard drive, or user interface. It exists solely to sample environmental telemetry and transmit those measurements across a local network to an IoT gateway or cloud MQTT broker.
* **Relevance to Attacks**: The open nature of network transmission means that an attacker who gains physical or logical access to the network medium can intercept, alter, spoof, or overwhelm the communications of legitimate devices.

---

### 2.2 Network Topologies and Device Addressing (MAC vs. IP)

Every networked device operates with multiple layers of identification:

```
[ Physical Link / Hop-by-Hop ] ---------> MAC Address (Layer 2, Data Link)
                                          Example: b8:27:eb:bf:9d:11 (Raspberry Pi OUI)

[ Logical / End-to-End Routing ] -------> IP Address (Layer 3, Network)
                                          Example: 192.168.1.50 (IPv4) or 2001:db8::1 (IPv6)
```

1. **MAC Address (Media Access Control)**:
   - **Definition**: A 48-bit (6-byte) physical hardware address permanently assigned to a Network Interface Card (NIC) by the manufacturer.
   - **Representation**: Hexadecimal notation separated by colons (e.g., `B8:27:EB:5A:1C:89`). The first 24 bits represent the Organizationally Unique Identifier (OUI, identifying the vendor like Espressif, Raspberry Pi, or Apple); the last 24 bits are the unique device serial number.
   - **Scope**: Link-local only. A MAC address is used strictly to move frames between devices on the *same physical local area network (LAN)* or Ethernet switch. When a packet crosses a router, the source and destination MAC addresses are stripped and replaced with the next hop's MAC addresses.
   - **IoT Security Relevance**: Attackers execute **ARP Spoofing (`MITM-ARPSPOOFING`)** to poison the local address resolution cache of IoT sensors, fooling the sensor into sending its traffic to the attacker's MAC address instead of the legitimate router gateway.

2. **IP Address (Internet Protocol)**:
   - **Definition**: A logical numerical address assigned to a device interface to enable global routing across interconnected networks (the Internet).
   - **IPv4 vs IPv6**: IPv4 uses 32-bit addresses written in dotted-decimal format (e.g., `192.168.1.100`), providing $2^{32} \approx 4.29 \times 10^9$ addresses. IPv6 uses 128-bit hexadecimal addresses to handle the trillions of globally connected IoT nodes.
   - **Scope**: End-to-end. The IP header remains intact from the originating sender to the final destination across dozens of intermediate Internet routers.
   - **CICIoT2023 Dataset Fact**: In the raw PCAPs of CICIoT2023, 105 physical devices and attacking machines had real IP addresses (e.g., victim smart cameras, smart bulbs, attacking Kali Linux VMs). However, in the distributed 39-feature CSV files, **all IP addresses and MAC addresses were deliberately stripped** to prevent machine learning algorithms from trivial "identity memorization" (e.g., learning that IP `192.168.1.105` is always malicious).

---

### 2.3 Ports, Sockets, and End-to-End Multiplexing

A single IoT device may run multiple applications simultaneously (e.g., an HTTP web configuration server, an MQTT client sending sensor readings, and an SSH daemon for remote management).

* **Port Number**: A 16-bit unsigned integer ($0$ to $65,535$) that identifies a specific application process running on an operating system.
  - *Well-Known Ports ($0 - 1,023$)*: Reserved for core system protocols (e.g., Port 80 for HTTP, Port 443 for HTTPS, Port 53 for DNS, Port 22 for SSH, Port 23 for Telnet).
  - *Registered Ports ($1,024 - 49,151$)*: Vendor applications (e.g., Port 1883 for unencrypted MQTT, Port 8883 for secure MQTT over TLS).
  - *Dynamic / Ephemeral Ports ($49,152 - 65,535$)*: Temporary client ports opened dynamically when making an outbound request.
* **Network Socket**: The combination of an IP address and a Port number:
  $$\text{Socket} = (\text{IP Address}, \text{Port Number})$$
  Example: `192.168.1.50:80` defines a web server listening on internal device 50.
* **Relevance to CICIoT2023 Attacks**:
  - In `DICTIONARYBRUTEFORCE`, an adversary targets Port 22 (SSH) or Port 23 (Telnet), repeatedly sending thousands of credential guesses to crack default IoT passwords.
  - In `RECON-PORTSCAN`, an adversary systematically sends packets to ports 1 through 1,024 to discover which services are exposed on an IoT camera.

---

### 2.4 The TCP/IP Protocol Stack vs. OSI Reference Model

Network communications are structured into layered abstractions, where each layer provides a service to the layer above it and encapsulates the data received:

```
+------------------------------------+------------------------------------+
|       OSI 7-Layer Model            |       TCP/IP 4-Layer Model         |
+------------------------------------+------------------------------------+
| 7. Application Layer               |                                    |
| 6. Presentation Layer              | 4. Application Layer (HTTP, DNS)  |
| 5. Session Layer                   |                                    |
+------------------------------------+------------------------------------+
| 4. Transport Layer (TCP, UDP)      | 3. Transport Layer (TCP, UDP)      |
+------------------------------------+------------------------------------+
| 3. Network Layer (IP, ICMP)        | 2. Internet / Network Layer (IP)   |
+------------------------------------+------------------------------------+
| 2. Data Link Layer (Ethernet, ARP) | 1. Network Access / Link Layer     |
| 1. Physical Layer (Wi-Fi, 802.15.4)|    (Ethernet, 802.11 Wi-Fi, BLE)   |
+------------------------------------+------------------------------------+
```

#### Layer Responsibilities in IoT Security:
1. **Link Layer (Layer 1 & 2)**: Transmits raw bits over physical media and handles local frame delivery via MAC addresses.
   - Protocols: Ethernet (IEEE 802.3), Wi-Fi (802.11), Zigbee (802.15.4), Bluetooth Low Energy (BLE), ARP (Address Resolution Protocol).
   - Features in CICIoT2023: `ARP`, `LLC` (Logical Link Control).
2. **Internet Layer (Layer 3)**: Packages data into packets (datagrams), assigns logical IP addresses, and routes packets across router hops.
   - Protocols: IPv4, IPv6, ICMP (ping/error messages), IGMP (multicast group management).
   - Features in CICIoT2023: `Header_Length`, `Protocol Type`, `Time_To_Live`, `ICMP`, `IGMP`, `IPv`.
3. **Transport Layer (Layer 4)**: Manages process-to-process delivery, reliability, flow control, and congestion.
   - Protocols: TCP (Transmission Control Protocol) and UDP (User Datagram Protocol).
   - Features in CICIoT2023: `TCP`, `UDP`, and all TCP control flags (`fin_flag_number`, `syn_flag_number`, `rst_flag_number`, `psh_flag_number`, `ack_flag_number`).
4. **Application Layer (Layer 5/7)**: Formats end-user application payloads.
   - Protocols: HTTP, HTTPS, DNS, Telnet, SMTP, SSH, IRC, MQTT, CoAP.
   - Features in CICIoT2023: Protocol indicators `HTTP`, `HTTPS`, `DNS`, `Telnet`, `SMTP`, `SSH`, `IRC`.

---

### 2.5 Anatomy of a Network Packet (Ethernet, IP, TCP/UDP Headers)

When an application transmits data (such as a 100-byte temperature payload), the operating system encapsulates it in nested protocol envelopes:

```
+-------------------------------------------------------------------------------+
| Ethernet Frame (14 bytes)                                                     |
| [ Dest MAC (6B) | Src MAC (6B) | EtherType (2B): 0x0800 = IPv4 ]             |
|   +-------------------------------------------------------------------------+ |
|   | IPv4 Header (20 to 60 bytes)                                            | |
|   | [ Version(4b)|IHL(4b)|ToS(8b)|Total Length(16b)|TTL(8b)|Protocol(8b)...] | |
|   |   +-------------------------------------------------------------------+ | |
|   |   | TCP Header (20 to 60 bytes)                                       | | |
|   |   | [ Src Port (16b) | Dst Port (16b) | Seq (32b) | Ack (32b)         | | |
|   |   |   Flags: URG | ACK | PSH | RST | SYN | FIN | Window Size (16b) ]  | | |
|   |   |   +-------------------------------------------------------------+ | | |
|   |   |   | Application Payload (e.g., HTTP GET /sensor/temp)           | | | |
|   |   |   +-------------------------------------------------------------+ | | |
|   |   +-------------------------------------------------------------------+ | |
|   +-------------------------------------------------------------------------+ |
+-------------------------------------------------------------------------------+
```

#### Detailed Breakdown of Critical Header Fields Present in CICIoT2023:

1. **IPv4 Protocol Field (`Protocol Type`)**:
   - An 8-bit field specifying which Layer 4 protocol payload is encapsulated inside the IP packet.
   - Key numeric identifiers:
     - `1` = **ICMP** (Internet Control Message Protocol, used in ping and ping floods)
     - `2` = **IGMP** (Internet Group Management Protocol, used in multicast)
     - `6` = **TCP** (Transmission Control Protocol)
     - `17` = **UDP** (User Datagram Protocol)
     - `47` = **GRE** (Generic Routing Encapsulation, used in Mirai botnet tunnel attacks)
   - *In our dataset*: Look at `Protocol Type`. You will see rows with values `6` (TCP), `17` (UDP), and `47` (GRE)!

2. **IPv4 Time-To-Live (`Time_To_Live` / TTL)**:
   - An 8-bit integer initialized by the sending operating system (typically `64` in Linux, Android, and IoT firmware; `128` in Windows; `255` in Cisco network gear).
   - Each intermediate router decrements this value by $1$. If $\text{TTL} = 0$, the router drops the packet and sends an ICMP Time Exceeded packet back. This prevents lost packets from looping endlessly across misconfigured routers.
   - *Why it matters to NIDS*: In volumetric spoofing attacks (`DDOS-SYNONYMOUSIP_FLOOD`), attackers generate random fake IP addresses from all over the world. However, if the attack script runs on a local machine, the TTL arrives with a constant or mismatched value, exposing the spoofing anomaly!

3. **TCP Control Flags**:
   - The TCP header contains 9 single-bit flags used to manage connection establishment, data push, flow termination, and congestion:
     - `SYN` (Synchronize): Initiates a new connection handshake.
     - `ACK` (Acknowledgment): Confirms receipt of previously transmitted sequence bytes.
     - `FIN` (Finish): Gracefully terminates a connection.
     - `RST` (Reset): Abruptly aborts a connection due to an error or rejected port.
     - `PSH` (Push): Instructs the receiver to deliver buffered data immediately to the application without waiting for a full buffer.
     - `ECE` (Explicit Congestion Notification Echo): Signals network router congestion.
     - `CWR` (Congestion Window Reduced): Confirms receipt of ECE.

#### The TCP Three-Way Handshake vs. SYN Flood Attack:

```
Legitimate TCP Handshake:
Client                               Server
  | ------------ SYN (Seq=x) ----------> |  (Client requests connection)
  | <------- SYN-ACK (Seq=y, Ack=x+1) -- |  (Server allocates TCB memory buffer)
  | ------------ ACK (Ack=y+1) --------> |  (Connection ESTABLISHED, data begins)

SYN Flood Attack (DOS-SYN_FLOOD / DDOS-SYN_FLOOD):
Attacker (Spoofed IP)                 Server
  | ------------ SYN -----------------> |  Server creates half-open connection entry in RAM
  | ------------ SYN -----------------> |  Server creates half-open connection entry in RAM
  | ------------ SYN -----------------> |  Server creates half-open connection entry in RAM
  | ------------ SYN -----------------> |  ...
  |   (Attacker NEVER sends ACK!)       |  Transmission Control Block (TCB) table EXHAUSTED!
  |                                     |  Result: Legitimate IoT clients CANNOT connect!
```

*In CICIoT2023*: When a SYN flood occurs, the feature `syn_flag_number` spikes to $\approx 1.0$, while `ack_flag_number` drops to $\approx 0.0$, and `Rate` spikes from normal rates ($<50$ packets/sec) to tens of thousands of packets per second.

---

### 2.6 The Mechanics of IoT Device Communication

Why do we need specialized IoT Intrusion Detection rather than standard IT network intrusion detection?

1. **Constrained Hardware Profiles**:
   - Traditional IT servers have multi-core CPUs, gigabytes of RAM, and active operating system patching.
   - IoT devices (e.g., smart plugs, temperature sensors, IP cameras) are powered by low-power 8-bit or 32-bit microcontrollers (e.g., ESP8266, ESP32, ARM Cortex-M) with only $64\text{ KB}$ to $512\text{ KB}$ of RAM and no local firewall.
2. **Deterministic, Predictable Traffic**:
   - Human network traffic (browsing the web, streaming YouTube) is bursty, highly variable, and unpredictable.
   - Legitimate IoT traffic is strictly periodic and deterministic: an environmental sensor wakes up every 60 seconds, connects to an MQTT broker, transmits a 50-byte JSON telemetry payload, and returns to deep sleep.
3. **Severe Asymmetry in Vulnerability**:
   - Because IoT devices cannot run local antivirus agents, an adversary who compromises an IoT device (e.g., through default credentials like `admin:admin`) can convert it into a silent **botnet zombie** (such as `MIRAI-GREIP_FLOOD` or `MIRAI-UDPPLAIN`).
   - Thousands of compromised IoT devices are then commanded in unison to execute massive distributed denial-of-service (DDoS) campaigns against critical infrastructure.

---

## 3. Section 2: Network Traffic, Flows, and Tabular Transformation

### 3.1 Packets vs. Connections vs. Sessions vs. Flows

It is essential to distinguish these four concepts clearly:

| Entity | Protocol Layer | Scope | Lifespan | Example |
| :--- | :--- | :--- | :--- | :--- |
| **Packet** | Layer 3 (Network) | Single atomic datagram traversing a wire | Microseconds | A single 1500-byte IP packet carrying TCP data. |
| **Connection** | Layer 4 (Transport) | Formal stateful relationship between two sockets | Seconds to Hours | A TCP connection bounded by `SYN` and `FIN`/`RST`. |
| **Session** | Layer 7 (Application) | An ongoing application-level dialog | Minutes to Days | An HTTP authenticated session or SSH login terminal. |
| **Flow** | Analytical construct | Set of packets sharing an identical key within a time window | Seconds | All packets sharing `(SrcIP, DstIP, SrcPort, DstPort, Protocol)`. |

---

### 3.2 The Fundamental 5-Tuple Definition

In network measurement and intrusion detection, a **Flow** is defined by a mathematical tuple called the **5-tuple**:

$$\text{5-Tuple} = \Big( \text{IP}_{\text{src}}, \quad \text{IP}_{\text{dst}}, \quad \text{Port}_{\text{src}}, \quad \text{Port}_{\text{dst}}, \quad \text{Protocol} \Big)$$

Where:
- $\text{IP}_{\text{src}}$: Source IP address (e.g., `192.168.1.100`)
- $\text{IP}_{\text{dst}}$: Destination IP address (e.g., `192.168.1.1`)
- $\text{Port}_{\text{src}}$: Source port number (e.g., `54210`)
- $\text{Port}_{\text{dst}}$: Destination port number (e.g., `80`)
- $\text{Protocol}$: Transport layer protocol number (`6` for TCP, `17` for UDP)

---

### 3.3 Bidirectional Flows (Biflows) vs. Unidirectional Flows

1. **Unidirectional Flow**:
   - Only aggregates packets traveling in one direction: $\text{Device A} \to \text{Device B}$.
   - Packets returned from $\text{Device B} \to \text{Device A}$ are treated as a completely separate flow.
2. **Bidirectional Flow (Biflow)**:
   - Groups both the forward direction ($\text{Client} \to \text{Server}$) and backward direction ($\text{Server} \to \text{Client}$) into a single unified flow record.
   - Symmetry property:
     $$\Big( \text{IP}_A, \text{IP}_B, \text{Port}_A, \text{Port}_B, \text{Proto} \Big) \equiv \Big( \text{IP}_B, \text{IP}_A, \text{Port}_B, \text{Port}_A, \text{Proto} \Big)$$
   - *Why Biflows Matter for IDS*: Security decisions require observing request-response symmetry. For instance, in a normal HTTP GET request, the client sends a small packet ($300\text{ bytes}$), and the server replies with a large HTML payload ($50,000\text{ bytes}$). If a client sends 10,000 requests and receives zero replies, it indicates a scanning probe or flood!

---

### 3.4 Sliding-Window Flow Feature Aggregation

How does a network monitoring engine convert high-speed packet streams into statistical numbers?

Instead of evaluating packets one-by-one (which is computationally impossible at gigabit speeds), network tools group packets into a **sliding observation window**.

Two common window types exist:
1. **Time-based Window**: Aggregates all packets observed over the last $T$ seconds (e.g., $T = 1.0\text{ s}$).
2. **Count-based Window**: Aggregates the last $W$ packets (e.g., $W = 100\text{ packets}$).

#### Mathematical Extraction of Flow Features:

Let an observation window contain $N$ packets with packet lengths $\{L_1, L_2, \dots, L_N\}$ arriving at timestamps $\{t_1, t_2, \dots, t_N\}$.

1. **Inter-Arrival Time (IAT)**:
   The time interval between successive packet arrivals:
   $$\Delta t_i = t_i - t_{i-1}, \quad \text{for } i \in \{2, 3, \dots, N\}$$
   The feature `IAT` in CICIoT2023 is the average inter-arrival time:
   $$\overline{\text{IAT}} = \frac{1}{N - 1} \sum_{i=2}^N \Delta t_i$$

2. **Packet Length Statistics**:
   - **Minimum Length (`Min`)**: $\min_{i} L_i$
   - **Maximum Length (`Max`)**: $\max_{i} L_i$
   - **Average Length (`AVG`)**: $\mu = \frac{1}{N} \sum_{i=1}^N L_i$
   - **Total Length Sum (`Tot sum`)**: $\sum_{i=1}^N L_i$
   - **Standard Deviation (`Std`)**: $\sigma = \sqrt{\frac{1}{N} \sum_{i=1}^N (L_i - \mu)^2}$
   - **Variance (`Variance`)**: $\sigma^2 = \text{Std}^2$

3. **Packet Transmission Rate (`Rate`)**:
   The number of packets observed per unit of time:
   $$\text{Rate} = \frac{N}{t_N - t_1} \quad \text{[packets / second]}$$

4. **Flag Proportions (`syn_flag_number`, `ack_flag_number`, etc.)**:
   Let $I_{\text{SYN}}(i) \in \{0, 1\}$ indicate whether packet $i$ has the SYN flag enabled.
   $$\text{syn\_flag\_number} = \frac{1}{N} \sum_{i=1}^N I_{\text{SYN}}(i)$$
   *(Notice: This explains why `syn_flag_number` in our CSV files is a float between 0.0 and 1.0, rather than an integer! A value of $0.99$ means $99\%$ of the packets in that 100-packet window had the SYN flag set!)*

---

### 3.5 Packet Capture (PCAP) to Tabular CSV Feature Extraction

The pipeline that created our raw dataset operates as follows:

```
[ Physical IoT Network & Kali Attackers ]
                  │
                  ▼ (tcpdump / Wireshark libpcap)
       [ Raw .pcap Trace Files ]  (8.66 GB of raw Ethernet frames)
                  │
                  ▼ (CICFlowMeter-V4 or custom DPKI extractor)
  * Flow segmentation via 5-tuple
  * Window buffering (packet count / time)
  * Statistical feature extraction (mean, std, rate, flags)
  * Header stripping (IPs, MACs, ports dropped)
                  │
                  ▼
  [ 63 Distributed CSV Files ]  (Merged01.csv to Merged63.csv)
  * 39 Numerical Features + 1 Ground-Truth Label
```

---

## 4. Section 3: Intrusion Detection Systems (IDS) Fundamentals

### 4.1 IDS Taxonomy: NIDS vs. HIDS

1. **Host-Based Intrusion Detection System (HIDS)**:
   - Installed directly on the host operating system.
   - Monitors internal system calls, file system modifications, CPU/RAM usage, kernel logs, and process execution (e.g., OSSEC, Wazuh).
   - *Limitation in IoT*: Microcontroller IoT devices (smart bulbs, smart thermostats) lack the memory, CPU headroom, and operating system capability to run a local HIDS agent.
2. **Network-Based Intrusion Detection System (NIDS)**:
   - Deployed at a central network choke point (e.g., an IoT gateway, home router, or switch SPAN/mirror port).
   - Passively inspects incoming and outgoing network traffic without requiring software agents on the individual IoT devices.
   - *Our Research Scope*: Our capstone project investigates a **NIDS** designed to protect an entire fleet of IoT devices from the network edge.

---

### 4.2 Detection Paradigms: Signature-Based vs. Anomaly-Based (ML)

```
+------------------------------------+------------------------------------+
| Signature-Based NIDS (Snort, Zeek) | Anomaly / ML-Based NIDS (Our Work) |
+------------------------------------+------------------------------------+
| Matches exact byte patterns or     | Learns normal vs. malicious        |
| known regular expressions.         | statistical flow distributions.    |
| Example: String match for          | Example: Detects sudden spike in   |
| 'UNION SELECT' in HTTP URI.        | Rate + SYN flags + low IAT.        |
| + Near-zero false alarm rate on    | + Capable of detecting zero-day    |
|   known attack signatures.         |   attacks and variant mutations.   |
| - Completely blind to unknown or   | - Higher false positive rate;      |
|   zero-day attack variants.        |   susceptible to distribution drift|
+------------------------------------+------------------------------------+
```

---

### 4.3 Classification Granularity: Binary vs. 8-Category vs. 34-Class

In our research, we evaluate models across three progressive tiers of classification granularity:

```mermaid
graph TD
    subgraph Level 1: Binary [2 Classes]
        B1[Benign]
        B2[Attack]
    end

    subgraph Level 2: Functional Categories [8 Classes]
        C1[Benign]
        C2[DDoS]
        C3[DoS]
        C4[Mirai]
        C5[Recon]
        C6[Spoofing]
        C7[Web]
        C8[BruteForce]
    end

    subgraph Level 3: Fine-Grained Profiles [34 Classes]
        P1[33 Specific Attack Signatures + Benign]
    end

    Level 1 --> Level 2 --> Level 3
```

1. **Binary Classification (2 Classes: `Benign` vs `Attack`)**:
   - *Objective*: Fast, high-throughput triage filter.
   - *Security Role*: Should legitimate traffic pass, or should a packet drop rule be triggered immediately?
2. **Functional Category Classification (8 Classes)**:
   - *Objective*: Actionable threat categorization (`DDoS`, `DoS`, `Mirai`, `Recon`, `Spoofing`, `Web`, `BruteForce`, `Benign`).
   - *Security Role*: Dictates the mitigation response. (e.g., A `Recon` port scan requires firewall IP blacklisting; a `DDoS` flood requires BGP flowspec rate-limiting; a `Web` SQL injection requires web application firewall blocking).
3. **Fine-Grained Attack Classification (34 Classes)**:
   - *Objective*: Exact signature attribution across all 33 individual attack profiles.
   - *Security Role*: Post-incident forensic attribution and vulnerability patching.

---

## 5. Section 4: The CICIoT2023 Dataset — Deep Scientific Audit

### 5.1 Origin, Topology, and Testbed Architecture (Neto et al., 2023)

* **Authors**: Vitor A. de Jesus Neto, Arash Dadkhah, Robson Ferreira, Alireza Zohourian, Rongxing Lu, and Ali A. Ghorbani (Canadian Institute for Cybersecurity, University of New Brunswick).
* **Publication**: *Sensors* (MDPI), 2023.
* **Citation**: Neto et al., *"CICIoT2023: A Real-Time Dataset and Benchmark for Large-Scale Attacks in IoT Environment"*, Sensors 2023, 23(12), 5941.
* **Physical Testbed Topology**:
  - **105 Devices in Total**:
    - **67 Physical IoT Devices** (Smart cameras, smart plugs, smart bulbs, audio hubs, microcontrollers from vendors including TP-Link, Tuya, Xiaomi, Sonoff, Philips Hue, etc.).
    - **30 Attacking Machines / Virtual Machines** running Kali Linux, executing automated attack suites (`hping3`, `scapy`, `LOIC`, `nmap`, `hydra`).
    - **8 Edge Routers / Gateways / Victim Servers**.
  - Capture duration spanned multiple days, resulting in **63 serialized CSV partitions** (`Merged01.csv` to `Merged63.csv`), totaling **8.66 GB** and **~46.6 million flow records**.

---

### 5.2 The 39 Numerical Features: Complete Physical & Mathematical Dictionary

Below is the definitive, audited dictionary of all 39 features present in our repository's CSV files (as defined in [`track1_dataset_evaluation/src/config.py`](file:///c:/Users/priya/Downloads/capstone/track1_dataset_evaluation/src/config.py)):

| Index | Feature Name | Protocol Layer | Unit / Range | Mathematical / Physical Meaning | Relevance to IoT Attack Detection |
| :---: | :--- | :---: | :---: | :--- | :--- |
| **1** | `Header_Length` | L3/L4 | Bytes ($0 - 60$) | Mean length of the protocol header. IPv4 headers are typically $20\text{B}$; TCP headers are $20 - 60\text{B}$. | Tunnels (`MIRAI-GREIP_FLOOD`) encapsulate extra headers; malformed fragmentation attacks distort header sizes. |
| **2** | `Protocol Type` | L3 | Integer | IANA protocol number from IPv4 header ($1=\text{ICMP}, 2=\text{IGMP}, 6=\text{TCP}, 17=\text{UDP}, 47=\text{GRE}$). | Distinguishes transport protocol families instantly (e.g. UDP floods vs TCP floods). |
| **3** | `Time_To_Live` | L3 | Hops ($0 - 255$) | Mean IPv4 TTL counter. Decremented by 1 at each router hop. | Exposes spoofed flood vectors where simulated packets carry artificial or static TTLs. |
| **4** | `Rate` | L3/L4 | Packets/sec | Rate of packet arrivals within the observation window. | Massive discriminator for volumetric floods ($>20,000\text{ pps}$) vs normal IoT sensing ($<50\text{ pps}$). |
| **5** | `fin_flag_number` | L4 | Ratio $[0.0, 1.0]$ | Proportion of packets with TCP `FIN` flag set. | Identifies `DDOS-RSTFINFLOOD` attacks designed to tear down connection states. |
| **6** | `syn_flag_number` | L4 | Ratio $[0.0, 1.0]$ | Proportion of packets with TCP `SYN` flag set. | Critical discriminator for `DOS-SYN_FLOOD` and `DDOS-SYN_FLOOD`. |
| **7** | `rst_flag_number` | L4 | Ratio $[0.0, 1.0]$ | Proportion of packets with TCP `RST` flag set. | Spikes during port scanning (`RECON-PORTSCAN`) when closed ports reject connection attempts. |
| **8** | `psh_flag_number` | L4 | Ratio $[0.0, 1.0]$ | Proportion of packets with TCP `PSH` flag set. | Identifies `DDOS-PSHACK_FLOOD` and interactive command injection sessions. |
| **9** | `ack_flag_number` | L4 | Ratio $[0.0, 1.0]$ | Proportion of packets with TCP `ACK` flag set. | Distinguishes connection establishment from ongoing data transfer and ACK flood attacks. |
| **10** | `ece_flag_number` | L4 | Ratio $[0.0, 1.0]$ | Proportion of packets with TCP `ECE` congestion flag set. | Abnormal flag combinations indicate custom packet-crafting tools (`scapy`). |
| **11** | `cwr_flag_number` | L4 | Ratio $[0.0, 1.0]$ | Proportion of packets with TCP `CWR` flag set. | Near-zero variance feature; useful for detecting malformed protocol headers. |
| **12** | `ack_count` | L4 | Count ($\ge 0$) | Cumulative count of ACK packets observed in the sliding window buffer. | Tracks absolute volume of acknowledged packets. |
| **13** | `syn_count` | L4 | Count ($\ge 0$) | Cumulative count of SYN packets observed in the window buffer. | Rapidly increments during SYN flooding campaigns. |
| **14** | `fin_count` | L4 | Count ($\ge 0$) | Cumulative count of FIN packets observed in the window buffer. | Tracks connection teardown volume. |
| **15** | `rst_count` | L4 | Count ($\ge 0$) | Cumulative count of RST packets observed in the window buffer. | Tracks connection reset frequency. |
| **16** | `HTTP` | L7 | Ratio $[0.0, 1.0]$ | Proportion of traffic matching HTTP port/signature. | Crucial for isolating web application attacks (`SQLINJECTION`, `XSS`, `COMMANDINJECTION`). |
| **17** | `HTTPS` | L7 | Ratio $[0.0, 1.0]$ | Proportion of traffic matching encrypted TLS/HTTPS (Port 443). | Legitimate IoT cloud communications often run over TLS. |
| **18** | `DNS` | L7 | Ratio $[0.0, 1.0]$ | Proportion of traffic matching DNS (Port 53). | Detects `DNS_SPOOFING` cache-poisoning attacks. |
| **19** | `Telnet` | L7 | Ratio $[0.0, 1.0]$ | Proportion of traffic matching unencrypted Telnet (Port 23). | Primary propagation vector for IoT botnets (`Mirai`). |
| **20** | `SMTP` | L7 | Ratio $[0.0, 1.0]$ | Proportion of traffic matching email SMTP (Port 25). | Rare in IoT; near-zero feature in normal IoT smart homes. |
| **21** | `SSH` | L7 | Ratio $[0.0, 1.0]$ | Proportion of traffic matching encrypted SSH (Port 22). | Target of `DICTIONARYBRUTEFORCE` credential attacks. |
| **22** | `IRC` | L7 | Ratio $[0.0, 1.0]$ | Proportion of traffic matching Internet Relay Chat (Port 6667). | Traditional botnet Command & Control (C2) protocol. |
| **23** | `TCP` | L4 | Ratio $[0.0, 1.0]$ | Proportion of flows utilizing Transmission Control Protocol. | Establishes Layer 4 transport baseline. |
| **24** | `UDP` | L4 | Ratio $[0.0, 1.0]$ | Proportion of flows utilizing User Datagram Protocol. | Spikes to $1.0$ in `DOS-UDP_FLOOD` and `MIRAI-UDPPLAIN`. |
| **25** | `DHCP` | L7/L2 | Ratio $[0.0, 1.0]$ | Proportion of traffic matching DHCP IP configuration. | Link-level configuration protocol. |
| **26** | `ARP` | L2 | Ratio $[0.0, 1.0]$ | Proportion of traffic matching Address Resolution Protocol. | Core indicator for `MITM-ARPSPOOFING`. |
| **27** | `ICMP` | L3 | Ratio $[0.0, 1.0]$ | Proportion of traffic matching ICMP control messages. | Discriminates `DDOS-ICMP_FLOOD` and `RECON-PINGSWEEP`. |
| **28** | `IGMP` | L3 | Ratio $[0.0, 1.0]$ | Proportion of traffic matching IP Multicast Group Management. | Multicast signaling for smart speakers and media devices. |
| **29** | `IPv` | L3 | Ratio $[0.0, 1.0]$ | Proportion of packets conforming to standard IPv4. | Sanity check for valid Layer 3 encapsulation. |
| **30** | `LLC` | L2 | Ratio $[0.0, 1.0]$ | Proportion of packets with Logical Link Control headers. | Data Link Layer control framing. |
| **31** | `Tot sum` | Statistical | Bytes | Total sum of all packet frame sizes within the observation window. | High in large data transfers; low in pure SYN/ACK control floods. |
| **32** | `Min` | Statistical | Bytes | Minimum packet size observed in the window buffer. | Many flood attacks use minimal fixed packets ($60\text{B}$). |
| **33** | `Max` | Statistical | Bytes | Maximum packet size observed in the window buffer. | Exposes jumbo frames or large buffer-overflow payloads. |
| **34** | `AVG` | Statistical | Bytes | Arithmetic mean of packet sizes in the window buffer. | Average flow byte footprint. |
| **35** | `Std` | Statistical | Bytes | Standard deviation of packet sizes ($\sigma$). | Volumetric floods have $\sigma \approx 0$ (fixed packet size); web traffic has high $\sigma$. |
| **36** | `Tot size` | Statistical | Bytes | Total byte size aggregated across all packet payloads. | Metric of total network bandwidth consumption. |
| **37** | `IAT` | Statistical | Seconds | Average Inter-Arrival Time between successive packets. | Drops to microseconds ($10^{-5}\text{ s}$) in floods; seconds in normal IoT. |
| **38** | `Number` | Statistical | Count | Total packet count in window (typically $100$ in count windows). | Normalization denominator for rate calculations. |
| **39** | `Variance` | Statistical | Bytes$^2$ | Statistical variance of packet lengths ($\sigma^2 = \text{Std}^2$). | Quadratic dispersion metric of flow packet sizes. |

---

### 5.3 The 34 Attack Labels & 8 Functional Attack Profiles

The 34 classes in CICIoT2023 map into **8 functional categories** as defined in [`track1_dataset_evaluation/src/config.py`](file:///c:/Users/priya/Downloads/capstone/track1_dataset_evaluation/src/config.py):

```
+-----------------------------------------------------------------------------------------------+
| CATEGORY          | SPECIFIC ATTACK LABELS INCLUDED                                            |
+-----------------------------------------------------------------------------------------------+
| 1. Benign (1)     | BENIGN (Normal sensor telemetry, smart speaker audio, configuration)      |
| 2. DDoS (12)      | DDOS-RSTFINFLOOD, DDOS-PSHACK_FLOOD, DDOS-SYN_FLOOD, DDOS-UDP_FLOOD,      |
|                   | DDOS-TCP_FLOOD, DDOS-ICMP_FLOOD, DDOS-SYNONYMOUSIP_FLOOD,                  |
|                   | DDOS-ACK_FRAGMENTATION, DDOS-UDP_FRAGMENTATION, DDOS-ICMP_FRAGMENTATION,   |
|                   | DDOS-SLOWLORIS, DDOS-HTTP_FLOOD                                           |
| 3. DoS (4)        | DOS-UDP_FLOOD, DOS-SYN_FLOOD, DOS-TCP_FLOOD, DOS-HTTP_FLOOD               |
| 4. Mirai (3)      | MIRAI-GREETH_FLOOD, MIRAI-GREIP_FLOOD, MIRAI-UDPPLAIN                     |
| 5. Recon (5)      | RECON-PINGSWEEP, RECON-OSSCAN, RECON-PORTSCAN, VULNERABILITYSCAN,          |
|                   | RECON-HOSTDISCOVERY                                                       |
| 6. Spoofing (2)   | DNS_SPOOFING, MITM-ARPSPOOFING                                            |
| 7. Web (6)        | BROWSERHIJACKING, BACKDOOR_MALWARE, XSS, UPLOADING_ATTACK,                 |
|                   | SQLINJECTION, COMMANDINJECTION                                            |
| 8. BruteForce (1) | DICTIONARYBRUTEFORCE                                                      |
+-----------------------------------------------------------------------------------------------+
```

---

### 5.4 The Mechanics of Class Imbalance in CICIoT2023

In the raw, natural traffic of CICIoT2023, the sample counts are staggeringly imbalanced:

```
[ Volumetric Floods (DDoS / DoS) ] --------> >90.0% of all flows (~42 million flows)
[ Normal Benign Traffic ] -----------------> ~5.0% of flows (~2.3 million flows)
[ Stealthy Web Exploits & Brute Force ] ---> <0.5% of flows (~50,000 flows total)
```

#### Why Does This Imbalance Occur Physically?
1. **The Nature of Flooding**: To deny service to a server, an attacker must blast tens of thousands of packets per second. In 10 minutes of a UDP flood, millions of packets are generated.
2. **The Nature of an Exploit**: To execute an SQL injection (`SQLINJECTION`), an attacker sends **one single malicious HTTP POST request** carrying the payload `' OR '1'='1`. In network telemetry, this single request generates only 1 or 2 flow records!
3. **The Literature Accuracy Paradox**:
   If an algorithm simply learns to predict `DDoS` every time, its accuracy is:
   $$\text{Accuracy} = \frac{42,000,000 \text{ true positives}}{46,600,000 \text{ total flows}} = 90.1\%!$$
   Even worse, if it detects benign traffic and floods, it achieves **$99.0\%$ overall accuracy**, while its detection rate on `SQLINJECTION`, `COMMANDINJECTION`, and `BACKDOOR_MALWARE` is **$0.0\%$!**
   *This is why Macro F1 and per-class recall are our primary research metrics, not accuracy!*

---

### 5.5 Scientific Boundaries: What CICIoT2023 Can and Cannot Establish

To defend our research with academic rigor, we must clearly acknowledge what this dataset enables and what its physical boundaries prevent:

* **What CICIoT2023 CAN Establish**:
  1. Statistical separability between volumetric flooding attacks and normal IoT communications.
  2. The efficacy of cost-sensitive machine learning algorithms under extreme class imbalance.
  3. Feature importance rankings (e.g., proving whether 20 features can match 39 features).
  4. Generalization across independent capture files (`cross_capture_test_clean.csv`).
* **What CICIoT2023 CANNOT Establish**:
  1. **Physical Edge Device Hardware Feasibility**: Running inference on a Windows 11 laptop CPU does not prove an algorithm can run inside an 8-bit microcontroller with 64 KB RAM.
  2. **Topological Source Separation**: Because IP addresses were stripped, a model *cannot* distinguish a distributed botnet attack from a single-source DoS attack if their packet rates and flags are identical (e.g., `DOS-SYN_FLOOD` vs. `DDOS-SYN_FLOOD`).

---

## 6. Mapping Concepts to Repository Implementation

| Concept | Where it Appears in Our Codebase | What it Does in the Code |
| :--- | :--- | :--- |
| **39 Feature Definitions** | [`track1_dataset_evaluation/src/config.py`](file:///c:/Users/priya/Downloads/capstone/track1_dataset_evaluation/src/config.py) | Defines `FEATURE_COLUMNS` list used by preprocessors. |
| **8-Class Mapping** | [`track1_dataset_evaluation/src/config.py`](file:///c:/Users/priya/Downloads/capstone/track1_dataset_evaluation/src/config.py) | Maps 34 raw strings into 8 functional categories (`LABEL_MAPPING_8CLASSES`). |
| **Streaming Chunk Loader** | [`track2_ids_engineering/src/data_loader.py`](file:///c:/Users/priya/Downloads/capstone/track2_ids_engineering/src/data_loader.py) | Uses `pd.read_csv(chunksize=100000)` to stream multi-gigabyte CSVs safely. |
| **Stratified Sampling** | [`track2_ids_engineering/src/create_sample.py`](file:///c:/Users/priya/Downloads/capstone/track2_ids_engineering/src/create_sample.py) | Capped floods at 20,000 flows while retaining 100% of rare attacks. |
| **Leak-Free Partitioning** | [`track1_dataset_evaluation/src/data_partitioning.py`](file:///c:/Users/priya/Downloads/capstone/track1_dataset_evaluation/src/data_partitioning.py) | Groups identical feature vectors via hashes into 70/15/15 partitions. |
| **Cross-Capture Holdout** | [`track1_dataset_evaluation/src/cross_capture.py`](file:///c:/Users/priya/Downloads/capstone/track1_dataset_evaluation/src/cross_capture.py) | Samples unvisited files `Merged26`–`Merged40` to test cross-capture generalization. |
| **Balanced Class Weights** | [`track1_dataset_evaluation/src/models.py`](file:///c:/Users/priya/Downloads/capstone/track1_dataset_evaluation/src/models.py) | `BalancedXGBClassifier` computes $w_c = \frac{N}{K \cdot N_c}$ during `.fit()`. |
| **Metric Profiling** | [`track1_dataset_evaluation/src/evaluate.py`](file:///c:/Users/priya/Downloads/capstone/track1_dataset_evaluation/src/evaluate.py) | Evaluates Macro F1, Recall, Precision, Latency ($\mu\text{s}$), and Throughput. |

---

## 7. Comprehensive Networking & IDS Glossary

- **5-Tuple**: The combination of `(Source IP, Destination IP, Source Port, Destination Port, Protocol)` uniquely identifying a network flow.
- **ARP Spoofing**: An attack where an adversary sends fake ARP messages onto a LAN to associate the attacker's MAC address with the IP address of a legitimate gateway.
- **Bidirectional Flow (Biflow)**: An analytical record combining both forward and reverse traffic between two sockets.
- **Inter-Arrival Time (IAT)**: The time duration elapsed between the arrival of two consecutive packets in a flow.
- **Macro F1-Score**: The unweighted arithmetic mean of F1-scores across all individual classes, giving equal importance to minority and majority classes:
  $$\text{Macro F1} = \frac{1}{K} \sum_{c=1}^K F1_c$$
- **NIDS (Network Intrusion Detection System)**: An appliance or software engine monitoring network traffic to identify unauthorized access or attacks.
- **TCB (Transmission Control Block)**: Internal operating system memory structure storing sequence numbers and socket states for an active TCP connection.
- **TCP Three-Way Handshake**: The 3-packet sequence (`SYN` $\to$ `SYN-ACK` $\to$ `ACK`) used to establish a reliable TCP session.
- **Time-to-Live (TTL)**: An IPv4 header field decremented by routers to prevent packets from endlessly circulating in network loops.
- **Weighted F1-Score**: An F1-score average where each class is weighted by its sample support ($N_c / N$), which can mask complete detection failure on minority classes.

---

## 8. Self-Defense Checklist for Oral Examination

When presenting or defending your project before faculty or reviewers, you should be able to answer each of the following questions from memory:

1. **Why do we use network flows instead of raw packet payloads in our ML pipeline?**
   - *Answer*: Payload inspection (DPI) is computationally prohibitive at gigabit line-rates, violates user privacy, and fails completely when traffic is encrypted with TLS/HTTPS. Flow statistics capture structural and behavioral anomalies without inspecting payload content.
2. **Why are the TCP flag values in our dataset decimals like `0.99` instead of binary integers `0` or `1`?**
   - *Answer*: Features are computed over an observation window (e.g., 100 packets). The flag features represent the proportion of packets in that window that carried that specific control flag. A value of `0.99` indicates that $99\%$ of packets in that window had that flag enabled.
3. **Why did earlier research report 99% accuracy while having poor minority attack detection?**
   - *Answer*: The "Accuracy Paradox" caused by extreme class imbalance. Over $90\%$ of raw traffic consists of volumetric flood attacks. A model that accurately classifies floods will score high overall accuracy even if it misses every stealthy web attack.
4. **Why was it necessary to remove exact duplicate feature vectors before splitting into train and test sets?**
   - *Answer*: Volumetric attack scripts generate thousands of identical packet templates, producing identical 39-feature vectors. If split randomly, the exact same vector appears in both train and test sets. The model memorizes these coordinates, creating artificial test accuracy that does not generalize to unseen traffic.
