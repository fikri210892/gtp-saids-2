#!/usr/bin/env python3
"""
step1_extract_features.py
Extract features dari PCAP files untuk CNN training
"""

from scapy.all import *
from scapy.contrib.gtp import *
import pandas as pd
import numpy as np
import sys

class FeatureExtractor:
    def __init__(self):
        self.features_list = []
        
    def calculate_entropy(self, data):
        """Calculate Shannon entropy"""
        if not data or len(data) == 0:
            return 0
        
        byte_counts = np.bincount(np.frombuffer(data, dtype=np.uint8), minlength=256)
        probabilities = byte_counts[byte_counts > 0] / len(data)
        entropy = -np.sum(probabilities * np.log2(probabilities))
        return entropy
    
    def extract_packet_features(self, pkt):
        """Extract 20 features dari packet"""
        
        features = {}
        
        # Feature 1-7: Basic packet info
        features['pkt_size'] = len(pkt)
        features['ip_len'] = pkt[IP].len if pkt.haslayer(IP) else 0
        features['ttl'] = pkt[IP].ttl if pkt.haslayer(IP) else 0
        features['ip_proto'] = pkt[IP].proto if pkt.haslayer(IP) else 0
        features['ip_flags'] = int(pkt[IP].flags) if pkt.haslayer(IP) else 0
        features['frag_offset'] = pkt[IP].frag if pkt.haslayer(IP) else 0
        features['is_fragment'] = 1 if pkt.haslayer(IP) and pkt[IP].frag > 0 else 0
        
        # Feature 8-10: UDP info
        features['src_port'] = pkt[UDP].sport if pkt.haslayer(UDP) else 0
        features['dst_port'] = pkt[UDP].dport if pkt.haslayer(UDP) else 0
        features['udp_len'] = pkt[UDP].len if pkt.haslayer(UDP) else 0
        
        # Feature 11-13: GTP info
        if pkt.haslayer(GTPHeader):
            features['gtp_teid'] = pkt[GTPHeader].teid
            features['gtp_type'] = pkt[GTPHeader].gtp_type
            features['gtp_seq'] = pkt[GTPHeader].seq if hasattr(pkt[GTPHeader], 'seq') else 0
        else:
            features['gtp_teid'] = 0
            features['gtp_type'] = 0
            features['gtp_seq'] = 0
        
        # Feature 14-16: Payload analysis
        if pkt.haslayer(Raw):
            payload = bytes(pkt[Raw])
            features['payload_len'] = len(payload)
            features['payload_entropy'] = self.calculate_entropy(payload)
            features['null_bytes'] = payload.count(b'\x00')
        else:
            features['payload_len'] = 0
            features['payload_entropy'] = 0
            features['null_bytes'] = 0
        
        # Feature 17-20: Statistical features
        features['has_gtp'] = 1 if pkt.haslayer(GTPHeader) else 0
        features['pkt_rate'] = 0  # Placeholder (akan dihitung dari timestamp)
        features['byte_rate'] = 0  # Placeholder
        features['inter_arrival'] = 0  # Placeholder
        
        return features
    
    def extract_from_pcap(self, pcap_file, label):
        """Extract features dari PCAP file"""
        
        print(f"[*] Processing {pcap_file} (label={label})...")
        
        try:
            packets = rdpcap(pcap_file)
        except Exception as e:
            print(f"[!] Error reading {pcap_file}: {e}")
            return
        
        prev_time = None
        
        for pkt in packets:
            # Only process GTP packets (UDP port 2152)
            if pkt.haslayer(UDP) and (pkt[UDP].dport == 2152 or pkt[UDP].sport == 2152):
                features = self.extract_packet_features(pkt)
                
                # Calculate inter-arrival time
                if prev_time is not None:
                    features['inter_arrival'] = float(pkt.time) - prev_time
                prev_time = float(pkt.time)
                
                # Add label
                features['label'] = label
                
                self.features_list.append(features)
        
        print(f"[+] Extracted {len([f for f in self.features_list if f['label']==label])} packets")
    
    def save_to_csv(self, output_file):
        """Save features to CSV"""
        
        if len(self.features_list) == 0:
            print("[!] No features extracted!")
            return
        
        df = pd.DataFrame(self.features_list)
        df.to_csv(output_file, index=False)
        
        print(f"\n{'='*60}")
        print(f"[✓] Features saved to: {output_file}")
        print(f"{'='*60}")
        print(f"Total samples:    {len(df)}")
        print(f"Normal (label=0): {len(df[df['label']==0])}")
        print(f"Attack (label=1): {len(df[df['label']==1])}")
        print(f"Features:         {len(df.columns)-1}")  # -1 for label column
        print(f"{'='*60}")

# Main
if __name__ == "__main__":
    print("="*60)
    print("  GTP Feature Extraction for CNN Training")
    print("="*60)
    
    extractor = FeatureExtractor()
    
    # Extract normal traffic (label=0)
    extractor.extract_from_pcap('normal_traffic.pcap', label=0)
    
    # Extract attack traffic (label=1)
    extractor.extract_from_pcap('attack_malformed.pcap', label=1)
    extractor.extract_from_pcap('attack_flood.pcap', label=1)
    extractor.extract_from_pcap('attack_invalid_teid.pcap', label=1)
    extractor.extract_from_pcap('attack_spoofing.pcap', label=1)
    extractor.extract_from_pcap('attack_fragmented.pcap', label=1)
    
    # Save to CSV
    extractor.save_to_csv('training_dataset.csv')
    
    print("\n[✓] Feature extraction completed!")
    print("[*] Next step: python3 step2_train_cnn.py")
