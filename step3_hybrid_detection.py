#!/usr/bin/env python3
"""
step3_hybrid_detection.py
Hybrid IDS: Suricata (Signature) + CNN (Anomaly)
"""

import subprocess
import json
import pandas as pd
import numpy as np
import pickle
import sys
import os
from scapy.all import *
from scapy.contrib.gtp import *
from tensorflow import keras
from collections import defaultdict

class HybridIDS:
    def __init__(self, pcap_file):
        self.pcap_file = pcap_file
        
        # Check if file exists
        if not os.path.exists(pcap_file):
            print(f"[!] Error: {pcap_file} not found!")
            sys.exit(1)
        
        # Load CNN model
        print("[*] Loading CNN model...")
        self.model = keras.models.load_model('gtp_cnn_model.h5')
        
        with open('scaler.pkl', 'rb') as f:
            self.scaler = pickle.load(f)
        
        print("[+] CNN model loaded")
        
        # Results storage
        self.suricata_alerts = []
        self.cnn_predictions = []
        self.hybrid_results = []
        
    def calculate_entropy(self, data):
        """Calculate Shannon entropy"""
        if not data or len(data) == 0:
            return 0
        
        byte_counts = np.bincount(np.frombuffer(data, dtype=np.uint8), minlength=256)
        probabilities = byte_counts[byte_counts > 0] / len(data)
        entropy = -np.sum(probabilities * np.log2(probabilities))
        return entropy
    
    def extract_packet_features(self, pkt):
        """Extract features from packet"""
        
        features = []
        
        # Feature 1-7: Basic packet info
        features.append(len(pkt))
        features.append(pkt[IP].len if pkt.haslayer(IP) else 0)
        features.append(pkt[IP].ttl if pkt.haslayer(IP) else 0)
        features.append(pkt[IP].proto if pkt.haslayer(IP) else 0)
        features.append(int(pkt[IP].flags) if pkt.haslayer(IP) else 0)
        features.append(pkt[IP].frag if pkt.haslayer(IP) else 0)
        features.append(1 if pkt.haslayer(IP) and pkt[IP].frag > 0 else 0)
        
        # Feature 8-10: UDP info
        features.append(pkt[UDP].sport if pkt.haslayer(UDP) else 0)
        features.append(pkt[UDP].dport if pkt.haslayer(UDP) else 0)
        features.append(pkt[UDP].len if pkt.haslayer(UDP) else 0)
        
        # Feature 11-13: GTP info
        if pkt.haslayer(GTPHeader):
            features.append(pkt[GTPHeader].teid)
            features.append(pkt[GTPHeader].gtp_type)
            features.append(pkt[GTPHeader].seq if hasattr(pkt[GTPHeader], 'seq') else 0)
        else:
            features.extend([0, 0, 0])
        
        # Feature 14-16: Payload analysis
        if pkt.haslayer(Raw):
            payload = bytes(pkt[Raw])
            features.append(len(payload))
            features.append(self.calculate_entropy(payload))
            features.append(payload.count(b'\x00'))
        else:
            features.extend([0, 0, 0])
        
        # Feature 17-20: Statistical features
        features.append(1 if pkt.haslayer(GTPHeader) else 0)
        features.extend([0, 0, 0])  # Placeholders
        
        return np.array(features)
    
    def run_suricata(self):
        """Run Suricata on PCAP"""
        print(f"\n{'='*60}")
        print("ENGINE A: SURICATA (Signature-based Detection)")
        print("="*60)
        print(f"[*] Running Suricata on {self.pcap_file}...")
        
        output_dir = "suricata_output"
        os.makedirs(output_dir, exist_ok=True)
        
        result = subprocess.run([
            'suricata',
            '-r', self.pcap_file,
            '-l', output_dir,
            '-c', '/etc/suricata/suricata.yaml'
        ], capture_output=True, text=True)
        
        # Parse eve.json
        eve_file = f"{output_dir}/eve.json"
        
        if not os.path.exists(eve_file):
            print("[!] Warning: eve.json not found")
            return
        
        print(f"[*] Parsing alerts from {eve_file}...")
        
        with open(eve_file, 'r') as f:
            for line in f:
                try:
                    event = json.loads(line)
                    if event.get('event_type') == 'alert':
                        self.suricata_alerts.append({
                            'timestamp': event['timestamp'],
                            'src_ip': event['src_ip'],
                            'dest_ip': event['dest_ip'],
                            'src_port': event.get('src_port', 0),
                            'dest_port': event.get('dest_port', 0),
                            'signature': event['alert']['signature'],
                            'sid': event['alert']['signature_id'],
                            'severity': event['alert']['severity']
                        })
                except:
                    continue
        
        print(f"[+] Suricata alerts: {len(self.suricata_alerts)}")
        
        if len(self.suricata_alerts) > 0:
            print("\nTop 5 signatures detected:")
            sigs = {}
            for alert in self.suricata_alerts:
                sig = alert['signature']
                sigs[sig] = sigs.get(sig, 0) + 1
            
            for sig, count in sorted(sigs.items(), key=lambda x: x[1], reverse=True)[:5]:
                print(f"  - {sig}: {count} alerts")
    
    def run_cnn(self):
        """Run CNN prediction on PCAP"""
        print(f"\n{'='*60}")
        print("ENGINE B: CNN (Anomaly-based Detection)")
        print("="*60)
        print(f"[*] Running CNN detection on {self.pcap_file}...")
        
        packets = rdpcap(self.pcap_file)
        print(f"[*] Total packets: {len(packets)}")
        
        gtp_count = 0
        
        for idx, pkt in enumerate(packets):
            if pkt.haslayer(UDP) and (pkt[UDP].dport == 2152 or pkt[UDP].sport == 2152):
                gtp_count += 1
                
                # Extract features
                features = self.extract_packet_features(pkt)
                
                # Clean features
                features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
                
                # Normalize
                features_scaled = self.scaler.transform([features])
                features_cnn = features_scaled.reshape(1, -1, 1)
                
                # Predict
                score = self.model.predict(features_cnn, verbose=0)[0][0]
                
                if score > 0.5:  # Threshold
                    self.cnn_predictions.append({
                        'packet_id': idx,
                        'timestamp': float(pkt.time),
                        'src_ip': pkt[IP].src if pkt.haslayer(IP) else 'N/A',
                        'dest_ip': pkt[IP].dst if pkt.haslayer(IP) else 'N/A',
                        'src_port': pkt[UDP].sport if pkt.haslayer(UDP) else 0,
                        'dest_port': pkt[UDP].dport if pkt.haslayer(UDP) else 0,
                        'score': float(score),
                        'prediction': 'ATTACK' if score > 0.7 else 'SUSPICIOUS'
                    })
        
        print(f"[+] GTP packets analyzed: {gtp_count}")
        print(f"[+] CNN predictions (score>0.5): {len(self.cnn_predictions)}")
        print(f"    - High confidence (>0.7): {len([p for p in self.cnn_predictions if p['score']>0.7])}")
        print(f"    - Medium confidence (0.5-0.7): {len([p for p in self.cnn_predictions if 0.5<p['score']<=0.7])}")
        
        if len(self.cnn_predictions) > 0:
            avg_score = np.mean([p['score'] for p in self.cnn_predictions])
            max_score = np.max([p['score'] for p in self.cnn_predictions])
            print(f"\nCNN Statistics:")
            print(f"  Average score: {avg_score:.3f}")
            print(f"  Max score: {max_score:.3f}")
    
    def correlate_results(self):
        """Correlate Suricata + CNN results"""
        print(f"\n{'='*60}")
        print("CORRELATION ENGINE")
        print("="*60)
        print("[*] Correlating results from both engines...")
        
        # Group by flow (src_ip:dest_ip)
        flows = defaultdict(lambda: {'suricata': [], 'cnn': []})
        
        for alert in self.suricata_alerts:
            key = f"{alert['src_ip']}:{alert['dest_ip']}"
            flows[key]['suricata'].append(alert)
        
        for pred in self.cnn_predictions:
            key = f"{pred['src_ip']}:{pred['dest_ip']}"
            flows[key]['cnn'].append(pred)
        
        # Correlation logic
        for flow_key, detections in flows.items():
            suricata_count = len(detections['suricata'])
            cnn_count = len(detections['cnn'])
            cnn_max_score = max([p['score'] for p in detections['cnn']]) if cnn_count > 0 else 0
            
            # Decision logic
            if suricata_count > 0 and cnn_count > 0:
                # Both detected - CRITICAL
                result = {
                    'flow': flow_key,
                    'severity': 'CRITICAL',
                    'confidence': 'VERY_HIGH',
                    'detection_source': 'BOTH',
                    'suricata_alerts': suricata_count,
                    'cnn_detections': cnn_count,
                    'cnn_max_score': cnn_max_score,
                    'action': 'BLOCK',
                    'description': 'Confirmed attack by both engines'
                }
            elif suricata_count > 0:
                # Only Suricata - HIGH
                result = {
                    'flow': flow_key,
                    'severity': 'HIGH',
                    'confidence': 'HIGH',
                    'detection_source': 'SURICATA',
                    'suricata_alerts': suricata_count,
                    'cnn_detections': 0,
                    'cnn_max_score': 0,
                    'action': 'ALERT',
                    'description': 'Known attack pattern detected'
                }
            elif cnn_max_score >= 0.9:
                # CNN high confidence - HIGH
                result = {
                    'flow': flow_key,
                    'severity': 'HIGH',
                    'confidence': 'MEDIUM',
                    'detection_source': 'CNN',
                    'suricata_alerts': 0,
                    'cnn_detections': cnn_count,
                    'cnn_max_score': cnn_max_score,
                    'action': 'ALERT',
                    'description': f'Unknown anomaly detected (score: {cnn_max_score:.3f})'
                }
            elif cnn_max_score >= 0.7:
                # CNN medium confidence - MEDIUM
                result = {
                    'flow': flow_key,
                    'severity': 'MEDIUM',
                    'confidence': 'LOW',
                    'detection_source': 'CNN',
                    'suricata_alerts': 0,
                    'cnn_detections': cnn_count,
                    'cnn_max_score': cnn_max_score,
                    'action': 'LOG',
                    'description': f'Potential anomaly (score: {cnn_max_score:.3f})'
                }
            else:
                continue
            
            self.hybrid_results.append(result)
        
        print(f"[+] Correlated flows: {len(self.hybrid_results)}")
    
    def generate_report(self):
        """Generate detection report"""
        print(f"\n{'='*60}")
        print("HYBRID IDS DETECTION REPORT")
        print("="*60)
        print(f"PCAP File: {self.pcap_file}\n")
        
        print("Engine A (Suricata - Signature-based):")
        print(f"  Total Alerts: {len(self.suricata_alerts)}")
        
        print("\nEngine B (CNN - Anomaly-based):")
        print(f"  Total Predictions: {len(self.cnn_predictions)}")
        print(f"  High Confidence (>0.7): {len([p for p in self.cnn_predictions if p['score']>0.7])}")
        print(f"  Medium Confidence (0.5-0.7): {len([p for p in self.cnn_predictions if 0.5<p['score']<=0.7])}")
        
        print("\nHybrid Results:")
        print(f"  Total Detected Flows: {len(self.hybrid_results)}")
        
        # Count by severity
        critical = len([r for r in self.hybrid_results if r['severity']=='CRITICAL'])
        high = len([r for r in self.hybrid_results if r['severity']=='HIGH'])
        medium = len([r for r in self.hybrid_results if r['severity']=='MEDIUM'])
        
        print(f"  CRITICAL: {critical}")
        print(f"  HIGH:     {high}")
        print(f"  MEDIUM:   {medium}")
        
        # Count by source
        both = len([r for r in self.hybrid_results if r['detection_source']=='BOTH'])
        suricata_only = len([r for r in self.hybrid_results if r['detection_source']=='SURICATA'])
        cnn_only = len([r for r in self.hybrid_results if r['detection_source']=='CNN'])
        
        print(f"\nDetection Source:")
        print(f"  Both Engines: {both}")
        print(f"  Suricata Only: {suricata_only}")
        print(f"  CNN Only: {cnn_only}")
        
        print(f"\n{'-'*60}")
        print("DETECTED FLOWS:")
        print("-"*60)
        
        for result in sorted(self.hybrid_results, key=lambda x: {'CRITICAL':0, 'HIGH':1, 'MEDIUM':2}[x['severity']]):
            print(f"\n[{result['severity']}] {result['flow']}")
            print(f"  Source: {result['detection_source']}")
            print(f"  Confidence: {result['confidence']}")
            print(f"  Suricata Alerts: {result['suricata_alerts']}")
            print(f"  CNN Detections: {result['cnn_detections']}")
            if result['cnn_max_score'] > 0:
                print(f"  CNN Max Score: {result['cnn_max_score']:.3f}")
            print(f"  Action: {result['action']}")
            print(f"  Description: {result['description']}")
        
        print(f"\n{'='*60}")
        
        # Save to files
        if len(self.suricata_alerts) > 0:
            df_suricata = pd.DataFrame(self.suricata_alerts)
            df_suricata.to_csv('suricata_alerts.csv', index=False)
            print("[+] Saved: suricata_alerts.csv")
        
        if len(self.cnn_predictions) > 0:
            df_cnn = pd.DataFrame(self.cnn_predictions)
            df_cnn.to_csv('cnn_predictions.csv', index=False)
            print("[+] Saved: cnn_predictions.csv")
        
        if len(self.hybrid_results) > 0:
            df_hybrid = pd.DataFrame(self.hybrid_results)
            df_hybrid.to_csv('hybrid_results.csv', index=False)
            print("[+] Saved: hybrid_results.csv")
        
        print("="*60)

# Main
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 step3_hybrid_detection.py <pcap_file>")
        print("\nExample:")
        print("  python3 step3_hybrid_detection.py gtp_attack.pcap")
        sys.exit(1)
    
    pcap_file = sys.argv[1]
    
    ids = HybridIDS(pcap_file)
    ids.run_suricata()
    ids.run_cnn()
    ids.correlate_results()
    ids.generate_report()
