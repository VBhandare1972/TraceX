import time
from feature_extractor import predict_url, get_model_and_scaler

def run_tests():
    print("================ TRACEX ML VERIFICATION ================")
    model, scaler, feature_names = get_model_and_scaler()
    print(f"[+] Loaded Model: {type(model).__name__}")
    print(f"[+] Scaler: {type(scaler).__name__}")
    print(f"[+] Feature Count: {len(feature_names)}")

    test_cases = [
        # Legitimate URLs
        ("https://www.google.com", "Safe"),
        ("https://en.wikipedia.org", "Safe"),
        ("https://github.com/torvalds/linux", "Safe"),
        ("https://www.amazon.com/dp/B08N5WRWNW", "Safe"),
        
        # Suspicious / Phishing URLs
        ("http://192.168.1.10/login/bank-of-america/verify.php", "Dangerous"),
        ("http://paypal-security-update.account-verification-login.com/webscr?cmd=_login-submit", "Dangerous"),
        ("http://secure-update-appleid.auth-recovery-9281.net/signin?id=98723", "Dangerous")
    ]

    all_passed = True
    latencies = []

    print("\n--- Running Test Cases ---")
    for url, expected_class in test_cases:
        t0 = time.perf_counter()
        result = predict_url(url)
        t1 = time.perf_counter()
        latency_ms = (t1 - t0) * 1000
        latencies.append(latency_ms)

        threat = result["threat_class"]
        risk = result["risk_score"]
        prob = result["phishing_probability"]
        status = "PASS" if threat == expected_class else "CHECK"
        if status != "PASS":
            all_passed = False

        print(f"[{status}] {url}")
        print(f"       Class: {threat} (Expected: {expected_class}) | Risk Score: {risk}/100 | Time: {latency_ms:.2f}ms")
        print(f"       Reasons: {result['risk_reasons']}")

    avg_latency = sum(latencies) / len(latencies)
    print(f"\nAverage Inference Latency: {avg_latency:.2f} ms")
    if avg_latency < 50.0:
        print("[+] Latency is well within real-time browser extension requirements (< 200 ms)!")
    print("========================================================\n")

if __name__ == "__main__":
    run_tests()
