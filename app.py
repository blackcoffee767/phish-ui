import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import shap
from scipy.stats import kendalltau
import warnings
from urllib.parse import urlparse, urlunparse
import tldextract
import re
import random
import difflib

warnings.filterwarnings('ignore')
np.random.seed(42)
random.seed(42)

# Page config
st.set_page_config(page_title="Phishing Evasion Explorer", layout="wide", page_icon="🔍")

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        padding: 1rem;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 2rem;
    }
    .section-header {
        font-size: 1.5rem;
        font-weight: bold;
        color: #2c3e50;
        padding: 0.5rem;
        border-left: 4px solid #667eea;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
        background-color: #f8f9fa;
    }
    .success-box {
        background-color: #d4edda;
        border: 2px solid #28a745;
        border-radius: 8px;
        padding: 1rem;
        margin: 0;
        height: 100%;
    }
    .fail-box {
        background-color: #f8d7da;
        border: 2px solid #dc3545;
        border-radius: 8px;
        padding: 1rem;
        margin: 0;
        height: 100%;
    }
    .prediction-box {
        border-radius: 10px;
        padding: 1.5rem;
        margin: 0;
        border: 3px solid;
        height: 100%;
    }
    .high-risk {
        background-color: #fff3cd;
        border-color: #ff9800;
    }
    .low-risk {
        background-color: #d4edda;
        border-color: #28a745;
    }
    /* Diff Styles */
    .diff-text {
        font-family: 'Courier New', monospace;
        font-size: 1.1rem;
        padding: 10px;
        background-color: #f1f1f1;
        border-radius: 5px;
        word-break: break-all;
        min-height: 60px;
    }
    .diff-removed {
        background-color: #ffcccc;
        color: #b30000;
        text-decoration: line-through;
        padding: 0 4px;
        border-radius: 3px;
    }
    .diff-added {
        background-color: #ccffcc;
        color: #006600;
        font-weight: bold;
        padding: 0 4px;
        border-radius: 3px;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# FEATURE EXTRACTION
# ============================================================================
def extract_features_final(url_string):
    """Extract phishing detection features from a URL."""
    try:
        total_len_raw = len(url_string)
        if total_len_raw > 0:
            prob = [url_string.count(c) / total_len_raw for c in set(url_string)]
            entropy_val = -sum(p * np.log2(p) for p in prob if p > 0)
        else:
            entropy_val = 0.0
        
        url_lower = url_string.lower().strip()
        parsed = urlparse(url_lower)
        extracted = tldextract.extract(url_lower)
        
        if extracted.registered_domain:
            reg_domain = extracted.registered_domain
        elif extracted.domain and extracted.suffix:
            reg_domain = extracted.domain + '.' + extracted.suffix
        else:
            reg_domain = extracted.domain + extracted.suffix
        
        features = {}
        features['url_len'] = len(url_lower)
        features['dom_len'] = len(reg_domain)
        features['tld_len'] = len(extracted.suffix)
        features['path_len'] = len(parsed.path)
        features['query_len'] = len(parsed.query)
        features['letter_cnt'] = sum(c.isalpha() for c in url_lower)
        features['digit_cnt'] = sum(c.isdigit() for c in url_lower)
        features['special_cnt'] = sum(not c.isalnum() for c in url_lower)
        features['subdom_cnt'] = extracted.subdomain.count('.') + 1 if extracted.subdomain else 0
        features['eq_cnt'] = url_lower.count('=')
        features['qm_cnt'] = url_lower.count('?')
        features['amp_cnt'] = url_lower.count('&')
        features['dot_cnt'] = url_lower.count('.')
        features['dash_cnt'] = url_lower.count('-')
        features['under_cnt'] = url_lower.count('_')
        features['slash_cnt'] = url_lower.count('/')
        
        domain_part = parsed.netloc.split(':')[0]
        features['is_ip'] = 1 if re.match(r'^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$', domain_part) else 0
        features['is_https'] = 1 if parsed.scheme == 'https' else 0
        
        total_len = max(1, len(url_lower))
        features['letter_ratio'] = features['letter_cnt'] / total_len
        features['digit_ratio'] = features['digit_cnt'] / total_len
        features['spec_ratio'] = features['special_cnt'] / total_len
        features['entropy'] = entropy_val
        
        return features
    except Exception as e:
        return None

# ============================================================================
# MUTATION FUNCTIONS
# ============================================================================
def mutate_tld_typosquatting(url_string):
    extracted = tldextract.extract(url_string)
    current_tld = extracted.suffix
    short_to_long = {'com': ['xyz', 'net', 'org'], 'co': ['com', 'io'], 'io': ['com', 'co']}
    long_to_short = {'xyz': ['co', 'io'], 'net': ['co', 'io'], 'org': ['co', 'io']}
    
    new_tld = None
    if current_tld in short_to_long: new_tld = random.choice(short_to_long[current_tld])
    elif current_tld in long_to_short: new_tld = random.choice(long_to_short[current_tld])
    else: new_tld = 'co' if len(current_tld) > 2 else 'com'
    
    if new_tld:
        parsed = urlparse(url_string)
        old_netloc = parsed.netloc
        new_netloc = old_netloc.rsplit('.', 1)[0] + '.' + new_tld if '.' in old_netloc else old_netloc
        new_parsed = parsed._replace(netloc=new_netloc)
        return urlunparse(new_parsed)
    return url_string

def mutate_char_omission(url_string):
    parsed = urlparse(url_string)
    extracted = tldextract.extract(url_string)
    domain = extracted.domain
    if len(domain) > 3:
        pos = random.randint(1, len(domain) - 2)
        new_domain = domain[:pos] + domain[pos+1:]
        subdomain_part = extracted.subdomain + '.' if extracted.subdomain else ''
        new_netloc = subdomain_part + new_domain + '.' + extracted.suffix
        new_parsed = parsed._replace(netloc=new_netloc)
        return urlunparse(new_parsed)
    return url_string

def mutate_combosquatting(url_string):
    parsed = urlparse(url_string)
    extracted = tldextract.extract(url_string)
    prefixes = ['secure-', 'verify-', 'login-', 'account-', 'update-']
    suffixes = ['-secure', '-verify', '-login', '-account', '-update']
    domain = extracted.domain
    if random.random() > 0.5: new_domain = random.choice(prefixes) + domain
    else: new_domain = domain + random.choice(suffixes)
    subdomain_part = extracted.subdomain + '.' if extracted.subdomain else ''
    new_netloc = subdomain_part + new_domain + '.' + extracted.suffix
    new_parsed = parsed._replace(netloc=new_netloc)
    return urlunparse(new_parsed)

def mutate_visual_spoofing(url_string):
    parsed = urlparse(url_string)
    extracted = tldextract.extract(url_string)
    domain = extracted.domain
    substitutions = {'w': 'vv', 'm': random.choice(['rn', 'nn']), 'd': 'cl', 'h': 'ln'}
    chars = list(domain)
    mutated = False
    for i, char in enumerate(chars):
        if char in substitutions and not mutated:
            chars[i] = substitutions[char]
            mutated = True
            break
    if mutated:
        new_domain = ''.join(chars)
        subdomain_part = extracted.subdomain + '.' if extracted.subdomain else ''
        new_netloc = subdomain_part + new_domain + '.' + extracted.suffix
        new_parsed = parsed._replace(netloc=new_netloc)
        return urlunparse(new_parsed)
    return url_string

def mutate_subdomain_concat(url_string):
    extracted = tldextract.extract(url_string)
    if extracted.subdomain and '.' in extracted.subdomain:
        parts = extracted.subdomain.rsplit('.', 1)
        new_subdomain = ''.join(parts)
        parsed = urlparse(url_string)
        new_netloc = new_subdomain + '.' + extracted.domain + '.' + extracted.suffix
        new_parsed = parsed._replace(netloc=new_netloc)
        return urlunparse(new_parsed)
    elif extracted.subdomain:
        parsed = urlparse(url_string)
        new_netloc = extracted.subdomain + extracted.domain + '.' + extracted.suffix
        new_parsed = parsed._replace(netloc=new_netloc)
        return urlunparse(new_parsed)
    return url_string

MUTATIONS = {
    'TLD Typosquatting (.com → .co/.io)': mutate_tld_typosquatting,
    'Char Omission (google → gogle)': mutate_char_omission,
    'Combosquatting (add secure-/login- prefix)': mutate_combosquatting,
    'Visual Spoofing (w→vv, m→rn, d→cl)': mutate_visual_spoofing,
    'Subdomain Concat (sub.domain → subdomain)': mutate_subdomain_concat
}

# ============================================================================
# HELPERS
# ============================================================================
def get_robust_shap(explainer, x):
    try:
        vals = explainer.shap_values(x.reshape(1, -1))
        if isinstance(vals, list): return vals[1][0]
        if vals.ndim == 3: return vals[0, :, 1]
        return vals[0]
    except:
        return np.zeros(x.shape[0])

def get_stability_metrics(vals1, vals2):
    try:
        idx1 = np.argsort(np.abs(vals1))[-10:]
        idx2 = np.argsort(np.abs(vals2))[-10:]
        jaccard = len(set(idx1) & set(idx2)) / 10.0
        tau, _ = kendalltau(np.argsort(np.abs(vals1)), np.argsort(np.abs(vals2)))
        tau_normalized = (tau + 1) / 2 
        return jaccard, tau_normalized if not np.isnan(tau_normalized) else 0.5
    except:
        return 0.0, 0.5

def generate_diff_html(s1, s2):
    matcher = difflib.SequenceMatcher(None, s1, s2)
    before_html = ""
    after_html = ""
    for opcode, i1, i2, j1, j2 in matcher.get_opcodes():
        if opcode == 'equal':
            before_html += s1[i1:i2]
            after_html += s2[j1:j2]
        elif opcode == 'delete':
            before_html += f'<span class="diff-removed">{s1[i1:i2]}</span>'
        elif opcode == 'insert':
            after_html += f'<span class="diff-added">{s2[j1:j2]}</span>'
        elif opcode == 'replace':
            before_html += f'<span class="diff-removed">{s1[i1:i2]}</span>'
            after_html += f'<span class="diff-added">{s2[j1:j2]}</span>'
    return before_html, after_html

# ============================================================================
# MAIN
# ============================================================================
@st.cache_resource
def load_model_and_data():
    df = pd.read_csv('URL-Phish.csv')
    feature_cols = [c for c in df.columns if c not in ['url', 'dom', 'tld', 'label']]
    X_train, X_test, y_train, y_test = train_test_split(
        df[feature_cols].values, df['label'].values,
        test_size=0.2, stratify=df['label'], random_state=42
    )
    rf = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=42)
    rf.fit(X_train, y_train)
    explainer = shap.TreeExplainer(rf)
    return rf, explainer, feature_cols

@st.cache_data
def load_evaded_urls():
    return pd.read_csv('urls_with_evasion.csv')

def main():
    st.markdown('<div class="main-header">🔍 Phishing Evasion & XAI Stability Explorer</div>', unsafe_allow_html=True)
    
    with st.sidebar:
        st.markdown("### 📚 Research Context")
        st.info("This tool uses **Explanation Stability** to predict if an evasion attack will succeed.")
    
    with st.spinner("Loading AI Brain..."):
        rf, explainer, feature_cols = load_model_and_data()
        df_urls = load_evaded_urls()
    
    # 1. Select URL
    st.markdown('<div class="section-header">Step 1: Select a Phishing URL</div>', unsafe_allow_html=True)
    url_options = [f"{row['url'][:80]}..." for _, row in df_urls.iterrows()]
    selected_display = st.selectbox("Choose a URL:", url_options)
    selected_idx = url_options.index(selected_display)
    selected_url = df_urls.iloc[selected_idx]['url']
    
    # Pre-calculate original prediction (variables needed for later)
    f_orig = extract_features_final(selected_url)
    x_orig = np.array([f_orig[f] for f in feature_cols])
    orig_pred = rf.predict([x_orig])[0]

    # 2. Select Mutation
    st.markdown('<div class="section-header">Step 2: Select Mutation</div>', unsafe_allow_html=True)
    selected_mutation = st.radio("Technique:", list(MUTATIONS.keys()))
    
    if st.button("🚀 Apply Mutation & Predict", type="primary"):
        with st.spinner("Analyzing Stability..."):
            mutation_func = MUTATIONS[selected_mutation]
            mutated_url = mutation_func(selected_url)
            
            if mutated_url == selected_url:
                st.error("No change applied. Try a different URL/Mutation combo.")
                return
                
            f_mut = extract_features_final(mutated_url)
            x_mut = np.array([f_mut[f] for f in feature_cols])
            
            shap_orig = get_robust_shap(explainer, x_orig)
            shap_mut = get_robust_shap(explainer, x_mut)
            jac, ken = get_stability_metrics(shap_orig, shap_mut)
            stability = (jac + ken) / 2
            
            # --- VIEW 1: VISUAL DIFF ---
            st.markdown("### 🧬 Visual Changes")
            before_html, after_html = generate_diff_html(selected_url, mutated_url)
            c1, c2 = st.columns(2)
            c1.markdown("**Original (Removed items in Red):**")
            c1.markdown(f'<div class="diff-text">{before_html}</div>', unsafe_allow_html=True)
            c2.markdown("**Mutated (Added items in Green):**")
            c2.markdown(f'<div class="diff-text">{after_html}</div>', unsafe_allow_html=True)
            
            # --- PREPARE DATA ---
            EVASION_THRESHOLD = 0.6400
            predicted_evasion = (stability < EVASION_THRESHOLD)
            confidence = abs(stability - EVASION_THRESHOLD) / EVASION_THRESHOLD * 100
            mut_pred = rf.predict([x_mut])[0]
            mut_proba = rf.predict_proba([x_mut])[0]
            mut_conf = mut_proba[1] if mut_pred == 1 else mut_proba[0]
            evaded = (orig_pred == 1 and mut_pred == 0)
            prediction_correct = (predicted_evasion == evaded)

            # --- VIEW 2: SPLIT SCREEN (PREDICTION | TRUTH) ---
            st.markdown("---")
            pred_col, truth_col = st.columns(2)

            with pred_col:
                st.markdown('<div class="section-header">🔮 Evasion Prediction</div>', unsafe_allow_html=True)
                if predicted_evasion:
                    st.markdown(f"""
                    <div class="prediction-box high-risk">
                        <h3 style="margin: 0; color: #ff9800;">⚠️ HIGH RISK - Evasion Predicted</h3>
                        <p style="font-size: 1.1rem; margin-top: 10px;">
                            <strong>Stability Score:</strong> {stability:.4f} < {EVASION_THRESHOLD}
                        </p>
                        <p style="margin-top: 10px;">The AI is confused! Its reasoning has shifted significantly.</p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="prediction-box low-risk">
                        <h3 style="margin: 0; color: #28a745;">✅ LOW RISK - Detection Expected</h3>
                        <p style="font-size: 1.1rem; margin-top: 10px;">
                            <strong>Stability Score:</strong> {stability:.4f} ≥ {EVASION_THRESHOLD}
                        </p>
                        <p style="margin-top: 10px;">The AI is stable. Its reasoning is consistent.</p>
                    </div>
                    """, unsafe_allow_html=True)

            with truth_col:
                st.markdown('<div class="section-header">🕵️ True Outcome</div>', unsafe_allow_html=True)
                with st.expander("Click to Reveal Truth", expanded=False):
                    if evaded:
                        st.markdown(f"""<div class="success-box">
                        <h3 style="color: #28a745; margin: 0;">✅ EVASION SUCCESSFUL</h3>
                        <p>The AI thinks this is <strong>BENIGN</strong> ({mut_conf:.1%}).</p>
                        <hr>
                        <p><strong>Was our Prediction Correct?</strong> {'YES' if prediction_correct else 'NO'}</p>
                        </div>""", unsafe_allow_html=True)
                    else:
                        st.markdown(f"""<div class="fail-box">
                        <h3 style="color: #dc3545; margin: 0;">❌ EVASION FAILED</h3>
                        <p>The AI still thinks this is <strong>PHISHING</strong> ({mut_conf:.1%}).</p>
                        <hr>
                        <p><strong>Was our Prediction Correct?</strong> {'YES' if prediction_correct else 'NO'}</p>
                        </div>""", unsafe_allow_html=True)

            # --- VIEW 3: DEEP DIVE (HIDDEN) ---
            st.markdown("---")
            with st.expander("📉 Deep Dive: Why did the AI panic? (Top 10 Feature Comparison)"):
                st.caption("Compare what the AI cared about BEFORE vs. AFTER. A chaotic change means the model is guessing.")
                
                # Get Top 10 Indices
                idx_orig = np.argsort(np.abs(shap_orig))[-10:][::-1]
                idx_mut = np.argsort(np.abs(shap_mut))[-10:][::-1]
                
                # Create sets for checking Added/Removed status
                top10_orig_names = [feature_cols[i] for i in idx_orig]
                top10_mut_names = [feature_cols[i] for i in idx_mut]

                # Create Comparison Data
                comparison_data = []
                for i in range(10):
                    # Original Feature Info
                    f_name_orig = top10_orig_names[i]
                    val_orig = abs(shap_orig[idx_orig[i]])
                    
                    # Mark if DROPPED
                    if f_name_orig not in top10_mut_names:
                        display_orig = f"{f_name_orig} (❌ Dropped)"
                    else:
                        display_orig = f_name_orig

                    # Mutated Feature Info
                    f_name_mut = top10_mut_names[i]
                    val_mut = abs(shap_mut[idx_mut[i]])

                    # Mark if NEW
                    if f_name_mut not in top10_orig_names:
                        display_mut = f"{f_name_mut} (✨ New)"
                    else:
                        display_mut = f_name_mut
                    
                    comparison_data.append({
                        "Rank": i + 1,
                        "Original Feature (Before)": display_orig,
                        "Impact A": val_orig,
                        "Mutated Feature (After)": display_mut,
                        "Impact B": val_mut
                    })
                
                df_compare = pd.DataFrame(comparison_data)
                
                # Stylize columns for better readability
                st.dataframe(
                    df_compare,
                    column_config={
                        "Impact A": st.column_config.ProgressColumn("Importance", min_value=0, max_value=max(df_compare["Impact A"].max(), df_compare["Impact B"].max()), format="%.3f"),
                        "Impact B": st.column_config.ProgressColumn("Importance", min_value=0, max_value=max(df_compare["Impact A"].max(), df_compare["Impact B"].max()), format="%.3f"),
                    },
                    hide_index=True,
                    use_container_width=True
                )
                
                st.info("💡 **Key:** (✨ New) = Feature entered Top 10 | (❌ Dropped) = Feature left Top 10")

if __name__ == "__main__":
    main()