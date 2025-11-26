import streamlit as st
import joblib
import numpy as np

# Load your real skill extractor
vectorizer = joblib.load("vectorizer.pkl")

st.title("Skill Extractor")
st.caption("Trained on 11,543 real freelance job postings")

text = st.text_area("Paste any job description below", height=220)

if st.button("Extract Skills", type="primary"):
    if text.strip():
        X = vectorizer.transform([text])
        scores = X.toarray()[0]
        features = vectorizer.get_feature_names_out()
        
        top_idx = np.argsort(scores)[-30:][::-1]
        skills = []
        for i in top_idx:
            word = features[i]
            if scores[i] > 0.08 and len(word) > 2:
                if word not in ["experience", "work", "ability", "team", "project", "good", "strong", "excellent", "year", "skill"]:
                    skills.append(word.title())
        
        if skills:
            st.success(f"Found {len(skills)} skills:")
            cols = st.columns(4)
            for i, skill in enumerate(skills[:24]):
                cols[i % 4].write(f"• **{skill}**")
        else:
            st.info("No strong skills detected")
    else:
        st.warning("Paste a job description first")