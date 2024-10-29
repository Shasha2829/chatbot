import os
import streamlit as st
from typing import Generator
from groq import Groq, AuthenticationError, APIError
import json

st.set_page_config(page_icon="💬", layout="wide", page_title="Shaineze")

def icon(emoji: str):
    """Affiche un emoji comme icône de page de style Notion."""
    st.write(
        f'<span style="font-size: 78px; line-height: 1">{emoji}</span>',
        unsafe_allow_html=True,
    )

icon("🗿")
st.subheader("Ia Ultime")

# Utilisation de la clé API depuis l'environnement
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    st.error("Clé API Groq manquante.")
else:
    try:
        client = Groq(api_key=GROQ_API_KEY)
        st.success("Client Groq initialisé avec succès.")
    except AuthenticationError:
        st.error("Erreur d'authentification. Vérifiez votre clé API.", icon="🚨")
        st.stop()
    except Exception as e:
        st.error(f"Erreur lors de l'initialisation du client Groq : {str(e)}", icon="🚨")
        st.stop()
        
    # Initialisation des variables de session
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "selected_model" not in st.session_state:
        st.session_state.selected_model = None

    # Liste des modèles
    models = {
        "llama-guard-3-8b": {"name": "Llama Guard 3 8B", "tokens": 8192, "developer": "Meta"},
        "llava-v1.5-7b-4096-preview": {"name": "LLaVA 1.5 7B", "tokens": 4096, "developer": "Haotian Liu"},
        "llama3-70b-8192": {"name": "Meta Llama 3 70B", "tokens": 8192, "developer": "Meta"},
        "llama3-8b-8192": {"name": "Meta Llama 3 8B", "tokens": 8192, "developer": "Meta"},
        "mixtral-8x7b-32768": {"name": "Mixtral 8x7B", "tokens": 32768, "developer": "Mistral"},
        "whisper-large-v3": {"name": "Whisper Large V3", "tokens": 25000, "developer": "OpenAI"},
        "whisper-large-v3-turbo": {"name": "Whisper Large V3 Turbo", "tokens": 25000, "developer": "OpenAI"},
    }

    # Sélection du modèle et max_tokens
    col1, col2 = st.columns(2)
    with col1:
        model_option = st.selectbox(
            "Choisissez un modèle :",
            options=list(models.keys()),
            format_func=lambda x: models[x]["name"],
            index=0
        )

    if st.session_state.selected_model != model_option:
        st.session_state.messages = []
        st.session_state.selected_model = model_option

    max_tokens_range = models[model_option]["tokens"]
    with col2:
        max_tokens = st.slider(
            "Max Tokens :",
            min_value=512,
            max_value=max_tokens_range,
            value=min(8000, max_tokens_range),
            step=512,
            help=f"Ajustez le nombre maximum de tokens pour la réponse du modèle. Max pour le modèle sélectionné : {max_tokens_range}"
        )

    st.write(f"Modèle sélectionné : **{models[model_option]['name']}**")
    st.write(f"Développeur : {models[model_option]['developer']}")
    st.write(f"Nombre maximal de tokens : {models[model_option]['tokens']}")

    if st.button("Effacer l'historique"):
        st.session_state.messages = []

    for message in st.session_state.messages:
        avatar = '🐉' if message["role"] == "assistant" else '👨‍👩‍👧‍👧'
        with st.chat_message(message["role"], avatar=avatar):
            st.markdown(message["content"])

    def generate_chat_responses(chat_completion) -> Generator[str, None, None]:
        full_content = ""
        for chunk in chat_completion:
            if chunk.choices[0].delta.content:
                full_content += chunk.choices[0].delta.content
        yield full_content

    if prompt := st.chat_input("Entrez votre message ici..."):
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.chat_message("user", avatar='👤'):
            st.markdown(prompt)

        try:
            with st.spinner('Génération de la réponse...'):
                progress_bar = st.progress(0)
                chat_completion = client.chat.completions.create(
                    model=model_option,
                    messages=st.session_state.messages,
                    max_tokens=max_tokens,
                    stream=True
                )
                full_response = ""
                with st.chat_message("assistant", avatar="👽"):
                    for i, chunk in enumerate(generate_chat_responses(chat_completion)):
                        full_response += chunk
                        st.markdown(chunk)
                        progress_bar.progress((i + 1) / 100)
                
                progress_bar.empty()
                st.session_state.messages.append({"role": "assistant", "content": full_response})

        except Exception as e:
            st.error(f"Erreur : {e}")

    if st.button("Télécharger l'historique"):
        history = [{"role": msg["role"], "content": msg["content"]} for msg in st.session_state.messages]
        history_json = json.dumps(history, ensure_ascii=False, indent=2)
        st.download_button(
            label="Télécharger l'historique",
            data=history_json,
            file_name="historique_conversation.json",
            mime="application/json"
        )

    # Suppression de la fonctionnalité de mode nuit
    # Suppression de l'affichage du graphique exemple

    tone = st.selectbox("Choisissez le ton de la réponse", ["Formelle", "Amicale", "Neutre"])
    length = st.slider("Longueur de la réponse", min_value=50, max_value=500, step=50, value=150)

    if st.button("Obtenir un résumé"):
        summary_prompt = "Résumez cette conversation: " + " ".join([msg["content"] for msg in st.session_state.messages])
        try:
            summary_completion = client.chat.completions.create(
                model=model_option,
                messages=[{"role": "user", "content": summary_prompt}],
                max_tokens=500
            )
            st.write("Résumé de la conversation :")
            st.markdown(summary_completion.choices[0].message["content"])
        except Exception as e:
            st.error(f"Erreur lors de la génération du résumé : {e}")

