__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
import streamlit as st
import os
import traceback
import logging
from crewai import Agent, Task, Crew, Process
from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
from langchain_anthropic import ChatAnthropic
from dotenv import load_dotenv
from crewai_tools import *

import warnings
warnings.filterwarnings("ignore", category=SyntaxWarning)

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Charger les variables d'environnement
load_dotenv()

# Fonction pour récupérer les clés API
def get_api_key(key_name):
    try:
        return st.secrets.get(key_name) or os.getenv(key_name)
    except Exception:
        return os.getenv(key_name)

# Configuration des LLM (similaire à votre code précédent)
def create_llm(provider_and_model, temperature=0.1):
    # Votre logique de création de LLM existante
    providers = {
        "OpenAI": lambda: ChatOpenAI(
            api_key=get_api_key("OPENAI_API_KEY"), 
            model=provider_and_model.split(": ")[1], 
            temperature=temperature
        ),
    }
    
    provider = provider_and_model.split(": ")[0]
    if provider in providers:
        return providers[provider]()
    else:
        raise ValueError(f"Provider {provider} non supporté")
    
    # Instructions spécifiques sur l'utilisation des outils
TOOLS_INSTRUCTIONS = """
IMPORTANT: Pour les outils de communication et délégation, utilisez toujours des chaînes de caractères simples, pas des objets ou dictionnaires complexes.

Format CORRECT pour la délégation:
{
  "task": "Description de la tâche en texte simple",
  "context": "Contexte de la tâche en texte simple",
  "coworker": "Nom du collègue"
}

Format CORRECT pour les questions:
{
  "question": "Votre question en texte simple",
  "context": "Contexte de la question en texte simple",
  "coworker": "Nom du collègue"
}

N'utilisez JAMAIS ce format incorrect:
{
  "task": {"description": "...", "type": "str"},
  "context": {"description": "...", "type": "str"}
}
"""

def create_agents(main_topic, subtopics):
    """
    Création dynamique des agents basée sur le sujet principal
    """
    logger.info(f"Création des agents pour le sujet: {main_topic}")

    agents = [
        Agent(
            role="Analyste de Recherche",
            goal=f"Analyser en profondeur le sujet : {main_topic} et {', '.join(subtopics)}",
            backstory=f"Expert dans l'analyse approfondie de sujets complexes, spécialisé sur {main_topic} et {', '.join(subtopics)}",
            verbose=True,
            allow_delegation=True,
            tools=[
                SerperDevTool(serper_api_key=get_api_key("SERPER_API_KEY")),
                WebsiteSearchTool()
            ],
            tools_instructions=TOOLS_INSTRUCTIONS,
            llm=create_llm("OpenAI: gpt-4o-mini", 0.3)
        ),
        Agent(
            role="Synthétiseur de Contenu",
            goal=f"Synthétiser les informations collectées sur {main_topic} et {', '.join(subtopics)}",
            backstory="Expert dans la transformation d'informations brutes en insights structurés et compréhensibles",
            verbose=True,
            allow_delegation=True,
            tools_instructions=TOOLS_INSTRUCTIONS,
            llm=create_llm("OpenAI: gpt-4o-mini", 0.2)
        ),
        Agent(
            role="Rédacteur de Rapport",
            goal=f"Rédiger un rapport détaillé et accessible sur {main_topic} et {', '.join(subtopics)}",
            backstory="Rédacteur professionnel capable de transformer des analyses complexes en contenu clair et engageant",
            verbose=True,
            allow_delegation=True,
            tools_instructions=TOOLS_INSTRUCTIONS,
            llm=create_llm("OpenAI: gpt-4o-mini", 0.4)
        )
    ]
    return agents

def create_tasks(agents, main_topic, subtopics):
    logger.info(f"Création des tâches pour le sujet: {main_topic}")
    tasks = [
        Task(
            description=f"Rechercher des informations sur {main_topic} et {', '.join(subtopics)}",
            expected_output="Une liste structurée de sources fiables et récentes",
            agent=agents[0]
        ),
        Task(
            description=f"Analyser en détail les données trouvées sur {main_topic} et {', '.join(subtopics)}",
            expected_output=f"Un résumé analytique et structuré des aspects clés des recherches sur {main_topic} et {', '.join(subtopics)}",
            agent=agents[1]
        ),
        Task(
            description=f"Rédiger un rapport final complet sur {main_topic} et {', '.join(subtopics)}",
            expected_output="Un rapport détaillé de 1500-2000 mots",
            agent=agents[2]
        )
    ]
    return tasks

def handle_delegation_error(e, crew):
    """
    Gestion des erreurs spécifiques à la délégation entre agents
    """
    if "Arguments validation failed" in str(e) and ("DelegateWorkToolSchema" in str(e) or "AskQuestionToolSchema" in str(e)):
        logger.warning(f"Erreur de délégation détectée: {str(e)}")
        st.warning("Erreur détectée dans la communication entre agents. Tentative de récupération...")
        
        # Mettre à jour les instructions pour tous les agents
        for agent in crew.agents:
            agent.tools_instructions = TOOLS_INSTRUCTIONS + """
            ATTENTION SUPPLÉMENTAIRE: Il y a eu une erreur dans la façon dont vous formatez vos communications.
            N'utilisez que des chaînes de caractères simples, pas des objets avec 'description' et 'type'.
            """
        
        return True
    return False

@st.cache_data(ttl=3600)
def run_analysis(main_topic, subtopics_list):
    """
    Fonction avec cache pour exécuter l'analyse complète
    """
    inputs = {
        "main_topic": main_topic,
        "subtopics": subtopics_list
    }
    
    agents = create_agents(main_topic, subtopics_list)
    tasks = create_tasks(agents, main_topic, subtopics_list)

    crew = Crew(
        agents=agents,
        tasks=tasks,
        process=Process.sequential,
        memory=True,
        cache=False,
        max_rpm=10,
        verbose=True
    )
    
    try:
        logger.info("Démarrage de l'analyse")
        result = crew.kickoff(inputs=inputs)
        return result
    except Exception as delegation_error:
        logger.error(f"Erreur lors de l'analyse: {str(delegation_error)}")
        if handle_delegation_error(delegation_error, crew):
            # Réessayer après correction
            logger.info("Tentative de récupération après erreur de délégation")
            result = crew.kickoff(inputs=inputs)
            return result
        else:
            # Si ce n'est pas une erreur de délégation connue, relancer
            raise delegation_error    

def main():
    st.title("🔍 Monitoring & Analyse de Tendances")
    
    # Input du sujet principal
    main_topic = st.text_input(
        "Quel sujet voulez-vous analyser en profondeur ?", 
        placeholder="Ex: Intelligence Artificielle, Changement Climatique, Économie Numérique..."
    )
    
    # Input des sous-thèmes
    subtopics_input = st.text_input(
        "Quels sont les sous-thèmes ou aspects spécifiques à explorer ? (séparés par des virgules)", 
        placeholder="Ex: Éthique, Impact économique, Innovations récentes..."
    )
    
    # Bouton de lancement
    if st.button("Lancer l'analyse"):
        if main_topic and subtopics_input:
            with st.spinner("Analyse en cours... Cela peut prendre quelques minutes."):
                
                try:
                    # Préparation des inputs
                    subtopics_list = [s.strip() for s in subtopics_input.split(',')]
                    
                    result = run_analysis(main_topic, subtopics_list)
                    result_text = result.raw if hasattr(result, 'raw') else str(result)   
                        

                    # Affichage des résultats
                    st.success("Analyse terminée !")
                    
                    # Expander pour le résultat final
                    with st.expander("📄 Rapport Complet", expanded=True):
                        st.write(result_text)
                    
                    # Bouton de téléchargement
                    st.download_button(
                        label="Télécharger le rapport",
                        data=result_text,
                        file_name=f"rapport_analyse_{main_topic.replace(' ', '_')}.txt",
                        mime="text/plain"
                    )
                    
                except Exception as e:
                    # Journalisation de l'erreur détaillée
                    logger.error(f"Erreur critique: {traceback.format_exc()}")
                    st.error(f"Erreur détaillée : {traceback.format_exc()}")
                    st.error(f"Type d'erreur : {type(e)}")
                    st.error(f"Détails de l'erreur : {str(e)}")
                    
        else:
            st.warning("Veuillez saisir un sujet principal et des sous-thèmes.")

if __name__ == "__main__":
    main()