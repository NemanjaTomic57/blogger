import PyPDF2
from langchain.chat_models import init_chat_model
from langgraph.graph import MessagesState, StateGraph
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv

load_dotenv()

def load_chat_model():
    return (init_chat_model("mistral-large-latest", model_provider="mistralai"))

def pdf_to_text(file_path):
    with open(file_path, "rb") as file:
        reader = PyPDF2.PdfReader(file)
        text = ""
        for page in reader.pages:
            text += page.extract_text()
    return text

model = load_chat_model()

workflow = StateGraph(state_schema=MessagesState)
prompt_template = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            I have a text document that contains detailed information about IT
            topics. Please extract the top 50 most important points from the
            document. The points should be ordered by importance, starting with
            the most crucial. Provide a concise summary of each point, with
            enough detail for a general audience to understand its
            significance. Avoid overly technical language, but ensure clarity.
            For each point, provide a 10-word excerpt from the document to
            indicate where the information was found. Be sure to focus on
            extracting key insights, recommendations, or critical information
            that would be most relevant to a general audience interested in IT.
            """
        ),
        ("user", "{text}")
    ]
)

file_path = "wireguard.pdf"
file_text = pdf_to_text(file_path)

# Prepare the input message to pass to the model
prompt = prompt_template.invoke({
    "text": file_text
})

# Here, you would invoke the model with the prepared input
response = model.invoke(prompt)
print(response.content)
