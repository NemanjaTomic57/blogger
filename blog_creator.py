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
            topics. Please read through the provided text and create a
            well-structured, informative, and engaging blog post suitable for a
            general audience interested in IT. The blog post should focus on
            the key points from the document, explain complex concepts clearly,
            and highlight any insights or recommendations. Make sure the post
            flows logically, uses headings and subheadings where appropriate,
            and ends with a conclusion or call to action. Be sure to format the
            text for easy reading, and avoid jargon or overly technical
            language unless necessary.
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
