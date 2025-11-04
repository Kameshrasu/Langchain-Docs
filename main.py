from langchain_groq import ChatGroq
from langchain.chat_models import init_chat_model


llm = ChatGroq(
    api_key="",
    model="llama-3.3-70b-versatile",
    temperature=0.2,
)


# normal callig method of the llm where
# after generating the text by the model then  we get the final output when we use the invoke method
response = llm.invoke("Why do parrots talk")
print(response.content)


# this below method is used for the streaming respone
for chunk in llm.stream("Why do parrots have colorful feathers?"):
    print(chunk.text, end="", flush=True)




