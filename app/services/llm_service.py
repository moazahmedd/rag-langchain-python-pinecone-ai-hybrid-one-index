from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.schema import Document
from typing import List, Dict, Any
from config.settings import Settings

class LLMService:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.llm = ChatOpenAI(
            model_name=settings.LLM_MODEL,
            temperature=settings.LLM_TEMPERATURE
        )
        
        # Define prompt template using ChatPromptTemplate
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a helpful AI assistant. Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer. Ensure you provide atleast 200 words."),
            ("human", "Context: {context}\n\nQuestion: {question}")
        ])
        
        # Create chain using LCEL
        self.chain = (
            self.prompt 
            | self.llm 
            | StrOutputParser()
        )
        
        # Wrap chain to include input in output
        self.outer_chain = RunnablePassthrough().assign(text=self.chain)

    def get_structured_answer(self, query: str, documents: List[Document]) -> Dict[str, Any]:
        """
        Get a structured answer with sources from the LLM
        """
        print("llm_service: Getting structured answer...")
        # Prepare context from documents
        context = "\n\n".join([doc.page_content for doc in documents])
        sources = []
        
        # Format sources with required fields
        for doc in documents:
            source = {
                "content": doc.page_content,
                "page": doc.metadata.get("page", "N/A"),
                "id": doc.id
            }
            sources.append(source)

        # Run the chain
        response = self.outer_chain.invoke({
            "question": query,
            "context": context
        })

        return {
            "answer": response["text"],
            "sources": sources
        } 