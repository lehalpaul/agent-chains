from langchain.prompts import PromptTemplate
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI

from langchain.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
      HumanMessagePromptTemplate,
    MessagesPlaceholder
)
from langchain.agents import initialize_agent, Tool
from langchain.agents import AgentType
from langchain.memory import ConversationBufferMemory
from langchain.chains import LLMChain
from langchain.document_loaders import CSVLoader
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.chains import RetrievalQA

import streamlit as st
import os
import openai
def main():
    
    openai.api_key=os.environ["OPENAI_API_KEY"]
    st.title("Auto Dealership AI Assistant")

   
    if "messages" not in st.session_state:
        st.session_state.messages = []

   
    with st.form("user_input_form"):
        user_input = st.text_input("How can I assist you today?", key="input")
        submit_button = st.form_submit_button("Submit")

    if submit_button and user_input:
        
        st.session_state.messages.append({"role": "user", "content": user_input})
        response = run_conversation(user_input)
        
        st.session_state.messages.append({"role": "assistant", "content": response})

   
    for message in reversed(st.session_state.messages):
        with st.chat_message(message["role"]):
            st.markdown(message["content"])



def run_conversation(question):

  llm= ChatOpenAI(temperature=0.9,model="gpt-4",streaming=True)

  template="""
  You are an AI assistant tasked with creating phone scripts for sales representatives at Xanadu Automotive, focusing on engaging potential customers based on their expressed interests and provided comments. Your scripts will help manage inbound leads by responding accurately and efficiently.

  Task:
  Generate a phone script for an inbound lead. Use the following information to craft a friendly and professional script that includes necessary sections like greeting, vehicle information, and closing statements.

  Lead Information:
      • Prospect Status: [Prospect Status]
      • Request Date: 2024-05-23T15:49:48-05:00

  Vehicles of Interest:
      1. Interest: Buy, Status: Used, Year: 2022, Make: Kia, Model: Telluride, Trim: EX
      2. Interest: Trade-in, Status: New, Odometer Status: Replaced, Odometer Units: mi

  Customer Details:
      •	First Name: Tyler
  •	Last Name: Uebele
  •	Phone: 704-443-8469
  •	Address: 204 E Franklin, Monroe, NC 28212

  Vendor Name: Xanadu Automotive (TESTER)

  Provider Details:
      • ID: 1
      • Source: Unknown

  Lead Inventory Items:
      1. Inventory ID: 7302674, VIN: 5XYP34HC0NG195109, New/Used: Used, Year: 2022, Make: Kia,
        Model: Telluride, Transmission: Automatic, Odometer: 13,693 mi, Color: Sangria, Price: $34,695, Description:10-Speed Automatic, 4WD, Black Cloth.
        Features:Preferred Equipment Group 1LT|3.23 Rear Axle Ratio|Auto-Locking Rear Differential|Wheels: 17" x 8" Bright Silver Painted Aluminum|Wheels: 18" x 8.5" Bright Silver Painted Aluminum|Wheels: 20" x 9" Painted Aluminum|40/20/40 Front Split-Bench Seat|Cloth Seat Trim|10-Way Power Driver Seat w/Lumbar|Convenience Package II|All Star Edition Plus|Radio: Chevrolet Infotainment 3 Premium System|Electronic Cruise Control|Electric Rear-Window Defogger|Color-Keyed Carpeting Floor Covering|All-Weather Floor Liner (LPO) (AAK)|120-Volt Interior Power Outlet|Z71 Off-Road & Protection Package|Protection Package|Chevytec Spray-On Black Bedliner|Deep-Tinted Glass|Front License Plate Kit|LED Cargo Area Lighting|EZ Lift Power Lock & Release Tailgate|Rear Wheelhouse Liners|6" Rectangular Chrome Tubular Assist Steps (LPO)|Front Black Bowtie Emblem (LPO)|Standard Suspension Package|High Capacity Suspension Package|Z71 Off-Road Package|Trailering Package|Integrated Trailer Brake Controller|Remote Start Package|Skid Plates|Heavy-Duty Air Filter|SiriusXM w/360L|Power Sliding Rear Window w/Rear Defogger|Rear 60/40 Folding Bench Seat (Folds Up)|Chevrolet Connected Access Capable|Power Front Windows w/Passenger Express Down|Power Rear Windows w/Express Down|Keyless Open & Start|Power Front Windows w/Driver Express Up/Down|Front Rubberized Vinyl Floor Mats|Rear Rubberized-Vinyl Floor Mats|Bluetooth¬Æ For Phone|Remote Vehicle Starter System|Dual-Zone Automatic Climate Control|Hitch Guidance|Inside Rear-View Mirror w/Tilt|Heated Power-Adjustable Outside Mirrors|Chrome Mirror Caps|Hill Descent Control|Heated Driver & Front Outboard Passenger Seats|External Engine Oil Cooler|120-Volt Bed Mounted Power Outlet|Heated Steering Wheel|Auxiliary External Transmission Oil Cooler|220 Amp Alternator|170 Amp Alternator|Electrical Steering Column Lock|Dual Exhaust w/Polished Outlets|Wrapped Steering Wheel|Single-Speed Transfer Case|2-Speed Transfer Case|Convenience Package|All-Star Edition|Chevy Safety Assist|Hitch Guidance w/Hitch View|Standard Tailgate|IntelliBeam Automatic High Beam On/Off|Dual Rear USB Ports (Charge Only)|12.3" Multicolor Reconfigurable Digital Display|OnStar & Chevrolet Connected Services Capable|Following Distance Indicator|In-Vehicle Trailering System App|Forward Collision Alert|Universal Home Remote|Lane Keep Assist w/Lane Departure Warning|Automatic Emergency Braking|Steering Wheel Audio Controls|Front Pedestrian Braking|Theft Deterrent System (Unauthorized Entry)|HD Rear Vision Camera|Front Frame-Mounted Black Recovery Hooks|Wi-Fi Hot Spot Capable|Auto High-beam Headlights|AM/FM radio: SiriusXM with 360L|Premium audio system: Chevrolet Infotainment 3 Premium|Standard fuel economy fuel type: gasoline|4-Wheel Disc Brakes|6 Speakers|Air Conditioning|Electronic Stability Control|Tachometer|Voltmeter|ABS brakes|Alloy wheels|Automatic temperature control|Brake assist|Bumpers: chrome|Delay-off headlights|Driver door bin|Driver vanity mirror|Dual front impact airbags|Dual front side impact airbags|Front anti-roll bar|Front dual zone A/C|Front reading lights|Front wheel independent suspension|Fully automatic headlights|Heated door mirrors|Heated front seats|Heated steering wheel|Illuminated entry|Low tire pressure warning|Occupant sensing airbag|Outside temperature display|Overhead airbag|Overhead console|Panic alarm|Passenger door bin|Passenger vanity mirror|Power door mirrors|Power driver seat|Power steering|Power windows|Radio data system|Rear reading lights|Rear step bumper|Rear window defroster|Remote keyless entry|Security system|Speed control|Speed-sensing steering|Split folding rear seat|Steering wheel mounted audio controls|Telescoping steering wheel|Tilt steering wheel|
        Traction control|Trip computer|Variably intermittent wipers|Compass|Front Center Armrest w/Storage
      2. Inventory ID: 7648318, VIN: 5XYP3DHC5LG079451, New/Used: Used, Year: 2020, Make: Kia,
        Model: Telluride, Transmission: 8-Speed A/T, Odometer: 77,455 mi, Color: Ebony Black, Price: $25,830, Description:19/27 City/Highway MPG   ,Description:Preferred Equipment Group 3LT|3.47 Final Drive Axle Ratio|3.49 Final Drive Axle Ratio|Wheels: 18" Grazen Metallic Aluminum|Wheels: 18" High Gloss Black Painted Aluminum|Black Lug Nut & Wheel Lock Kit (LPO)|Perforated Leather-Appointed Seat Trim|Ride & Handling Suspension|Driver Confidence Package|Sound & Technology Package|Not Equipped w/Rear Park Assist|Radio: Chevrolet Infotainment 3 Plus System|Radio: Chevrolet Infotainment 3 Premium System|Power Panoramic Tilt-Sliding Sunroof|Midnight/Sport Edition|Front & Rear Black Bowties|8-Way Power Driver Seat Adjuster|6-Way Power Front Passenger Seat Adjuster|Power Driver Lumbar Control|Inside Rear-View Auto-Dimming Mirror|Outside Heated Power-Adjustable Body-Color Mirrors|Wireless Charging|Heated Driver & Front Passenger Seats|120-Volt Power Outlet|Adaptive Cruise Control|170 Amp Alternator|155 Amp Alternator|2 USB Data Ports w/SD Card Reader|Rear Power Programmable Liftgate|SiriusXM w/360L|Rear Park Assist w/Audible Warning|Rear Cross Traffic Alert|Universal Home Remote|Enhanced Automatic Emergency Braking|Lane Change Alert w/Side Blind Zone Alert|Bose Premium 8-Speaker Audio System Feature|6-Speaker Audio System Feature|HD Surround Vision|Black Roof-Mounted Side Rails|Variably intermittent wipers|Front beverage holders|Auto-dimming Rear-View mirror|Child-Seat-Sensing Airbag|Compass|Auto High-beam Headlights|AM/FM radio: SiriusXM with 360L|Emergency communication system: OnStar and Chevrolet connected services capable|Premium audio system: Chevrolet Infotainment 3 Plus|Apple CarPlay/Android Auto|4-Wheel Disc Brakes|6 Speakers|Air Conditioning|Electronic Stability Control|Front Bucket Seats|Front Center Armrest|Leather Shift Knob|Power Liftgate|Spoiler|Tachometer|Voltmeter|ABS brakes|Alloy wheels|Auto-dimming door mirrors|Automatic temperature control|Brake assist|Bumpers: body-color|Delay-off headlights|Driver door bin|Driver vanity mirror|Dual front impact airbags|Dual front side impact airbags|Four wheel independent suspension|Front anti-roll bar|Front dual zone A/C|Front reading lights|Fully automatic headlights|Garage door transmitter|Heated door mirrors|Heated front seats|Illuminated entry|Knee airbag|Leather steering wheel|Low tire pressure warning|Occupant sensing airbag|Outside temperature display|Overhead airbag|Overhead console|Panic alarm|Passenger door bin|Passenger vanity mirror|Power door mirrors|Power driver seat|Power passenger seat|Power steering|Power windows|Radio data system|Rear anti-roll bar|Rear reading lights|Rear seat center armrest|Rear window defroster|Rear window wiper|Remote keyless entry|Roof rack: rails only|Security system|Speed control|Speed-sensing steering|Split folding rear seat|Steering wheel mounted audio controls|Telescoping steering wheel|Tilt steering wheel|Traction control|Trip computer|Turn signal indicator mirrors

  Returning Customer: False
  Lead Type: Sales

  Comments: {question}

  Instructions:
  Create a script with the following sections:
      - Greeting
      - Introduction (Incorporate vehicles of interest and relevant customer comments)
      - Returning Customer Acknowledgement (if applicable)
      - Questions to Understand Lead’s Needs
      - Closing Statement (Include scheduling for a follow-up appointment or test drive)
  Ensure all scripts are formatted with clear labels for each section.
  """

  prompt_temp = ChatPromptTemplate.from_template(template)
  llm = ChatOpenAI(model_name="gpt-4", temperature=0.9)

  memory = ConversationBufferMemory(memory_key= "chat_history",return_messages=True)
  conversation = LLMChain(llm=llm,prompt=prompt_temp,verbose=False,memory=memory)

  loader = CSVLoader(file_path="cronicchevrolet.csv")
  data = loader.load()
  vectordb = FAISS.from_documents(data, OpenAIEmbeddings())
  retriever = vectordb.as_retriever()

  temp = """
 You're an AI assistant specialized in auto car dealership inventory management. 
    Given the context from a CSV file, {context}, your task is to provide an accurate answer to the user's question, {question}, based on the data available in the CSV.
        - Greet the customer first and maintain a friendly yet professional tone.
        - Provide detailed and precise answers with all relevant data based on the question asked.
        - Avoid single responses like "Yes" or "No".
        - Do not use the phrase "based on the data available in the CSV".
        - Offer additional assistance by asking if there is anything else the customer needs help with.
        - If the data needed to answer the question is missing or unclear, apologize and provide any related information that might be helpful.

    Example interaction:
    Customer: "Do you have any SUVs available?"
    Assistant: "Hello! Thank you for your inquiry. We currently have several SUVs available, including models such as the Toyota RAV4, Honda CR-V, and Ford Explorer. Could you please specify any particular features or brands you are interested in?"

    Professional response template:
    "Hello [Customer's Name], thank you for reaching out! [Answer to the question with detailed and relevant data.] If there is anything else you need assistance with, please let me know."

   """
  prompt_temp = ChatPromptTemplate.from_template(temp)
  llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.9)

  from langchain.schema.runnable import RunnablePassthrough
  from langchain.schema.output_parser import StrOutputParser

  rag_chain = (
      {"context": retriever,"question": RunnablePassthrough()}
      | prompt_temp
      | llm
      | StrOutputParser()
  )

 tools = [
      Tool.from_function(
        name = "Script Generation",
        func = conversation.run,
        description = """
        Useful for when you need to answer questions ,when user ask comments to generate script .
        <user>: Provide me script for Kia Telluride 2022
        <assistant>: check template

        """,
    ),

    Tool.from_function(
    name="Inventory",
   func=rag_chain.invoke,
    description="""
     Useful for when you need to answer questions about car data from inventory.
     <user>:What is the MSRP and current dealership price for the 2024 GMC Sierra 1500 Denali Ultimate?
     <assistant>: Answer based on prompt for  csv .
    """
    ),
  ]

  memory = ConversationBufferMemory(memory_key="chat_history")
  agent = initialize_agent(
      agent='conversational-react-description',
      tools=tools,
      llm=llm,
      verbose=False,
      max_iterations=10,
      memory=memory,
      handle_parsing_errors=True 
  )
  response =agent.run(question)

  return response


if __name__ == "__main__":
      main()
