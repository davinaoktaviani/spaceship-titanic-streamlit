import streamlit as st
import pickle
import pandas as pd

st.title("ASG 04 MD - Davina - Spaceship Titanic Model Deployment")

model = pickle.load(open("models/model.pkl","rb"))

st.header("Passenger Features")

Age = st.number_input("Age",0,100,30)
RoomService = st.number_input("RoomService",0,5000,0)
FoodCourt = st.number_input("FoodCourt",0,5000,0)
ShoppingMall = st.number_input("ShoppingMall",0,5000,0)
Spa = st.number_input("Spa",0,5000,0)
VRDeck = st.number_input("VRDeck",0,5000,0)

HomePlanet = st.selectbox("HomePlanet",[0,1,2])
CryoSleep = st.selectbox("CryoSleep",[0,1])
Destination = st.selectbox("Destination",[0,1,2])
VIP = st.selectbox("VIP",[0,1])
Deck = st.selectbox("Deck",[0,1,2,3,4])
Side = st.selectbox("Side",[0,1])
Cabin_num = st.number_input("Cabin_num",0,2000,0)

# Feature engineering (sama seperti training)
TotalSpending = RoomService + FoodCourt + ShoppingMall + Spa + VRDeck
Group_size = 1

input_data = pd.DataFrame({
"Age":[Age],
"RoomService":[RoomService],
"FoodCourt":[FoodCourt],
"ShoppingMall":[ShoppingMall],
"Spa":[Spa],
"VRDeck":[VRDeck],
"Cabin_num":[Cabin_num],
"Group_size":[Group_size],
"TotalSpending":[TotalSpending],
"HomePlanet":[HomePlanet],
"CryoSleep":[CryoSleep],
"Destination":[Destination],
"VIP":[VIP],
"Deck":[Deck],
"Side":[Side]
})

if st.button("Predict"):

    # urutan feature yang dipakai saat training
    feature_order = [
        "HomePlanet",
        "CryoSleep",
        "Destination",
        "VIP",
        "Deck",
        "Side",
        "Age",
        "RoomService",
        "FoodCourt",
        "ShoppingMall",
        "Spa",
        "VRDeck",
        "Cabin_num",
        "Group_size",
        "TotalSpending"
    ]

    input_data = input_data[feature_order]

    prediction = model.predict(input_data)

    if prediction[0] == 1:
        st.success("Passenger Transported")
    else:
        st.error("Passenger Not Transported")