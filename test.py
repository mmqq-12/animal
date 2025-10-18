import streamlit as st
import pandas as pd
#设定网页标题
st.title( 'My App')
st.write("My first app")
st.write(pd. DataFrame({
    'first co lumn': [1, 2,3,4],
    'second column': [10, 20,30, 401]
}))