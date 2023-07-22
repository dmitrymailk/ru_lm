import streamlit as st
import time
import numpy as np
import streamlit.components.v1 as components


class MyApp:
    def __init__(self) -> None:
        self.progress_text = "Operation in progress. Please wait."

        self.register_value("total", 0)
        self.register_value("text", "Streamlit is **_really_ cool**.")
        self.prompts = [
            "promt1",
            "promt2",
        ]
        self.build_interface()

    def build_interface(self):
        with st.container():
            st.markdown(st.session_state.text)

            components.html(
                f"""
                <link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/4.0.0/css/bootstrap.min.css" integrity="sha384-Gn5384xqQ1aoWXA+058RXPxPg6fy4IWvTNh0E263XmFcJlSAwiGgFAW/dAiS6JXm" crossorigin="anonymous">
                <div id="accordion">
                <div class="card" style="width: 500px; overflow-y: scroll; height: 300px">
                    <div class="card-body">
                        <code class="card-text">{st.session_state.text} </code>
                    </div>
                </div>
                """,
                height=300,
            )

            col1, col2 = st.columns(2)

            with col1:
                st.button("Bad", on_click=self.test_func)

            with col2:
                st.button("Good", on_click=self.test_func)

    def test_func(self):
        print("Do smt")
        self.set_value("text", self.prompts[st.session_state.total])
        st.session_state.total += 1
        print(st.session_state.total)

    def register_value(self, var_name: str, value):
        if var_name not in st.session_state:
            st.session_state[var_name] = value

    def set_value(self, var_name: str, value):
        if var_name in st.session_state:
            st.session_state[var_name] = value


app = MyApp()
