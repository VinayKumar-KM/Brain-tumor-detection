# session_state.py

import streamlit as st

class SessionState:
    def __init__(self, **kwargs):
        self._state = kwargs

    def __getattr__(self, attr):
        return self._state.get(attr, None)

    def __setattr__(self, attr, value):
        if attr == '_state':
            super().__setattr__(attr, value)
        else:
            self._state[attr] = value

# Function to get session state
def get_session_state():
    session = st.session_state
    if not hasattr(session, '_custom_session_state'):
        session._custom_session_state = SessionState()
    return session._custom_session_state
