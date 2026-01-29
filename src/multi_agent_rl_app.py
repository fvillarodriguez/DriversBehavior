import streamlit as st

def main(set_page_config: bool = True, show_exit_button: bool = True) -> None:
    if set_page_config:
        st.set_page_config(page_title="Multi Agent RL", layout="wide")
    
    st.title("Multi Agent RL")
    st.write("Bienvenido a la sección de Multi Agent RL.")
    
    # Aquí irá el contenido de la nueva sección
    
if __name__ == "__main__":
    main()
