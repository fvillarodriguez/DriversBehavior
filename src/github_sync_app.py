import streamlit as st
import src.git_sync as git_sync

def main(set_page_config: bool = False, show_exit_button: bool = False) -> None:
    if set_page_config:
        st.set_page_config(page_title="GitHub Sync", layout="wide")

    st.title("GitHub Synchronization")
    st.markdown("Sync your local code with the remote GitHub repository.")

    # Verificar si es un repo git
    if not git_sync.is_git_repo():
        st.error("⚠️ La carpeta actual NO es un repositorio Git.")
        st.info("Configura la URL remota para inicializar y descargar el código.")
        
        with st.expander("Inicializar Repositorio", expanded=True):
            remote_url = st.text_input("URL del Repositorio Remoto (e.g., git@github.com:user/repo.git)")
            if st.button("Inicializar y Sincronizar", type="primary"):
                if not remote_url:
                    st.error("Por favor ingresa una URL remota.")
                else:
                    with st.spinner("Inicializando repositorio..."):
                        success, logs = git_sync.initialize_repo(remote_url)
                    
                    st.session_state["sync_logs"] = logs
                    if success:
                        st.success("Repositorio inicializado correctamente! Recarga la página.")
                        if st.button("Recargar"):
                            st.rerun()
                    else:
                        st.error("Hubo un error en la inicialización. Revisa los logs.")
        
        st.subheader("Logs")
        logs = st.session_state.get("sync_logs", [])
        if logs:
            st.code("\n".join(logs), language="text")
        
        return # Detener renderizado del resto si no es repo

    # Sección de Configuración SSH (Única soportada ahora)
    st.subheader("Configuración SSH")
    
    ssh_key = git_sync.get_ssh_public_key()
    
    if ssh_key:
        st.success("✅ Clave SSH detectada.")
        st.text_area("Tu Clave Pública (Copia esto a GitHub -> Settings -> SSH Keys)", value=ssh_key, height=100, disabled=True)
        st.markdown("[Ir a GitHub SSH Settings](https://github.com/settings/keys)")
    else:
        st.warning("⚠️ No se detectó una clave SSH predeterminada (id_ed25519 o id_rsa).")
    
    with st.expander("Generar Nueva Clave SSH"):
        email_input = st.text_input("Email para la clave (opcional)")
        if st.button("Generar Clave SSH"):
            if ssh_key:
                st.error("Ya existe una clave SSH. Por seguridad, esta app no la sobrescribirá automáticamente. Elimínela manualmente si desea regenerarla.")
            else:
                success, msg = git_sync.generate_ssh_key(email=email_input)
                if success:
                    st.success(msg)
                    st.rerun() # Recargar para mostrar la clave
                else:
                    st.error(msg)

    st.divider()

    col1, col2 = st.columns([1, 4])
    
    with col1:
        if st.button("🔄 Sync Now", type="primary", use_container_width=True):
            with st.spinner("Synchronizing with GitHub..."):
                success, logs = git_sync.sync_with_github()
                
            if success:
                st.success("Synchronization successful!")
            else:
                st.error("Synchronization failed. check logs below.")
            
            st.session_state["sync_logs"] = logs

    with col2:
        st.subheader("Logs")
        logs = st.session_state.get("sync_logs", [])
        if logs:
            log_text = "\n".join(logs)
            st.code(log_text, language="text")
        else:
            st.info("No logs available. Run sync to see output.")

    if show_exit_button:
        if st.button("Exit"):
            st.stop()

if __name__ == "__main__":
    main(set_page_config=True)
