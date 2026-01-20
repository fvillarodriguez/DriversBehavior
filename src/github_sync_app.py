import streamlit as st
import src.git_sync as git_sync
import time

def main(set_page_config: bool = False, show_exit_button: bool = False) -> None:
    if set_page_config:
        st.set_page_config(page_title="GitHub Sync", layout="wide")

    st.title("GitHub Synchronization")
    st.markdown("Sync your local code with the remote GitHub repository.")
    
    logs = [] # Initialize logs to avoid UnboundLocalError


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
                    st.subheader("Live Logs")
                    log_placeholder = st.empty()
                    logs = []
                    
                    with st.spinner("Inicializando repositorio..."):
                        gen = git_sync.initialize_repo_stream(remote_url)
                        success = False
                        try:
                            while True:
                                msg = next(gen)
                                logs.append(msg)
                                log_placeholder.code("\n".join(logs), language="text")
                        except StopIteration as e:
                            success = e.value
                    
                    st.session_state["sync_logs"] = logs
                    if success:
                        st.success("Repositorio inicializado correctamente! Recarga la página.")
                        if st.button("Recargar"):
                            st.rerun()
                    else:
                        st.error("Hubo un error en la inicialización. Revisa los logs.")
        
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
    
    with st.expander("Configuración de Identidad Git (Nombre y Email)"):
        current_name, current_email = git_sync.get_git_user()
        col_name, col_email = st.columns(2)
        with col_name:
            new_name = st.text_input("Nombre de Usuario", value=current_name, help="Ej. Felipe Villar")
        with col_email:
            new_email = st.text_input("Email", value=current_email, help="Ej. felipe@example.com")
            
        if st.button("Guardar Configuración"):
            if new_name and new_email:
                success, msg = git_sync.configure_git_user(new_name, new_email)
                if success:
                    st.success(msg)
                    st.rerun()
                else:
                    st.error(msg)
            else:
                st.warning("Por favor ingresa ambos campos.")

    st.divider()

    col1, col2 = st.columns([1, 4])
    
    with col1:
        if st.button("🔄 Sync Now", type="primary", width='stretch'):
            st.session_state["sync_logs"] = [] # Clear previous logs
            
            with col2:
                st.subheader("Live Logs")
                log_placeholder = st.empty()
            
            logs = []
            with st.spinner("Synchronizing with GitHub..."):
                gen = git_sync.sync_with_github_stream()
                success = False
                try:
                    while True:
                        msg = next(gen)
                        logs.append(msg)
                        log_placeholder.code("\n".join(logs), language="text")
                except StopIteration as e:
                    success = e.value
                except Exception as e:
                    logs.append(f"Error inesperado: {e}")
                    log_placeholder.code("\n".join(logs), language="text")
                    success = False
            
            st.session_state["sync_logs"] = logs
            
            with col1:
                if success:
                    st.success("Success!")
                else:
                    st.error("Failed.")

    st.divider()
    st.subheader("Opciones Avanzadas")
    
    with st.expander("Restablecer copia desde Github (Hard Reset)", expanded=False):
        st.warning("⚠️ CUIDADO: Esta opción borrará TODOS sus cambios locales no guardados (commits pendientes o archivos modificados en el directorio de trabajo) y dejará la carpeta idéntica a `origin/main`.")
        
        if st.button("🚨 Restablecer copia desde Github", type="secondary", help="git fetch origin && git reset --hard origin/main"):
            st.session_state["sync_logs"] = []
            
            # Re-use log placeholder if possible, or create new one
            log_placeholder = st.empty()
            logs = []
            
            with st.spinner("Restableciendo repositorio..."):
                gen = git_sync.force_reset_stream()
                success = False
                try:
                    while True:
                        msg = next(gen)
                        logs.append(msg)
                        log_placeholder.code("\n".join(logs), language="text")
                except StopIteration as e:
                    success = e.value
                except Exception as e:
                    logs.append(f"Error inesperado: {e}")
                    log_placeholder.code("\n".join(logs), language="text")
                    success = False

            if success:
                st.success("Repositorio restablecido correctamente.")
                time.sleep(1)
                st.rerun()
            else:
                st.error("Hubo un error al restablecer. Revise los logs.")

    # Mostrar logs persistentes si no se está ejecutando (o después de ejecutarse)
    if "sync_logs" in st.session_state and st.session_state["sync_logs"]:
         pass
         
    # Para persistencia visual tras interacción
    # (Reuse valid logic if needed, but the main log viewer is above)

    with col2:
        if "sync_logs" in st.session_state and st.session_state["sync_logs"]:
             if not logs: # Si no estamos en el bucle de ejecución
                 st.subheader("Logs (Last Run)")
                 st.code("\n".join(st.session_state["sync_logs"]), language="text")

    if show_exit_button:
        if st.button("Exit"):
            st.stop()

if __name__ == "__main__":
    main(set_page_config=True)
