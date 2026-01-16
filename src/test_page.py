import streamlit as st
import pytest
import sys
from io import StringIO
import contextlib
import os

def get_test_files():
    """Discover test files in the tests directory."""
    test_files = []
    tests_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "tests")
    if os.path.exists(tests_dir):
        for f in os.listdir(tests_dir):
            if f.startswith("test_") and f.endswith(".py"):
                test_files.append(f)
    return sorted(test_files)

def run_tests():
    st.title("Test Runner")
    st.write("Select the tests you want to execute:")
    
    test_files = get_test_files()
    selected_tests = []
    
    # Create checkboxes for each test file
    for test_file in test_files:
        if st.checkbox(test_file, value=True, key=test_file):
            selected_tests.append(test_file)
            
    if st.button("Run Selected Tests"):
        if not selected_tests:
            st.warning("Please select at least one test file.")
            return
            
        st.write("---")
        st.subheader("Test Execution Progress")
        
        # Container for results
        results_container = st.container()
        
        # Summary metrics
        total = len(selected_tests)
        passed = 0
        failed = 0
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i, test_file in enumerate(selected_tests):
            # Update status
            current_progress = (i) / total
            progress_bar.progress(current_progress)
            status_text.text(f"Running {test_file}...")
            
            # Create a placeholder for this test's result
            test_result_expander = results_container.expander(f"⏳ Running: {test_file}", expanded=True)
            
            # Capture output
            output_buffer = StringIO()
            with contextlib.redirect_stdout(output_buffer):
                # Run pytest on the specific file
                # Use -q for quiet output, just the summary mostly
                retcode = pytest.main(["-v", f"tests/{test_file}"])
            
            test_output = output_buffer.getvalue()
            
            # Update the expander with final status
            if retcode == 0:
                test_result_expander.success(f"✅ PASS: {test_file}")
                passed += 1
            else:
                test_result_expander.error(f"❌ FAIL: {test_file}")
                test_result_expander.code(test_output, language="text")
                failed += 1
                
        # Final status
        progress_bar.progress(1.0)
        status_text.text("Execution complete.")
        
        st.write("---")
        st.subheader("Summary")
        col1, col2, col3 = st.columns(3)
        col1.metric("Total", total)
        col2.metric("Passed", passed)
        col3.metric("Failed", failed)
        
        if failed == 0:
            st.success("🎉 All selected tests passed!")
        else:
            st.error(f"⚠️ {failed} tests failed.")

def main(set_page_config=False, show_exit_button=False):
    if set_page_config:
        st.set_page_config(page_title="Test Runner", layout="wide")
    
    run_tests()

if __name__ == "__main__":
    main(set_page_config=True)
