import json
import os

def extract_notebook_to_markdown(notebook_path, output_path):
    if not os.path.exists(notebook_path):
        print(f"Error: File {notebook_path} not found.")
        return

    with open(notebook_path, 'r', encoding='utf-8') as f:
        nb_data = json.load(f)

    markdown_content = []
    
    # Iterate through cells in chronological order
    for cell in nb_data.get('cells', []):
        cell_type = cell.get('cell_type')
        source = "".join(cell.get('source', []))
        
        if cell_type == 'markdown':
            # Markdown cells are added directly
            markdown_content.append(source)
            markdown_content.append("\n\n")
        elif cell_type == 'code':
            # Code cells are wrapped in python code blocks
            markdown_content.append("```python\n")
            markdown_content.append(source)
            if not source.endswith('\n'):
                markdown_content.append("\n")
            markdown_content.append("```\n\n")
            
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("".join(markdown_content))
    
    print(f"Extraction complete! Content written to: {output_path}")

if __name__ == "__main__":
    # Path to the specific notebook mentioned by the user
    notebook_file = "PROJET_RENDU.ipynb"
    output_file = "PROJET_RENDU_EXTRACTION.md"
    
    extract_notebook_to_markdown(notebook_file, output_file)
