#!/usr/bin/env python3
"""Fix all remaining Python 3.10+ type annotations"""

import os
import re
from pathlib import Path

def fix_file(file_path):
    """Fix type annotations in a single file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        original_content = content
        
        # Fix various type annotation patterns
        patterns = [
            # Variable annotations: var: Union[type1, type2, type3]
            (r':\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*\|\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*\|\s*([a-zA-Z_][a-zA-Z0-9_]*)(?:\s*=\s*[^,\n]*)?', r': Union[\1, \2, \3]'),
            
            # Variable annotations: var: Union[type1, type2]
            (r':\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*\|\s*([a-zA-Z_][a-zA-Z0-9_]*)(?:\s*=\s*[^,\n]*)?', r': Union[\1, \2]'),
            
            # Function parameters: param: Union[type, None]
            (r':\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*\|\s*None\s*=\s*None', r': Optional[\1] = None'),
            
            # Function parameters: param: Union[type, None]
            (r':\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*\|\s*None(?:\s*=\s*[^,\n]*)?', r': Optional[\1]'),
            
            # Return types: -> Optional[type]
            (r'->\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*\|\s*None', r'-> Optional[\1]'),
            
            # Return types: -> Union[type1, type2]
            (r'->\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*\|\s*([a-zA-Z_][a-zA-Z0-9_]*)', r'-> Union[\1, \2]'),
            
            # List[type] -> List[type]
            (r'list\[([a-zA-Z_][a-zA-Z0-9_]*)\]', r'List[\1]'),
            
            # Dict[type1, type2] -> Dict[type1, type2]
            (r'dict\[([a-zA-Z_][a-zA-Z0-9_]*), ([a-zA-Z_][a-zA-Z0-9_]*)\]', r'Dict[\1, \2]'),
            
            # Tuple[type1, type2] -> Tuple[type1, type2]
            (r'tuple\[([a-zA-Z_][a-zA-Z0-9_]*), ([a-zA-Z_][a-zA-Z0-9_]*)\]', r'Tuple[\1, \2]'),
        ]
        
        for pattern, replacement in patterns:
            content = re.sub(pattern, replacement, content)
        
        if content != original_content:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f"✅ Fixed: {file_path}")
            return True
        else:
            return False
            
    except Exception as e:
        print(f"❌ Error processing {file_path}: {e}")
        return False

def main():
    """Fix all Python files."""
    print("🔧 Fixing remaining Python 3.10+ type annotations...")
    
    # Get all Python files
    python_files = list(Path('.').rglob("*.py"))
    
    fixed_count = 0
    
    for file_path in python_files:
        # Skip virtual environment and cache directories
        if 'venv' in str(file_path) or '__pycache__' in str(file_path):
            continue
            
        if fix_file(str(file_path)):
            fixed_count += 1
    
    print(f"\n📊 Summary:")
    print(f"Files fixed: {fixed_count}")
    print(f"Total Python files processed: {len(python_files)}")
    
    if fixed_count > 0:
        print("\n🎉 Type annotation fixes completed!")
    else:
        print("\n✅ No more type annotation fixes were needed.")

if __name__ == "__main__":
    main()
