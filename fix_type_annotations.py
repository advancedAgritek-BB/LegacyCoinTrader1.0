#!/usr/bin/env python3
"""
Script to fix Python 3.10+ type annotations to be compatible with Python 3.9.
This converts the | union operator syntax to Union[] syntax.
"""

import os
import re
import sys
from pathlib import Path

def fix_type_annotations_in_file(file_path: str) -> bool:
    """Fix type annotations in a single file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        original_content = content
        
        # Fix simple union types: Optional[type] -> Optional[type]
        content = re.sub(r':\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*\|\s*None\s*=', r': Optional[\1] =', content)
        content = re.sub(r':\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*\|\s*None\s*\)', r': Optional[\1])', content)
        content = re.sub(r':\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*\|\s*None\s*->', r': Optional[\1] ->', content)
        
        # Fix complex union types: Union[type1, type2] -> Union[type1, type2]
        content = re.sub(r':\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*\|\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*=', r': Union[\1, \2] =', content)
        content = re.sub(r':\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*\|\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*\)', r': Union[\1, \2])', content)
        content = re.sub(r':\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*\|\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*->', r': Union[\1, \2] ->', content)
        
        # Fix more complex patterns
        content = re.sub(r':\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*\|\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*\|\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*=', r': Union[\1, \2, \3] =', content)
        content = re.sub(r':\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*\|\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*\|\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*\)', r': Union[\1, \2, \3])', content)
        
        # Fix List[type] -> List[type] (Python 3.9 compatibility)
        content = re.sub(r'list\[([a-zA-Z_][a-zA-Z0-9_]*)\]', r'List[\1]', content)
        content = re.sub(r'dict\[([a-zA-Z_][a-zA-Z0-9_]*), ([a-zA-Z_][a-zA-Z0-9_]*)\]', r'Dict[\1, \2]', content)
        
        if content != original_content:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f"✅ Fixed: {file_path}")
            return True
        else:
            print(f"⏭️  No changes needed: {file_path}")
            return False
            
    except Exception as e:
        print(f"❌ Error processing {file_path}: {e}")
        return False

def ensure_typing_imports(file_path: str) -> bool:
    """Ensure that Union and Optional are imported if needed."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Check if Union or Optional are used but not imported
        needs_union = 'Union[' in content
        needs_optional = 'Optional[' in content
        
        if needs_union or needs_optional:
            # Check current imports
            if 'from typing import' in content:
                # Add to existing typing import
                if 'Union' not in content and needs_union:
                    content = re.sub(r'from typing import ([^,\n]+)', r'from typing import \1, Union', content)
                if 'Optional' not in content and needs_optional:
                    content = re.sub(r'from typing import ([^,\n]+)', r'from typing import \1, Optional', content)
            else:
                # Add new typing import
                import_line = 'from typing import'
                if needs_union:
                    import_line += ' Union'
                if needs_optional:
                    import_line += ' Optional'
                if needs_union and needs_optional:
                    import_line += ', Optional'
                
                # Find the first import line and add before it
                lines = content.split('\n')
                for i, line in enumerate(lines):
                    if line.strip().startswith('import ') or line.strip().startswith('from '):
                        lines.insert(i, import_line)
                        break
                content = '\n'.join(lines)
            
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f"✅ Added typing imports: {file_path}")
            return True
            
    except Exception as e:
        print(f"❌ Error adding typing imports to {file_path}: {e}")
        return False
    
    return False

def main():
    """Main function to fix type annotations in all Python files."""
    print("🔧 Fixing Python 3.10+ type annotations for Python 3.9 compatibility...")
    
    # Get the project root directory
    project_root = Path(__file__).parent
    python_files = list(project_root.rglob("*.py"))
    
    print(f"Found {len(python_files)} Python files")
    
    fixed_count = 0
    typing_fixed_count = 0
    
    for file_path in python_files:
        # Skip virtual environment and cache directories
        if 'venv' in str(file_path) or '__pycache__' in str(file_path):
            continue
            
        if fix_type_annotations_in_file(str(file_path)):
            fixed_count += 1
            
        if ensure_typing_imports(str(file_path)):
            typing_fixed_count += 1
    
    print(f"\n📊 Summary:")
    print(f"Files with type annotations fixed: {fixed_count}")
    print(f"Files with typing imports added: {typing_fixed_count}")
    print(f"Total Python files processed: {len(python_files)}")
    
    if fixed_count > 0 or typing_fixed_count > 0:
        print("\n🎉 Type annotation fixes completed!")
        print("You can now try running the compatibility test again.")
    else:
        print("\n✅ No type annotation fixes were needed.")

if __name__ == "__main__":
    main()
