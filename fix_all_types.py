#!/usr/bin/env python3
"""Fix all remaining type annotations using simple string replacement"""

import os
from pathlib import Path

def fix_file(file_path):
    """Fix type annotations in a single file using simple string replacement."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        original_content = content
        
        # Simple string replacements for common patterns
        replacements = [
            # Variable annotations
            ('Union[pd.DataFrame, list, None]', 'Union[pd.DataFrame, list, None]'),
            ('Optional[pd.DataFrame]', 'Optional[pd.DataFrame]'),
            ('Optional[asyncio.Task]', 'Optional[asyncio.Task]'),
            ('Optional[threading.Thread]', 'Optional[threading.Thread]'),
            ('Optional[asyncio.subprocess.Process]', 'Optional[asyncio.subprocess.Process]'),
            ('Optional[asyncio.AbstractEventLoop]', 'Optional[asyncio.AbstractEventLoop]'),
            ('Optional[Mapping[str, Any]]', 'Optional[Mapping[str, Any]]'),
            ('Optional[List[str]]', 'Optional[List[str]]'),
            ('Optional[Dict[str, float]]', 'Optional[Dict[str, float]]'),
            ('Optional[Dict[str, pd.DataFrame]]', 'Optional[Dict[str, pd.DataFrame]]'),
            ('Mapping[str, pd.DataFrame] | Optional[pd.DataFrame]', 'Optional[Union[Mapping[str, pd.DataFrame], pd.DataFrame]]'),
            ('RouterConfig | Optional[Mapping[str, Any]]', 'Optional[Union[RouterConfig, Mapping[str, Any]]]'),
            ('Optional[Union[RouterConfig, dict]]', 'Optional[Union[RouterConfig, dict]]'),
            ('Optional[Mapping[str, int]]', 'Optional[Mapping[str, int]]'),
            ('Optional[Callable[[pd.DataFrame], Tuple[float, str]]]', 'Optional[Callable[[pd.DataFrame], Tuple[float, str]]]'),
            ('Optional[Callable[[pd.DataFrame], tuple]]', 'Optional[Callable[[pd.DataFrame], tuple]]'),
            ('Optional[Tuple[str, float]]', 'Optional[Tuple[str, float]]'),
            ('Optional[Union[list, dict]]', 'Optional[Union[list, dict]]'),
            ('Optional[Union[Tuple[list, float], tuple[list, float, float]]]', 'Optional[Union[Tuple[list, float], tuple[list, float, float]]]'),
            ('Optional[Union[Iterable[str], str, Any]]', 'Optional[Union[Iterable[str], str, Any]]'),
            ('Optional[Union[Iterable[str], str]]', 'Optional[Union[Iterable[str], str]]'),
            ('Optional[Sequence[dict]]', 'Optional[Sequence[dict]]'),
            ('List[Optional[int]]', 'List[Optional[int]]'),
            ('Optional[Dict[str, pd.DataFrame]]', 'Optional[Dict[str, pd.DataFrame]]'),
            ('Optional[aiohttp.ClientSession]', 'Optional[aiohttp.ClientSession]'),
            ('"Optional[TradingBotController]"', '"Optional[TradingBotController]"'),
            ('Optional[List[str]] = None', 'Optional[List[str]] = None'),
        ]
        
        for old, new in replacements:
            content = content.replace(old, new)
        
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
    print("🔧 Fixing all remaining type annotations...")
    
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
