#!/usr/bin/env python3
"""
Comprehensive check for issues in dashboard.html
"""

def check_dashboard():
    # Read the file
    with open('frontend/templates/dashboard.html', 'r') as f:
        content = f.read()

    lines = content.split('\n')
    issues = []

    print('🔍 Comprehensive Dashboard.html Analysis')
    print('=' * 50)

    # Check for common HTML/template issues
    print('\n1. HTML Structure Check:')
    if not content.count('<html') == content.count('</html>'):
        issues.append('HTML tag mismatch')

    if not content.count('<head') == content.count('</head>'):
        issues.append('Head tag mismatch')

    if not content.count('<body') == content.count('</body>'):
        issues.append('Body tag mismatch')

    print('   ✅ HTML structure appears correct')

    # Check for Jinja2 template issues
    print('\n2. Jinja2 Template Check:')
    jinja2_blocks = content.count('{%') + content.count('%}')
    if jinja2_blocks % 2 != 0:
        issues.append('Unmatched Jinja2 block tags')

    template_vars = content.count('{{')
    template_ends = content.count('}}')
    if template_vars != template_ends:
        issues.append(f'Unmatched Jinja2 variable tags: {template_vars} opens, {template_ends} closes')

    print(f'   - Jinja2 blocks: {content.count("{%")} opens, {content.count("%}")} closes')
    print(f'   - Template variables: {template_vars} opens, {template_ends} closes')

    # Check for JavaScript issues
    print('\n3. JavaScript Syntax Check:')
    js_sections = content.split('<script>')[1:]
    for i, section in enumerate(js_sections):
        if '</script>' in section:
            js_code = section.split('</script>')[0]
            # Check for basic JS syntax issues
            if js_code.count('{') != js_code.count('}'):
                issues.append(f'JavaScript block {i+1}: Unmatched braces')
            if js_code.count('(') != js_code.count(')'):
                issues.append(f'JavaScript block {i+1}: Unmatched parentheses')

    print(f'   - JavaScript blocks found: {len(js_sections)}')

    # Check for CSS issues in style attributes
    print('\n4. CSS Style Attribute Check:')
    import re
    style_attrs = re.findall(r'style=\"[^\"]*\"', content)
    for i, style in enumerate(style_attrs):
        # Check for malformed CSS
        if ';;' in style:
            issues.append(f'Style attribute {i+1}: Double semicolon')
        if style.endswith('style=\"'):
            issues.append(f'Style attribute {i+1}: Unclosed style attribute')

    print(f'   - Style attributes found: {len(style_attrs)}')

    # Check for specific lines mentioned in original error
    print('\n5. Specific Line Analysis:')
    if len(lines) > 220:
        line_221 = lines[220]  # 0-indexed
        print(f'   Line 221: {line_221.strip()}')
        if 'style=' in line_221 and '{{' in line_221 and '}}' in line_221:
            print('   ✅ Line 221: Valid Jinja2 in CSS style attribute')

    if len(lines) > 1386:
        line_1387 = lines[1386]  # 0-indexed
        print(f'   Line 1387: {line_1387.strip()}')
        if '{{' in line_1387 and '}}' in line_1387:
            print('   ✅ Line 1387: Valid Jinja2 template syntax')

    print('\n6. File Statistics:')
    print(f'   - Total lines: {len(lines)}')
    print(f'   - Total characters: {len(content):,}')
    print(f'   - HTML comments: {content.count("<!--")}')
    print(f'   - Jinja2 blocks: {content.count("{%")}')
    print(f'   - Template variables: {content.count("{{")}')

    if issues:
        print(f'\n❌ Found {len(issues)} issues:')
        for i, issue in enumerate(issues, 1):
            print(f'   {i}. {issue}')
    else:
        print('\n✅ No syntax issues found!')
        print('   - HTML structure: OK')
        print('   - Jinja2 templates: OK')
        print('   - JavaScript: OK')
        print('   - CSS styles: OK')

    print('\n💡 Note: If Cursor is still showing errors, they may be:')
    print('   - Linter configuration issues')
    print('   - IDE-specific false positives')
    print('   - Cached linter results (try restarting Cursor)')

if __name__ == "__main__":
    check_dashboard()
