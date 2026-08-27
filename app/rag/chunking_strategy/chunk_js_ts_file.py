from pathlib import Path

from tree_sitter_language_pack import get_parser


LANGUAGE_BY_SUFFIX = {
    ".js": "javascript",
    ".jsx": "javascript",
    ".ts": "typescript",
    ".tsx": "tsx",
}

FUNCTION_VALUE_TYPES = {
    "arrow_function",
    "function_expression",
}

# ==================================================
# CHUNK JS / TS FILE
# ==================================================

def chunk_js_ts_file(path: Path):

    parser = get_parser(
        LANGUAGE_BY_SUFFIX[path.suffix]
    )

    source = path.read_bytes()

    tree = parser.parse(source)

    chunks = []

    def text(node):
        return source[
            node.start_byte:node.end_byte
        ].decode("utf-8", errors="ignore")

    def add_chunk(node, chunk_type, name, class_name=None):

        chunks.append({
            "code": text(node),
            "file": str(path),
            "type": chunk_type,
            "name": name,
            "class": class_name,
            "start_line": node.start_point[0] + 1,
            "end_line": node.end_point[0] + 1,
        })

    # -------------------------
    # Unwrap "export" / "export default"
    # -------------------------

    def unwrap_export(node):

        if node.type != "export_statement":
            return node

        declaration = node.child_by_field_name("declaration")

        if declaration is not None:
            return declaration

        # anonymous default export, e.g. `export default () => {}`
        return node.child_by_field_name("value")

    # -------------------------
    # Class + methods
    # -------------------------

    def chunk_class(node):

        name_node = node.child_by_field_name("name")

        class_name = (
            text(name_node) if name_node else "<anonymous>"
        )

        add_chunk(node, "class", class_name)

        body = node.child_by_field_name("body")

        if body is None:
            return

        for member in body.children:

            if member.type != "method_definition":
                continue

            method_name_node = member.child_by_field_name("name")

            method_name = (
                text(method_name_node)
                if method_name_node
                else "<anonymous>"
            )

            add_chunk(member, "method", method_name, class_name)

    # -------------------------
    # Function assigned to a const/let/var
    # -------------------------

    def chunk_lexical_declaration(node):

        for declarator in node.children:

            if declarator.type != "variable_declarator":
                continue

            value = declarator.child_by_field_name("value")

            if value is None or value.type not in FUNCTION_VALUE_TYPES:
                continue

            name_node = declarator.child_by_field_name("name")

            name = (
                text(name_node) if name_node else "<anonymous>"
            )

            add_chunk(declarator, "function", name)

    # -------------------------
    # Imports (grouped into a single chunk)
    # -------------------------

    import_nodes = [
        child for child in tree.root_node.children
        if child.type == "import_statement"
    ]

    if import_nodes:

        code = "\n".join(text(node) for node in import_nodes)

        chunks.append({
            "code": code,
            "file": str(path),
            "type": "imports",
            "name": path.name,
            "class": None,
            "start_line": import_nodes[0].start_point[0] + 1,
            "end_line": import_nodes[-1].end_point[0] + 1,
        })

    # -------------------------
    # Top-level statements only
    # -------------------------

    for top_level in tree.root_node.children:

        node = unwrap_export(top_level)

        if node is None:
            continue

        if node.type == "function_declaration":

            name_node = node.child_by_field_name("name")

            name = (
                text(name_node) if name_node else "<anonymous>"
            )

            add_chunk(node, "function", name)

        elif node.type == "class_declaration":

            chunk_class(node)

        elif node.type == "lexical_declaration":

            chunk_lexical_declaration(node)

        elif node.type in FUNCTION_VALUE_TYPES:

            # anonymous default export function/arrow
            add_chunk(node, "function", "default")

    return chunks
