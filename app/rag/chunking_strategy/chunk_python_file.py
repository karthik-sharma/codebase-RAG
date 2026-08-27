import ast
from pathlib import Path


def chunk_python_file(path: Path):

    source = path.read_text(
        encoding="utf-8",
        errors="ignore",
    )

    tree = ast.parse(source)

    lines = source.splitlines()

    chunks = []

    # -------------------------
    # Imports (grouped into a single chunk)
    # -------------------------

    import_nodes = [
        node for node in tree.body
        if isinstance(node, (ast.Import, ast.ImportFrom))
    ]

    if import_nodes:

        code = "\n".join(
            "\n".join(lines[node.lineno - 1:node.end_lineno])
            for node in import_nodes
        )

        chunks.append({
            "code": code,
            "file": str(path),
            "type": "imports",
            "name": path.name,
            "class": None,
            "start_line": import_nodes[0].lineno,
            "end_line": import_nodes[-1].end_lineno,
        })

    for node in tree.body:

        # -------------------------
        # Top-level function
        # -------------------------

        if isinstance(
            node,
            (ast.FunctionDef, ast.AsyncFunctionDef),
        ):

            start_line = node.lineno
            end_line = node.end_lineno

            code = "\n".join(
                lines[start_line - 1:end_line]
            )

            chunks.append({
                "code": code,
                "file": str(path),
                "type": "function",
                "name": node.name,
                "class": None,
                "start_line": start_line,
                "end_line": end_line,
            })


        # -------------------------
        # Class
        # -------------------------

        elif isinstance(node, ast.ClassDef):

            # Add the class itself

            start_line = node.lineno
            end_line = node.end_lineno

            code = "\n".join(
                lines[start_line - 1:end_line]
            )

            chunks.append({
                "code": code,
                "file": str(path),
                "type": "class",
                "name": node.name,
                "class": None,
                "start_line": start_line,
                "end_line": end_line,
            })


            # -------------------------
            # Methods inside class
            # -------------------------

            for child in node.body:

                if isinstance(
                    child,
                    (
                        ast.FunctionDef,
                        ast.AsyncFunctionDef,
                    ),
                ):

                    start_line = child.lineno
                    end_line = child.end_lineno

                    code = "\n".join(
                        lines[start_line - 1:end_line]
                    )

                    chunks.append({
                        "code": code,
                        "file": str(path),
                        "type": "method",
                        "name": child.name,
                        "class": node.name,
                        "start_line": start_line,
                        "end_line": end_line,
                    })

    return chunks