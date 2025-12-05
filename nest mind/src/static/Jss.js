// async function fetchTree() {
//     const response = await fetch("/tree_json");
//     const data = await response.json();
//     const container = document.getElementById("tree-container");
//     container.innerHTML = "";
//     container.appendChild(renderNode(data));
// }

// function renderNode(node) {
//     const div = document.createElement("div");
//     div.className = "node";
//     div.innerHTML = `<strong>${node.title}</strong> <span class="node-id">[${node.id}]</span>`;
    
//     if (node.children && node.children.length > 0) {
//         const ul = document.createElement("ul");
//         node.children.forEach(child => {
//             const li = document.createElement("li");
//             li.appendChild(renderNode(child));
//             ul.appendChild(li);
//         });
//         div.appendChild(ul);
//     }
//     return div;
// }

// fetchTree();
