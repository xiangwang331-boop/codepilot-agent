import { StrictMode } from "react";
import { createRoot } from "react-dom/client";

import App from "./App";
import "./styles/tokens.css";
import "./styles/layout.css";
import "./styles/timeline.css";
import "./styles/panels.css";

const root = document.getElementById("root");
if (root === null) {
  throw new Error("找不到 #root 挂载点");
}

createRoot(root).render(
  <StrictMode>
    <App />
  </StrictMode>,
);
