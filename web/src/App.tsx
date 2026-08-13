import { BrowserRouter, Link, Route, Routes } from "react-router-dom";
import ProjectPage from "./pages/ProjectPage";
import ProjectsPage from "./pages/ProjectsPage";
import ReviewPage from "./pages/ReviewPage";
import SettingsPage from "./pages/SettingsPage";

function App() {
  return (
    <BrowserRouter>
      <header className="topbar">
        <Link to="/" className="logo">
          Nounbox
        </Link>
        <span className="muted">auto-labeling for datasets: detection and OCR</span>
        <Link to="/settings" className="topbar-link">
          Settings
        </Link>
      </header>
      <main className="container">
        <Routes>
          <Route path="/" element={<ProjectsPage />} />
          <Route path="/projects/:projectId" element={<ProjectPage />} />
          <Route
            path="/projects/:projectId/review/:imageId"
            element={<ReviewPage />}
          />
          <Route path="/settings" element={<SettingsPage />} />
        </Routes>
      </main>
    </BrowserRouter>
  );
}

export default App;
