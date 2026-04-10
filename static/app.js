/**
 * app.js - Core logic for AI Ally
 */

// --- Global State ---
let currentProjectId = null;

// --- Initialization ---
document.addEventListener('DOMContentLoaded', () => {
    // 1. Initial Data Load
    loadSystemStatus();
    loadProjects();

    // 2. Event Listeners (Wiring up the "Broken" buttons)
    
    // Create Project Button
    const createBtn = document.getElementById('btn-create-project');
    if (createBtn) createBtn.addEventListener('click', createProject);

    // Global Settings Button
    const globalSettingsBtn = document.getElementById('btn-global-settings');
    if (globalSettingsBtn) globalSettingsBtn.addEventListener('click', openGlobalSettings);

    // Project Settings Button
    const projectSettingsBtn = document.getElementById('btn-project-settings');
    if (projectSettingsBtn) projectSettingsBtn.addEventListener('click', openProjectSettings);

    // Enter key listener for the project name input
    const nameInput = document.getElementById('new-project-name');
    if (nameInput) {
        nameInput.addEventListener('keypress', (e) => {
            if (e.key === 'Enter') createProject();
        });
    }

    // Indexing Mode Radios (UI Toggles)
    document.querySelectorAll('input[name="ps-indexing-mode"]').forEach(radio => {
        radio.addEventListener('change', (e) => {
            const delimField = document.getElementById('delimiter-field');
            const chunkFields = document.getElementById('standard-chunking-fields');
            
            if (e.target.value === 'delimiter') {
                delimField.style.display = 'block';
                chunkFields.style.display = 'none';
            } else if (e.target.value === 'full_context') {
                delimField.style.display = 'none';
                chunkFields.style.display = 'none';
            } else {
                delimField.style.display = 'none';
                chunkFields.style.display = 'block';
            }
        });
    });
});

// --- Core Functions ---

async function loadSystemStatus() {
    const statusEl = document.getElementById('system-status');
    if (!statusEl) return;

    try {
        const res = await fetch('/api/status');
        const data = await res.json();
        
        if (data.embed_ok) {
            statusEl.textContent = 'System Ready';
            statusEl.className = 'ok'; 
        } else {
            statusEl.textContent = 'Ollama Offline';
            statusEl.className = 'err';
        }
    } catch (e) {
        statusEl.textContent = 'Connection Error';
        statusEl.className = 'err';
    }
}

async function loadProjects() {
    const list = document.getElementById('project-list');
    if (!list) return;

    try {
        const res = await fetch('/api/projects');
        const projects = await res.json();
        list.innerHTML = ''; 
        
        if (projects.length === 0) {
            list.innerHTML = '<li><span style="padding:0.75rem; display:block; color:var(--text-dim);">No projects found.</span></li>';
            return;
        }

        projects.forEach(proj => {
            const li = document.createElement('li');
            const btn = document.createElement('button');
            btn.className = 'proj-btn';
            btn.textContent = proj.name;
            btn.setAttribute('aria-label', `Select project ${proj.name}`);
            btn.addEventListener('click', () => selectProject(proj.id));
            li.appendChild(btn);
            list.appendChild(li);
        });
    } catch (e) {
        list.innerHTML = '<li><span style="padding:0.75rem; color:var(--error);">Failed to load projects.</span></li>';
    }
}

async function createProject() {
    const nameInput = document.getElementById('new-project-name');
    const name = nameInput.value.trim();

    if (!name) {
        alert("Please enter a project name.");
        return;
    }

    try {
        const res = await fetch('/api/projects', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ name: name })
        });

        if (res.ok) {
            const newProj = await res.json();
            nameInput.value = ''; 
            announceToScreenReader(`Project ${name} created successfully.`);
            await loadProjects(); 
            selectProject(newProj.id); 
        } else {
            const err = await res.json();
            alert("Error: " + err.error);
        }
    } catch (e) {
        alert("Failed to create project.");
    }
}

async function selectProject(id) {
    currentProjectId = id;
    
    // UI Visibility
    const noProjMsg = document.getElementById('no-project-msg');
    const chatArea = document.getElementById('chat-area');
    if (noProjMsg) noProjMsg.hidden = true;
    if (chatArea) chatArea.hidden = false;

    // Update Label
    const nameLabel = document.getElementById('current-project-name');
    if (nameLabel) nameLabel.textContent = `Project: ${id}`;
    
    announceToScreenReader(`Switched to project ${id}`);
}

// --- Placeholder Settings Logic ---

async function openGlobalSettings() {
    try {
        const res = await fetch('/api/settings');
        const data = await res.json();
        alert("Global Settings Loaded (Check Console for Data)");
        console.log(data);
    } catch (e) {
        alert("Could not load settings.");
    }
}

async function openProjectSettings() {
    if (!currentProjectId) return alert("Select a project first.");
    alert(`Opening settings for: ${currentProjectId}`);
}

// --- Accessibility Helper ---
function announceToScreenReader(message) {
    const announcer = document.getElementById('announcer');
    if (announcer) {
        announcer.textContent = message;
    }
}
