import os
import subprocess

os.makedirs("frontend/src", exist_ok=True)
os.makedirs("frontend/public", exist_ok=True)

with open("frontend/package.json", "w") as f:
    f.write('''{
  "name": "frontend",
  "private": true,
  "version": "0.0.0",
  "type": "module",
  "scripts": {
    "dev": "vite",
    "build": "tsc && vite build",
    "lint": "eslint .",
    "preview": "vite preview"
  },
  "dependencies": {
    "react": "^18.3.1",
    "react-dom": "^18.3.1",
    "lucide-react": "^0.378.0"
  },
  "devDependencies": {
    "@types/react": "^18.3.3",
    "@types/react-dom": "^18.3.0",
    "@vitejs/plugin-react": "^4.3.1",
    "typescript": "^5.2.2",
    "vite": "^5.3.4"
  }
}''')

with open("frontend/vite.config.ts", "w") as f:
    f.write('''import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vitejs.dev/config/
export default defineConfig({
  plugins: [react()],
})
''')

with open("frontend/tsconfig.json", "w") as f:
    f.write('''{
  "compilerOptions": {
    "target": "ES2020",
    "useDefineForClassFields": true,
    "lib": ["ES2020", "DOM", "DOM.Iterable"],
    "module": "ESNext",
    "skipLibCheck": true,
    "moduleResolution": "bundler",
    "allowImportingTsExtensions": true,
    "resolveJsonModule": true,
    "isolatedModules": true,
    "noEmit": true,
    "jsx": "react-jsx",
    "strict": true,
    "noUnusedLocals": true,
    "noUnusedParameters": true,
    "noFallthroughCasesInSwitch": true
  },
  "include": ["src"],
  "references": [{ "path": "./tsconfig.node.json" }]
}
''')

with open("frontend/tsconfig.node.json", "w") as f:
    f.write('''{
  "compilerOptions": {
    "composite": true,
    "skipLibCheck": true,
    "module": "ESNext",
    "moduleResolution": "bundler",
    "allowSyntheticDefaultImports": true,
    "strict": true
  },
  "include": ["vite.config.ts"]
}
''')

with open("frontend/index.html", "w") as f:
    f.write('''<!doctype html>
<html lang="en">
  <head>
    <meta charset="UTF-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <title>VISTA AI</title>
  </head>
  <body>
    <div id="root"></div>
    <script type="module" src="/src/main.tsx"></script>
  </body>
</html>
''')

with open("frontend/src/main.tsx", "w") as f:
    f.write('''import React from 'react'
import ReactDOM from 'react-dom/client'
import App from './App.tsx'
import './index.css'

ReactDOM.createRoot(document.getElementById('root')!).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>,
)
''')

with open("frontend/src/App.tsx", "w") as f:
    f.write('''import React from 'react'

function App() {
  return (
    <div>VISTA AI System Online</div>
  )
}

export default App
''')

with open("frontend/src/index.css", "w") as f:
    f.write('''body {
  margin: 0;
  padding: 0;
  background-color: #121212;
  color: #ffffff;
  font-family: Inter, system-ui, Avenir, Helvetica, Arial, sans-serif;
}
''')
    
print("Vite setup complete")
